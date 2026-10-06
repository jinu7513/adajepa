"""Verify a finished RGB/Event dataset, then run a two-update W&B smoke test.

This script never generates data or starts a full training run.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parent


def inspect_dataset(root):
    required = ("manifest.json", "generation_report.json", "samples.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(
            f"Dataset generation is not finished at {root}; missing: {', '.join(missing)}"
        )

    report = json.loads((root / "generation_report.json").read_text(encoding="utf-8"))

    # Use the training loader's own integrity checks before starting any optimizer step.
    from tracka.event_data import validate_event_windows

    manifest, rows = validate_event_windows(root)
    if manifest.get("schema_version") != "tracka-event-window-v1":
        raise ValueError("Unexpected Event dataset schema")

    count = len(rows)
    split_counts = Counter(row["split"] for row in rows)
    shift_counts = Counter(row["shift_kind"] for row in rows)
    expected_splits = ("train", "validation", "test")
    if any(split_counts[split] == 0 for split in expected_splits):
        raise ValueError(f"Dataset has an empty split: {dict(split_counts)}")
    if sum(split_counts.values()) != count:
        raise ValueError("Dataset split counts do not sum to the sample count")
    if report.get("accepted") != count or report.get("rejected") != manifest.get("rejected_count"):
        raise ValueError("Generation report and manifest disagree")
    if any(report.get("accepted_by_split", {}).get(split) != split_counts[split]
           for split in expected_splits):
        raise ValueError("Generation report split counts disagree with samples")
    if any(report.get("accepted_by_shift", {}).get(kind) != n
           for kind, n in shift_counts.items()):
        raise ValueError("Generation report shift counts disagree with samples")
    image_count = sum(1 for _ in (root / "images").glob("*.png"))
    if image_count != manifest["image_count"]:
        raise ValueError(
            f"Expected {manifest['image_count']} PNG images; found {image_count}"
        )

    print(f"Dataset OK: {root}", flush=True)
    print(f"Accepted: {count}; rejected: {report['rejected']}; images: {image_count}", flush=True)
    print(f"Splits: {dict(split_counts)}; shifts: {dict(shift_counts)}", flush=True)
    return manifest


def run_smoke(root):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = REPO / "tracka_outputs" / f"event_smoke_{stamp}_{uuid.uuid4().hex[:6]}"
    name = f"trackA-event-fixed-sum-smoke-{stamp}"
    command = [
        sys.executable, str(REPO / "train_event_encoder.py"),
        "--config-name", "tracka_event",
        f"dataset.output_path={root}",
        "training.total_optimizer_updates=2",
        "training.batch_size=2",
        "training.rank_samples=8",
        "training.validation_batches=1",
        "training.validate_every=2",
        "training.save_every=2",
        "logging.log_every=1",
        f"training.output_dir={output}",
        f"logging.name={name}",
        "logging.fallback_to_local=false",
    ]
    print(f"Starting exactly two optimizer updates; output: {output}", flush=True)
    subprocess.run(command, cwd=REPO, check=True)

    status_path = output / "run_status.json"
    checkpoint_path = output / "checkpoint_latest.pt"
    if not status_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError(f"Smoke run did not save status/checkpoint in {output}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "finished" or status.get("wandb_error"):
        raise RuntimeError(f"Smoke run or W&B logging failed: {status}")
    records = [json.loads(line) for line in (output / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    if not any(record.get("train/global_step") == 2 and "loss/total" in record
               for record in records):
        raise RuntimeError("Smoke run is missing step-2 loss metrics")
    identity_path = output / "wandb_identity.json"
    if not identity_path.is_file():
        raise RuntimeError("Smoke run has no W&B identity")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("mode") != "online" or not all(
            identity.get(key) for key in ("entity", "project", "id")):
        raise RuntimeError(f"W&B did not start an online run: {identity}")
    print(f"Smoke OK: {output}", flush=True)
    print("W&B: https://wandb.ai/{entity}/{project}/runs/{id}".format(**identity),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/event_windows",
                        help="Completed Event-window dataset (default: data/event_windows)")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate and summarize the dataset without training")
    args = parser.parse_args()
    root = Path(args.dataset).expanduser()
    root = (root if root.is_absolute() else REPO / root).resolve()
    inspect_dataset(root)
    if not args.check_only:
        run_smoke(root)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Event smoke stopped: {exc}") from exc
