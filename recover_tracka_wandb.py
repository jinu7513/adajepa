"""Upload durable local results to a NEW recovery run; original runs are untouched."""
import argparse
import json
from pathlib import Path

from tracka.logging import RunLogger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cfg = dict(metadata["config"]["logging"])
    cfg.update(mode=args.mode, fallback_to_local=False)
    cfg["project"] = args.project or cfg["project"]
    cfg["entity"] = args.entity or cfg["entity"]
    cfg["name"] = (cfg["name"] or args.run_dir.name) + "-recovered"
    import tempfile
    output = Path(tempfile.mkdtemp(prefix="wandb_recovery_", dir=args.run_dir))
    metadata["recovered_from"] = str(args.run_dir.resolve())
    with RunLogger(output, cfg, metadata) as logger:
        count = 0
        with open(args.run_dir / "metrics.jsonl", encoding="utf-8") as f:
            for line in f:
                logger.log(json.loads(line))
                count += 1
        for name in ("results", "diagnostics"):
            path = args.run_dir / (name + ".json")
            if path.exists():
                logger.table(name, json.loads(path.read_text(encoding="utf-8")))
        if (args.run_dir / "robustness.png").exists() and (args.run_dir / "results.json").exists():
            logger.robustness_plots(json.loads((args.run_dir / "results.json").read_text(encoding="utf-8")),
                                    args.run_dir / "robustness.png")
        destination = "offline recovery files" if args.mode == "offline" else "W&B recovery run"
        print(f"Recorded {count} metric records in {destination}: {logger.run_id}")
        if logger.error:
            raise RuntimeError("Recovery had W&B errors; local data retained: " + logger.error)


if __name__ == "__main__":
    main()
