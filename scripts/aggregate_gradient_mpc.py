"""Aggregate paired per-target P/J MPC outcomes without hiding missing runs."""

import argparse
import csv
import json
import math
from pathlib import Path


def wilson_interval(successes, total, z=1.96):
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return 100 * (center - radius), 100 * (center + radius)


def read_run(path, total=50):
    marker = path / "done.txt"
    if not marker.is_file():
        return None
    attempt = marker.read_text().strip()
    if attempt and attempt != "complete":
        path = Path(attempt)
    last = {}
    for line in (path / "per_iter_logs.json").open():
        rec = json.loads(line)
        last[int(rec["sample_idx"])] = rec
    if sorted(last) != list(range(total)):
        raise ValueError("Incomplete target indices in {}".format(path))
    winners = {i for i, rec in last.items() if rec["success"]}
    return winners, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    rows = []
    for condition, split in (("default", "ID"), ("purpleBlock", "validation_color"),
                             ("redBlock", "test_color")):
        outcomes = {}
        for mode in ("P", "J"):
            path = args.run_root / "eval" / mode / condition
            run = read_run(path)
            if run is None:
                rows.append({"mode": mode, "condition": condition, "split": split,
                             "successes": "N/A", "total": 50, "rate_pct": "N/A",
                             "ci95_low_pct": "N/A", "ci95_high_pct": "N/A"})
                continue
            winners, last = run
            outcomes[mode] = winners
            low, high = wilson_interval(len(winners), len(last))
            rows.append({"mode": mode, "condition": condition, "split": split,
                         "successes": len(winners), "total": len(last),
                         "rate_pct": 100 * len(winners) / len(last),
                         "ci95_low_pct": low, "ci95_high_pct": high})
            target_path = args.run_root / "eval" / mode / condition / "target_outcomes.csv"
            with target_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("target_index", "success", "last_mpc_step"))
                writer.writeheader()
                for i, rec in sorted(last.items()):
                    writer.writerow({"target_index": i, "success": bool(rec["success"]),
                                     "last_mpc_step": int(rec["step"])})
        if len(outcomes) == 2:
            print("{}: shared={} P-only={} J-only={}".format(
                condition, len(outcomes["P"] & outcomes["J"]),
                sorted(outcomes["P"] - outcomes["J"]),
                sorted(outcomes["J"] - outcomes["P"])))
    output = args.run_root / "mpc_summary.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Saved {}".format(output))


if __name__ == "__main__":
    main()
