"""Summarize completed frozen/AdaJEPA/adapter MPC cases from JSONL logs."""

import csv
import json
import sys
from pathlib import Path


def summarize(root):
    root = Path(root)
    summary = []
    curves = []
    for done in sorted((root / "runs").glob("*/*/done.txt")):
        condition, method = done.parent.parent.name, done.parent.name
        attempt = Path(done.read_text(encoding="utf-8").strip())
        path = attempt / "logs.json"
        paper = [json.loads(line) for line in path.open(encoding="utf-8")
                 if '"paper/mpc_step"' in line]
        if not paper:
            raise ValueError(f"No success curve in {path}")
        for row in paper:
            curves.append({"condition": condition, "method": method,
                           "mpc_step": int(row["paper/mpc_step"]),
                           "success_rate_pct": float(row["paper/success_rate_pct"])})
        summary.append({"condition": condition, "method": method,
                        "final_step": curves[-1]["mpc_step"],
                        "final_success_rate_pct": curves[-1]["success_rate_pct"],
                        "run_dir": str(attempt)})
    for name, fields, rows in (
        ("summary.csv", ("condition", "method", "final_step", "final_success_rate_pct", "run_dir"), summary),
        ("curves.csv", ("condition", "method", "mpc_step", "success_rate_pct"), curves),
    ):
        with (root / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize_latent_adapter_mpc.py OUT_ROOT")
    for row in summarize(sys.argv[1]):
        print(row)
