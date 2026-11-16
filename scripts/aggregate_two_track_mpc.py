"""Summarize completed goal-paired MPC runs separately by research track."""

import argparse
import csv
from pathlib import Path

from scripts.aggregate_gradient_mpc import read_run, wilson_interval


ARMS = {
    "cross_color_jepa": ("same_fixed", "same_window_random", "cross_fixed", "cross_window_random"),
    "mi_jepa": ("default_jepa_baseline", "default_jepa_plus_inverse"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_root", type=Path)
    args = parser.parse_args()
    for track, arms in ARMS.items():
        rows = []
        for arm in arms:
            for condition in ("default", "purpleBlock", "redBlock"):
                folder = args.out_root / track / "eval" / arm / condition
                run = read_run(folder)
                if run is None:
                    rows.append((track, arm, condition, "N/A", "N/A", "N/A", "N/A"))
                    continue
                winners, last = run
                low, high = wilson_interval(len(winners), len(last))
                rows.append((track, arm, condition, len(winners), 100 * len(winners) / len(last), low, high))
                with (folder / "target_outcomes.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("target_index", "success", "last_mpc_step"))
                    for index, result in sorted(last.items()):
                        writer.writerow((index, bool(result["success"]), int(result["step"])))
        report = args.out_root / track / "mpc_summary.csv"
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("track", "arm", "condition", "successes_of_50", "rate_pct",
                             "wilson95_low_pct", "wilson95_high_pct"))
            writer.writerows(rows)
        print("Saved {}".format(report))


if __name__ == "__main__":
    main()
