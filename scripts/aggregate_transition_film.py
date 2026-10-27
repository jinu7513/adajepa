"""Aggregate the Slurm A/B/C matrix into a paper-style success curve."""

import argparse
import csv
import json
from pathlib import Path


def read_curve(path):
    points = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "paper/mpc_step" in row and "paper/success_rate_pct" in row:
            points[int(row["paper/mpc_step"])] = float(row["paper/success_rate_pct"])
    if not points:
        raise ValueError(f"No paper success-rate curve in {path}")
    if any(step < 1 or step > 30 for step in points):
        raise ValueError(f"Unexpected MPC step in {path}: {sorted(points)}")
    return points


def aggregate(root):
    curves = {}
    for variant in "ABC":
        for condition in ("default", "redBlock"):
            case = root / "eval" / variant / condition
            marker = case / "done.txt"
            if not marker.is_file():
                raise FileNotFoundError(f"Evaluation not completed: {case}")
            attempt = Path(marker.read_text(encoding="utf-8").strip())
            if attempt.parent.resolve() != case.resolve():
                raise ValueError(f"Invalid evaluation marker: {marker}")
            curves[(variant, condition)] = read_curve(attempt / "logs.json")
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=True)
    long_rows = []
    summary = []
    for (variant, condition), points in curves.items():
        last_value = 0.0
        for step in range(1, 31):
            last_value = points.get(step, last_value)
            long_rows.append({
                "variant": variant, "condition": condition, "seed": 100,
                "n_evals": 50, "mpc_step": step,
                "success_rate_pct": last_value,
            })
        summary.append({"variant": variant, "condition": condition,
                        "final_success_rate_pct": last_value})
    with (output / "success_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_rows[0].keys())
        writer.writeheader()
        writer.writerows(long_rows)
    with (output / "final_success.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    labels = {"A": "A: released clean", "B": "B: color-trained", "C": "C: transition-FiLM"}
    for ax, condition in zip(axes, ("default", "redBlock")):
        for variant in "ABC":
            values = [row["success_rate_pct"] for row in long_rows
                      if row["variant"] == variant and row["condition"] == condition]
            ax.plot(range(1, 31), values, label=labels[variant], linewidth=2)
        ax.set_title(condition)
        ax.set_xlabel("MPC step")
        ax.grid(alpha=0.25)
        ax.set_xlim(1, 30)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Success rate (%)")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    figure_path = output / "success_vs_mpc_step.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"Saved {output / 'success_curve.csv'} and {figure_path}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    aggregate(parser.parse_args().run_root.resolve())
