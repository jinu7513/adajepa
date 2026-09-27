"""Collect probe runs into a comparable CSV and a standalone robustness figure."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, signatures = [], set()
    experiments = {}
    for folder in args.run_dirs:
        metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        cfg = metadata["config"]["evaluation"]
        signatures.add(json.dumps({"manifest": metadata["manifest_sha256"],
                                   "seed": cfg["seed"], "grid": cfg["corruption_grid"],
                                   "outside": cfg["outside_color_sampling"],
                                   "cap": cfg["max_states_per_split"],
                                   "alphas": cfg["ridge_alphas"]}, sort_keys=True))
        specification = {k: metadata.get(k) for k in ("training_objective", "training_model", "training_decoder",
                         "training_mask", "training_loss", "training_updates")}
        spec_key = json.dumps(specification, sort_keys=True)
        if spec_key not in experiments:
            experiments[spec_key] = metadata["training_objective"] + f" [config {len(experiments)+1}]"
        label = experiments[spec_key]
        for row in json.loads((folder / "results.json").read_text(encoding="utf-8")):
            rows.append({"objective": metadata["training_objective"],
                         "experiment": label,
                         "training_seed": metadata.get("training_seed", metadata["config"]["evaluation"]["seed"]),
                         "run_dir": str(folder), **row})
    if len(signatures) != 1:
        raise ValueError("Runs have different dataset or evaluation protocols; do not pool them")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "combined.csv"
    if target.exists():
        raise FileExistsError("Use a fresh aggregation output directory")
    with open(target, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = ["default", "colorA", "colorB", "redBlock", "redAgent", "redAnchor", "random_inside", "random_outside"]
    objectives = sorted({r["experiment"] for r in rows})
    fig, ax = plt.subplots(figsize=(13, 5))
    width = .8 / len(objectives)
    for j, objective in enumerate(objectives):
        means, stds = [], []
        for condition in conditions:
            entries = [r for r in rows if r["experiment"] == objective and r["condition"] == condition and r["segment"] == "full"]
            if len({r["training_seed"] for r in entries}) != len(entries):
                raise ValueError("Duplicate seed for the same objective; select one checkpoint per seed")
            values = [r["standardized_mse"] for r in entries]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0)
        ax.bar(np.arange(len(conditions)) + (j - (len(objectives)-1)/2) * width,
               means, width, yerr=stds, label=objective, capsize=2)
    ax.set_xticks(range(len(conditions)), conditions, rotation=25)
    ax.set_ylabel("Clean-trained probe: standardized state MSE")
    ax.set_title("Mean and sample SD across training seeds (single-run bars have no error estimate)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "comparison.png", dpi=180)
    (args.output_dir / "experiment_configs.json").write_text(json.dumps(
        {label: json.loads(spec) for spec, label in experiments.items()}, indent=2), encoding="utf-8")
    plt.close(fig)
    print(target)


if __name__ == "__main__":
    main()
