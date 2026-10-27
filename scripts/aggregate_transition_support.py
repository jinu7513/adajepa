"""Combine all support ablations with A/B/C MPC results after the Slurm DAG."""

import argparse
import csv
import json
from pathlib import Path


def read_curve(path):
    points = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if "paper/mpc_step" in item and "paper/success_rate_pct" in item:
            points[int(item["paper/mpc_step"])] = float(item["paper/success_rate_pct"])
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    variants = ["support_identity_vanilla"] + [f"support_{feature}_{base}_{affine}"
                for feature in ("delta", "residual")
                for base in ("frozen", "trainable")
                for affine in ("mlp", "linear")]
    offline_rows = []
    for variant in variants:
        path = root / "offline" / variant / "results.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing offline ablation results: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                offline_rows.append({"model": variant, **item})
    mpc_rows = []
    for variant in ("A", "B", "C", *variants):
        k_values = (("baseline",) if variant in ("A", "B", "C") else
                    (0,) if variant == "support_identity_vanilla" else (0, 2, 4, 8))
        for k in k_values:
            for condition in ("default", "redBlock"):
                case = (root / "eval" / variant / condition if k == "baseline"
                        else root / "eval" / variant / f"k{k}" / condition)
                pointer = case / "done.txt"
                if not pointer.is_file():
                    raise FileNotFoundError(f"MPC case has no completion marker: {case}")
                attempt = Path(pointer.read_text(encoding="utf-8").strip())
                curve = read_curve(attempt / "logs.json")
                if not curve:
                    raise ValueError(f"MPC case has no W&B paper curve: {attempt}")
                for step, success in sorted(curve.items()):
                    mpc_rows.append({"model": variant, "support_k": k, "condition": condition,
                                     "seed": 100, "n_evals": 50, "mpc_step": step,
                                     "success_rate_pct": success, "run_dir": str(attempt)})
    output = root / "summary"
    output.mkdir(exist_ok=True)
    for name, rows in (("offline_all_ablation.csv", offline_rows), ("mpc_success_curves.csv", mpc_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    highlighted = ("A", "B", "C", "support_identity_vanilla", "support_delta_frozen_mlp")
    for ax, condition in zip(axes, ("default", "redBlock")):
        for variant in highlighted:
            k = "baseline" if variant in ("A", "B", "C") else (0 if variant == "support_identity_vanilla" else 8)
            selected = [r for r in mpc_rows if r["model"] == variant and r["support_k"] == k and r["condition"] == condition]
            ax.plot([r["mpc_step"] for r in selected], [r["success_rate_pct"] for r in selected], label=variant)
        ax.set(title=condition, xlabel="MPC step", xlim=(1, 30), ylim=(0, 100))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Success rate (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "success_rate_vs_mpc_step.png", dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, domain in zip(axes, ("default", "redBlock")):
        for variant in variants:
            selected = [r for r in offline_rows if r["model"] == variant and r["domain"] == domain
                        and r["support_mode"] == ("identity" if variant == "support_identity_vanilla" else "correct")]
            selected.sort(key=lambda r: int(r["support_k"]))
            ax.plot([int(r["support_k"]) for r in selected],
                    [float(r["one_step_mse"]) for r in selected],
                    marker="o", label=variant.removeprefix("support_"))
        ax.set(title=domain, xlabel="Observed support transitions K", xticks=(0, 2, 4, 8))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("One-step latent MSE")
    axes[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(output / "offline_one_step_vs_support_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {len(offline_rows)} offline rows, {len(mpc_rows)} MPC points to {output}")


if __name__ == "__main__":
    main()
