"""Run the actual PushT loader and persist window/color sampling audits."""

import argparse
import json
from pathlib import Path

from datasets.pusht_dset import load_pusht_slice_train_val
from scripts.two_track_sampling import appearance_counts, audit_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track", choices=("cross_color_jepa", "mi_jepa"), required=True)
    parser.add_argument("--mode", default="cross_window_random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    kwargs = ({"color_domains": ("default", "blue", "green"),
               "validation_color_domain": "purpleBlock", "color_pair_mode": args.mode,
               "color_seed": args.seed} if args.track == "cross_color_jepa" else {})
    datasets, _ = load_pusht_slice_train_val(
        transform=None, n_rollout=None, data_path=str(args.data_root),
        normalize_action=True, num_hist=3, num_pred=1, frameskip=5,
        with_velocity=True, **kwargs
    )
    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_windows(datasets["train"], args.batch_size, args.epochs, args.output)
    if args.track == "cross_color_jepa":
        draws = [appearance_counts(datasets["train"], epoch)
                 for epoch in range(1, args.epochs + 1)]
        (args.output / "appearance_pair_counts.json").write_text(json.dumps(draws, indent=2) + "\n")
    print("Audited {} source episodes and {} windows".format(
        audit["episode_count"], audit["window_count"]))


if __name__ == "__main__":
    main()
