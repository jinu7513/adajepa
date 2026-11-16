"""Validate paired PushT color assets before a P/J gradient-flow comparison."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch


TRAIN_COLORS = ("default", "blue", "green")
VALIDATION_COLOR = "purpleBlock"
TEST_COLOR = "redBlock"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(root, split, colors, num_hist, frameskip):
    folder = root / split
    with (folder / "seq_lengths.pkl").open("rb") as stream:
        lengths = [int(x) for x in pickle.load(stream)]
    states = torch.load(folder / "states.pth", map_location="cpu")
    actions = torch.load(folder / "rel_actions.pth", map_location="cpu")
    if len(states) != len(actions) or len(states) != len(lengths):
        raise ValueError("Episode counts disagree in {}".format(folder))
    fingerprints = set()
    for i, length in enumerate(lengths):
        if length < 1 or length > len(states[i]) or length > len(actions[i]):
            raise ValueError("Invalid sequence length for {}/{}".format(split, i))
        digest = hashlib.sha256()
        digest.update(states[i, :length].contiguous().numpy().tobytes())
        digest.update(actions[i, :length].contiguous().numpy().tobytes())
        fingerprints.add(digest.hexdigest())
        for color in colors:
            video = folder / "colors" / color / "obses" / "episode_{:03d}.mp4".format(i)
            if not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError(str(video))
    counts = {color: 0 for color in colors}
    for i, length in enumerate(lengths):
        windows = max(0, length - (num_hist + 1) * frameskip + 1)
        if split == "train":
            for start in range(windows):
                counts[colors[(i * 1009 + start) % len(colors)]] += 1
        else:
            for color in colors:
                counts[color] += windows
    return {"episodes": len(lengths), "windows_per_color": counts,
            "trajectory_fingerprints": fingerprints}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-hist", type=int, default=3)
    parser.add_argument("--frameskip", type=int, default=5)
    args = parser.parse_args()
    root = args.data_root.resolve()
    train = inspect_split(root, "train", TRAIN_COLORS, args.num_hist, args.frameskip)
    val = inspect_split(root, "val", (VALIDATION_COLOR, TEST_COLOR), args.num_hist, args.frameskip)
    overlap = train["trajectory_fingerprints"] & val["trajectory_fingerprints"]
    if overlap:
        raise ValueError("{} physical trajectories occur in both train and val".format(len(overlap)))
    train.pop("trajectory_fingerprints")
    val.pop("trajectory_fingerprints")
    assets = {}
    expected_colors = {
        "transition_color_manifest.json": {"default": "#778899", "blue": "#003D99", "green": "#006B3C"},
        "transition_heldout_manifest.json": {"redBlock": "#FF0000", "purpleBlock": "#A020F0"},
    }
    for name, colors in expected_colors.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(str(path))
        content = json.loads(path.read_text())
        if content.get("colors") != colors or content.get("preview_only"):
            raise ValueError("Color manifest does not match the controlled split: {}".format(path))
        for source, expected_hash in content.get("source_hashes", {}).items():
            if sha256(root / source) != expected_hash:
                raise ValueError("Source physical trajectory changed after rendering: {}".format(source))
        assets[name] = {"sha256": sha256(path), "content": content}
    payload = {
        "data_root": str(root), "train_colors": TRAIN_COLORS,
        "validation_color": VALIDATION_COLOR, "test_color": TEST_COLOR,
        "physical_split_overlap": 0,
        "test_trajectory_holdout": False,
        "note": "Red test appearance uses val trajectories; it is not a separate physical test split.",
        "train": train, "val": val, "manifests": assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("Saved {}".format(args.output))


if __name__ == "__main__":
    main()
