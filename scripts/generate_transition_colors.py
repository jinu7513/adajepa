"""Render paired PushT color videos from existing physical state trajectories.

The script never changes source videos, actions, states, or train/val splits.
It is restartable: each completed MP4 is atomically installed and skipped on
subsequent invocations. Run it as a CPU Slurm stage before B/C training.
"""

import argparse
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader

from env.pusht.pusht_env import PushTEnv


COLORS = {"default": "LightSlateGray", "blue": "#003D99", "green": "#006B3C"}
HELD_OUT_COLORS = {"redBlock": "Red", "purpleBlock": "Purple"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_frames(state, shape, size, colors=COLORS):
    environments = {
        name: PushTEnv(color=color, shape=shape, with_velocity=True, render_size=size)
        for name, color in colors.items()
    }
    try:
        for env in environments.values():
            env.reset_to_state = state[0]
            env.reset()
        for frame_idx, current_state in enumerate(state):
            frames = {}
            for name, env in environments.items():
                if frame_idx:
                    env._set_state(current_state)
                frames[name] = env.render("rgb_array")
            yield frames
    finally:
        for env in environments.values():
            env.close()


def run_split(root, split, preview_only, max_default_mae, colors=COLORS):
    data_dir = root / split
    states = torch.load(data_dir / "states.pth", map_location="cpu").numpy()
    velocities = torch.load(data_dir / "velocities.pth", map_location="cpu").numpy()
    with open(data_dir / "seq_lengths.pkl", "rb") as handle:
        lengths = pickle.load(handle)
    shapes_path = data_dir / "shapes.pkl"
    if shapes_path.exists():
        with open(shapes_path, "rb") as handle:
            shapes = pickle.load(handle)
    else:
        shapes = ["T"] * len(lengths)
    if len(states) != len(lengths) or len(velocities) != len(lengths):
        raise ValueError(f"State/velocity/length episode counts disagree in {data_dir}")
    video_root = data_dir / "obses"
    missing_bytes = sum(
        (video_root / f"episode_{i:03d}.mp4").stat().st_size
        for i in range(len(lengths))
        for name in colors
        if not (data_dir / "colors" / name / "obses" / f"episode_{i:03d}.mp4").is_file()
        or (data_dir / "colors" / name / "obses" / f"episode_{i:03d}.mp4").stat().st_size == 0
    )
    if not preview_only and shutil.disk_usage(root).free < missing_bytes * 1.2:
        raise OSError(f"Insufficient free disk space to render {split} color videos")

    report = []
    indices = range(min(2, len(lengths))) if preview_only else range(len(lengths))
    for episode in indices:
        count = int(lengths[episode])
        source_path = video_root / f"episode_{episode:03d}.mp4"
        source = VideoReader(str(source_path), num_threads=1)
        if len(source) < count:
            raise ValueError(f"Source video shorter than state trajectory: {source_path}")
        source_shape = source[0].shape
        if source_shape[0] != source_shape[1] or source_shape[2] != 3:
            raise ValueError(f"Expected square RGB source frames: {source_path} {source_shape}")
        render_size = int(source_shape[0])
        selected_state = np.concatenate([states[episode, :count, :5], velocities[episode, :count, :2]], axis=-1)
        output_paths = {
            name: data_dir / "colors" / name / "obses" / f"episode_{episode:03d}.mp4"
            for name in colors
        }
        if not preview_only and all(path.is_file() and path.stat().st_size > 0 for path in output_paths.values()):
            continue
        for path in output_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        writers = {}
        temporary = {}
        mae = []
        changed = []
        try:
            if not preview_only:
                for name, path in output_paths.items():
                    tmp = path.with_name(path.stem + ".partial.mp4")
                    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (render_size, render_size))
                    if not writer.isOpened():
                        raise OSError(f"Could not create {tmp}")
                    writers[name] = writer
                    temporary[name] = tmp
            for frame_idx, frames in enumerate(paired_frames(selected_state, shapes[episode], render_size, colors)):
                if frame_idx < 8:
                    original = source[frame_idx].asnumpy().astype(np.float32)
                    if "default" in frames:
                        mae.append(float(np.abs(frames["default"].astype(np.float32) - original).mean() / 255.0))
                        changed.append(float(np.mean(np.any(frames["default"] != frames["blue"], axis=-1))))
                    if preview_only and frame_idx == 0:
                        sheet = np.concatenate([frames[name] for name in colors], axis=1)
                        preview = data_dir / f"transition_colors_episode_{episode:03d}.png"
                        cv2.imwrite(str(preview), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
                for name, writer in writers.items():
                    writer.write(cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR))
            default_mae = float(np.mean(mae)) if mae else None
            if default_mae is not None and default_mae > max_default_mae:
                raise ValueError(
                    f"Default re-render differs from source MP4 for {split}/{episode}: "
                    f"MAE={default_mae:.4f} > {max_default_mae:.4f}; inspect preview before continuing"
                )
            if "default" in colors and (not changed or max(changed) < 0.001):
                raise ValueError("Block recoloring did not visibly change the rendered frames")
        finally:
            for writer in writers.values():
                writer.release()
        for name, tmp in temporary.items():
            os.replace(tmp, output_paths[name])
        report.append({"episode": episode, "frames": count, "default_source_mae": default_mae})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--heldout-only", action="store_true",
                        help="Render only held-out red/purple videos for val; never train")
    parser.add_argument("--max-default-mae", type=float, default=0.06)
    args = parser.parse_args()
    root = args.data_root.resolve()
    colors = HELD_OUT_COLORS if args.heldout_only else COLORS
    splits = ("val",) if args.heldout_only else ("train", "val")
    report = {split: run_split(root, split, args.preview_only, args.max_default_mae, colors) for split in splits}
    payload = {"colors": colors, "preview_only": args.preview_only, "report": report}
    if not args.preview_only:
        payload["source_hashes"] = {
            f"{split}/{name}": sha256(root / split / name)
            for split in splits
            for name in ("states.pth", "rel_actions.pth", "seq_lengths.pkl")
        }
    prefix = "transition_heldout" if args.heldout_only else "transition_color"
    out = root / (f"{prefix}_preview_report.json" if args.preview_only else f"{prefix}_manifest.json")
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
