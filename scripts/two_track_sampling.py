"""Audits the actual map-style PushT window loader and appearance draws."""

import json
import hashlib
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def audit_windows(slicer, batch_size, epochs=3, output=None):
    slices = [(int(e), int(s)) for e, s, _ in slicer.slices]
    if len(slices) != len(set(slices)):
        raise ValueError("Duplicate physical window in one map-style epoch")
    per_episode = Counter(e for e, _ in slices)
    if not per_episode:
        raise ValueError("No valid training windows in the actual PushT loader")
    physical = slicer.dataset
    fingerprints = {
        hashlib.sha256(physical.states[i, :int(length)].contiguous().numpy().tobytes()).hexdigest()
        for i, length in enumerate(physical.seq_lengths)
    }
    result = {
        "episode_count": len(physical.seq_lengths),
        "source_episode_lengths": [int(v) for v in physical.seq_lengths],
        "unique_physical_trajectories": len(fingerprints),
        "window_count": len(slices), "unique_windows": len(slices),
        "window_stride_raw_frames": 1, "frameskip": int(slicer.frameskip),
        "frames_per_window": int(slicer.num_frames),
        "window_visits_per_complete_epoch": 1,
        "window_visits_over_complete_epochs": int(epochs),
        "batch_size": int(batch_size), "drop_last": False,
        "shuffle": "one NumPy permutation at dataset construction, no replacement; DataLoader shuffle=False",
        "optimizer_steps_per_complete_epoch": math.ceil(len(slices) / batch_size),
        "windows_per_episode": {str(k): v for k, v in sorted(per_episode.items())},
        "window_count_min_median_max": [min(per_episode.values()), statistics.median(per_episode.values()),
                                         max(per_episode.values())],
        "window_count_histogram": dict(sorted(Counter(per_episode.values()).items())),
        "resume_note": "Epoch-level resumption repeats the interrupted epoch from its first batch; see attempt logs.",
    }
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "data_sampling_audit.json").write_text(json.dumps(result, indent=2) + "\n")
        markdown = (
            "# Actual PushT loader sampling audit\n\n"
            "Generated from `TrajSlicerDataset.slices` in the real training process.\n\n"
            "- Source episodes: {}\n- Unique windows: {}\n- Window starts: stride 1 raw frame; frameskip {}.\n"
            "- Windows per episode: min {}, median {}, max {}.\n"
            "- Batch size: {}; drop_last=false; {} optimizer steps per complete epoch.\n"
            "- Each window is visited once per complete epoch without replacement. At {} complete epochs, "
            "each is visited {} times. Overlapping windows repeatedly use each physical trajectory.\n"
            "- Mid-epoch preemption repeats part of that epoch on restart; inspect attempt logs.\n"
        ).format(result["episode_count"], result["window_count"], slicer.frameskip,
                    *result["window_count_min_median_max"], batch_size,
                    result["optimizer_steps_per_complete_epoch"], epochs, epochs)
        (output / "data_sampling_audit.md").write_text(markdown)
    return result


def appearance_counts(slicer, epoch):
    dataset = slicer.dataset
    dataset.set_epoch(epoch)
    total = Counter()
    by_episode = defaultdict(Counter)
    for episode, start, _ in slicer.slices:
        pair = dataset.pair_for(episode, start)
        key = "{}->{}".format(*pair)
        total[key] += 1
        by_episode[int(episode)][key] += 1
    return {"epoch": int(epoch), "mode": dataset.color_pair_mode,
            "pair_counts": dict(total), "per_episode_pair_counts":
            {str(k): dict(v) for k, v in by_episode.items()}}
