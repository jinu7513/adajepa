"""Trajectory-contiguous PushT windows for RGB + frame-derived Event training."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from planning.image_corruption import corrupt_frames
from .common import absolute, file_hash, git_info, write_json
from .data import (CLEAN_RGB, check_state, contact_sheet, new_env, render_state,
                   source_trajectories, validate_dataset)
from .event import event_from_rgb


FRAME_KEYS = ("previous", "current", "next")
SHIFT_KINDS = ("colorA", "colorB", "gaussian", "salt_pepper", "blur")


def event_preview(rows, out, model_cfg, kinds):
    """Write clean/shift RGB and positive/negative event proxy contact sheet."""
    if not rows:
        return
    examples, labels = [], []

    def frame_tensor(image):
        return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)[None].float() / 127.5 - 1

    for images, kind in zip(rows, kinds):
        def make_event(first, second):
            event, diagnostics = event_from_rgb(frame_tensor(first), frame_tensor(second),
                mode=model_cfg["event_mode"], threshold=model_cfg["event_threshold"],
                eps=model_cfg["event_eps"], patch_size=model_cfg["patch_size"],
                shift_previous=model_cfg["shift_previous"])
            channels = event[0].numpy()
            positive = channels[0] > 0
            negative = (channels[1] > 0 if len(channels) == 2 else channels[0] < 0)
            view = np.zeros_like(second)
            view[positive] = [255, 70, 70]
            view[negative] = [70, 90, 255]
            return view, float(diagnostics["event_density"])

        clean_event, clean_density = make_event(images[0], images[1])
        shift_event, shift_density = make_event(images[3], images[4])
        examples.append([images[1], clean_event, images[4], shift_event])
        labels.append(f"{kind}: clean/event {clean_density:.3f} | shift/event {shift_density:.3f}")
    contact_sheet(examples, out / "event_contact_sheet.png", labels)


def generate_event_windows(cfg):
    """Render t-1,t,t+1 from original trajectories, never adjacent pair rows."""
    dc, gc = cfg["dataset"], cfg["generation"]
    base_root = absolute(dc["source_pairs_path"])
    base, base_rows = validate_dataset(base_root)
    base_hash = file_hash(base_root / "manifest.json")
    source_cfg = base["config"]["dataset"]
    trajectories = dict(source_trajectories(source_cfg))
    kinds = dc["shift_kinds"]
    if not kinds or any(kind not in SHIFT_KINDS for kind in kinds):
        raise ValueError("dataset.shift_kinds must use supported conditions")
    for kind in kinds:
        if kind in ("gaussian", "salt_pepper", "blur"):
            low, high = dc["corruption_strength"][kind]
            if (not np.isfinite([low, high]).all() or low < 0 or high < low
                    or (kind == "salt_pepper" and high > 1)):
                raise ValueError(f"Invalid {kind} strength range")
    out = absolute(dc["output_path"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Event dataset output is not empty; use a new output_path")
    out.mkdir(parents=True, exist_ok=True)
    if not gc["preview_only"] and not gc["ranges_reviewed"]:
        raise ValueError("Review the event-window preview before full generation")
    image_size = cfg["model"]["image_size"]
    if image_size != base["config"]["model"]["image_size"]:
        raise ValueError("Event image size must match the four-view source")
    rng = np.random.RandomState(dc["shift_seed"])
    if gc["preview_windows"] < 1:
        raise ValueError("generation.preview_windows must be positive")
    if gc["preview_only"]:
        eligible = [row for row in base_rows
                    if 0 < row["timestep"] < row["trajectory_length"] - 1]
        if not eligible:
            raise ValueError("No non-boundary center states for Event preview")
        n = min(gc["preview_windows"], len(eligible))
        indices = np.random.RandomState(dc["shift_seed"] + 1).choice(len(eligible), n, replace=False)
        selected = [eligible[i] for i in sorted(indices)]
    else:
        selected = base_rows
    kind_schedule = np.resize(np.asarray(kinds), len(selected))
    rng.shuffle(kind_schedule)
    records, rejects, previews, preview_kinds = [], [], [], []
    env = new_env(image_size)
    try:
        for base_row, kind in zip(selected, kind_schedule):
            uid, t = base_row["trajectory_id"], base_row["timestep"]
            states = trajectories[uid]
            if not 0 < t < len(states) - 1:
                rejects.append({"trajectory_id": uid, "timestep": t, "reason": "boundary"})
                continue
            source_triplet = states[t-1:t+2]
            seed = base_row["render_seed"]
            clean, actual = [], []
            for state in source_triplet:
                image, rendered_state = render_state(env, state, CLEAN_RGB, seed)
                clean.append(image)
                actual.append(rendered_state)
            # Re-rendering the center must agree with the already verified Track A
            # dataset; otherwise a renderer/version change could fake motion.
            with Image.open(base_root / base_row["images"]["clean"]) as im:
                expected_center = np.asarray(im.convert("RGB"))
            if not np.array_equal(clean[1], expected_center):
                raise ValueError(f"Center rendering differs from source pairs: {uid} t={t}")
            tolerances = source_cfg["reset_tolerances"]
            checks = [check_state(a, s, tolerances["source"])
                      for a, s in zip(actual, source_triplet)]
            checks.append(check_state(actual[1], base_row["canonical_render_state"],
                                      tolerances["canonical"]))
            if not all(ok for ok, _ in checks):
                rejects.append({"trajectory_id": uid, "timestep": t,
                                "reason": "reset_mismatch", "errors": [error for _, error in checks]})
                continue
            kind = str(kind)
            strength = None
            if kind in ("colorA", "colorB"):
                colors = np.asarray(base_row["eta_" + kind[-1]], dtype=np.uint8)
                shift_results = [render_state(env, state, colors, seed)
                                 for state in source_triplet]
                shifted = [image for image, _ in shift_results]
                with Image.open(base_root / base_row["images"][kind[-1]]) as im:
                    expected_shift_center = np.asarray(im.convert("RGB"))
                if not np.array_equal(shifted[1], expected_shift_center):
                    raise ValueError(f"Shift rendering differs from source pairs: {uid} t={t}")
                shift_checks = [check_state(state, source, tolerances["source"])
                                for (_, state), source in zip(shift_results, source_triplet)]
                if not all(ok for ok, _ in shift_checks):
                    rejects.append({"trajectory_id": uid, "timestep": t,
                                    "reason": "shift_reset_mismatch",
                                    "errors": [error for _, error in shift_checks]})
                    continue
            else:
                low, high = dc["corruption_strength"][kind]
                strength = float(rng.uniform(low, high))
                colors = None
                # The corruption kind and strength persist across the whole
                # window. Stochastic sensor noise is independently drawn per frame.
                shifted = list(corrupt_frames(np.stack(clean), kind, strength,
                                              int(rng.randint(2**31-1))))
            record = {"sample_id": base_row["sample_id"], "trajectory_id": uid,
                      "timestep": t, "trajectory_length": len(states),
                      "split": base_row["split"], "segment": base_row["segment"],
                      "source_states": source_triplet.tolist(),
                      "canonical_render_state": actual[1].tolist(),
                      "render_seed": seed, "shift_kind": kind, "strength": strength,
                      "shift_colors": colors.tolist() if colors is not None else None,
                      "images": {}}
            if len(previews) < gc["preview_windows"]:
                previews.append(clean + shifted)
                preview_kinds.append(kind)
            if not gc["preview_only"]:
                (out / "images").mkdir(exist_ok=True)
                for view, frames in (("clean", clean), ("shift", shifted)):
                    for name, image in zip(FRAME_KEYS, frames):
                        relative = f"images/{base_row['sample_id']:08d}_{view}_{name}.png"
                        Image.fromarray(image).save(out / relative)
                        record["images"][f"{view}_{name}"] = relative
            records.append(record)
    finally:
        env.close()
    contact_sheet(previews, out / "window_contact_sheet.png",
                  [f"{kind}: clean prev/current/next | shift prev/current/next"
                   for kind in preview_kinds])
    event_preview(previews, out, cfg["model"], preview_kinds)
    write_json(out / "generation_report.json", {"accepted": len(records),
               "rejected": len(rejects), "rejections": rejects,
               "accepted_by_split": {split: sum(r["split"] == split for r in records)
                                     for split in ("train", "validation", "test")},
               "accepted_by_shift": {kind: sum(r["shift_kind"] == kind for r in records)
                                     for kind in kinds},
               "source_pairs_manifest_sha256": base_hash})
    if gc["preview_only"]:
        return {"preview_path": str(out), "accepted": len(records)}
    if any(not any(r["split"] == split for r in records)
           for split in ("train", "validation", "test")):
        raise ValueError("Event dataset contains an empty trajectory split")
    with (out / "samples.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    manifest = {"schema_version": "tracka-event-window-v1", "config": cfg,
                "source_pairs_path": str(base_root), "source_pairs_manifest_sha256": base_hash,
                "source_path": source_cfg["source_path"], "splits": base["splits"],
                "source_reset_tolerances": source_cfg["reset_tolerances"]["source"],
                "canonical_reset_tolerances": source_cfg["reset_tolerances"]["canonical"],
                "sample_count": len(records), "image_count": 6 * len(records),
                "metadata_sha256": file_hash(out / "samples.jsonl"), "git": git_info(),
                "rejected_count": len(rejects)}
    write_json(out / "manifest.json", manifest)
    validate_event_windows(out)
    return manifest


def validate_event_windows(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(s) for s in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(rows) != manifest["sample_count"] or manifest["image_count"] != 6 * len(rows)
            or file_hash(root / "samples.jsonl") != manifest["metadata_sha256"]):
        raise ValueError("Event manifest integrity mismatch")
    split_sets = [set(manifest["splits"][split]) for split in ("train", "validation", "test")]
    if any(split_sets[i] & split_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Event dataset has trajectory split leakage")
    expected = {f"{view}_{frame}" for view in ("clean", "shift") for frame in FRAME_KEYS}
    seen = set()
    for row in rows:
        uid, t = row["trajectory_id"], row["timestep"]
        if (uid, t) in seen or uid not in manifest["splits"][row["split"]]:
            raise ValueError("Duplicate window or trajectory split leakage")
        seen.add((uid, t))
        states = np.asarray(row["source_states"])
        if (not 0 < t < row["trajectory_length"] - 1 or set(row["images"]) != expected
                or states.shape != (3, 7) or not np.isfinite(states).all()):
            raise ValueError("Invalid event window")
        for relative in row["images"].values():
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError("Missing/unsafe event image")
    return manifest, rows


class EventWindowDataset(Dataset):
    def __init__(self, root, split, image_size, load_shift_next=False):
        self.root = absolute(root)
        self.manifest, all_rows = validate_event_windows(self.root)
        self.rows = [r for r in all_rows if r["split"] == split]
        self.image_size = image_size
        self.load_shift_next = load_shift_next
        if not self.rows:
            raise ValueError(f"Empty event split: {split}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = {}
        for view in ("clean", "shift"):
            for frame in FRAME_KEYS:
                if view == "shift" and frame == "next" and not self.load_shift_next:
                    continue  # Stored for future ablations, unused by the baseline objective.
                key = f"{view}_{frame}"
                with Image.open(self.root / row["images"][key]) as im:
                    image = np.array(im.convert("RGB"), copy=True)
                if image.shape != (self.image_size, self.image_size, 3):
                    raise ValueError("Event image resolution differs from model config")
                item[key] = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1
        return item
