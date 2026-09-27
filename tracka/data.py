"""Offline four-view PushT dataset. Physical states are metadata/probe labels only."""
import colorsys
import json
import pickle
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from planning.image_corruption import corrupt_frames
from .common import absolute, file_hash, git_info, write_json

CLEAN_RGB = np.array([[119, 136, 153], [65, 105, 225], [144, 238, 144]], dtype=np.uint8)
VIEWS = ("clean", "A", "B", "corrupt")


def state_errors(actual, reference):
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {"agent_position": float(np.abs(delta[:2]).max()),
            "block_position": float(np.abs(delta[2:4]).max()),
            "angle": float(abs(np.arctan2(np.sin(delta[4]), np.cos(delta[4])))),
            "agent_velocity": float(np.abs(delta[5:7]).max())}


def check_state(actual, reference, tolerances):
    errors = state_errors(actual, reference)
    return all(errors[k] <= tolerances[k] for k in errors), errors


def valid_colors(rgb, previous, cfg):
    # Euclidean distance in rendered RGB / 255, in [0,sqrt(3)].
    x = np.asarray(rgb, dtype=float) / 255
    if cfg["distance_metric"] != "rgb_l2":
        raise ValueError("Initial distance_metric supports rgb_l2 only")
    if np.min(np.linalg.norm(x - 1, axis=-1)) < cfg["min_background_distance"]:
        return False
    if min(np.linalg.norm(x[i] - x[j]) for i, j in ((0, 1), (0, 2), (1, 2))) < cfg["min_inter_object_distance"]:
        return False
    for other in previous:
        if np.min(np.linalg.norm(x - np.asarray(other) / 255., axis=-1)) < cfg["min_inter_view_distance"]:
            return False
    return True


def sample_colors(rng, previous, cfg):
    if cfg["color_space"] != "hsv":
        raise ValueError("Initial color sampling supports hsv")
    for key in ("hue_range", "saturation_range", "value_range"):
        if not 0 <= cfg[key][0] <= cfg[key][1] <= 1:
            raise ValueError("HSV bounds must lie in [0,1]: " + key)
    for _ in range(cfg["max_attempts"]):
        hsv = np.stack([rng.uniform(*cfg[key], size=3)
                        for key in ("hue_range", "saturation_range", "value_range")], -1)
        continuous = np.array([colorsys.hsv_to_rgb(*c) for c in hsv])
        actual = np.rint(continuous * 255).astype(np.uint8)
        if valid_colors(actual, previous, cfg):
            return actual, {"hsv": hsv.tolist(), "rgb_float": continuous.tolist(), "rgb_uint8": actual.tolist()}
    raise ValueError("Color rejection sampling exhausted; relax configured bounds")


def new_env(image_size):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from env.pusht.pusht_env import PushTEnv
    return PushTEnv(with_velocity=True, with_target=True, render_size=image_size, render_action=False)


def render_state(env, source, rgb, seed):
    env.color, env.agent_color, env.goal_color_override = [tuple(map(int, c)) for c in rgb]
    env.seed(int(seed))
    env.reset_to_state = np.asarray(source, dtype=np.float64)
    obs, state = env.reset()
    return obs["visual"], state


def segment(timestep, length):
    u = timestep / max(length - 1, 1)
    return "early" if u < 1 / 3 else "middle" if u < 2 / 3 else "late"


def source_trajectories(cfg):
    root = absolute(cfg["source_path"])
    result = []
    for split in cfg["source_splits"]:
        folder = root / split
        states = torch.load(folder / "states.pth", map_location="cpu", weights_only=False).numpy()
        velocities = None
        if states.shape[-1] == 5:
            velocities = torch.load(folder / "velocities.pth", map_location="cpu", weights_only=False).numpy()
        with open(folder / "seq_lengths.pkl", "rb") as f:
            lengths = pickle.load(f)
        shapes = ["T"] * len(lengths)
        if (folder / "shapes.pkl").exists():
            with open(folder / "shapes.pkl", "rb") as f:
                shapes = pickle.load(f)
        for i, length in enumerate(lengths):
            uid = f"{split}/{i}"
            if cfg["trajectory_ids"] is not None and uid not in cfg["trajectory_ids"]:
                continue
            if shapes[i] != "T":
                continue
            st = states[i, :int(length)]
            if velocities is not None:
                st = np.concatenate([st, velocities[i, :int(length)]], -1)
            if st.shape[-1] != 7 or not np.isfinite(st).all():
                raise ValueError(f"Expected finite 7D PushT states: {uid}")
            result.append((uid, st))
    if len(result) < 3:
        raise ValueError("At least three source trajectories are required")
    return result


def split_ids(ids, ratios, seed):
    ratios = np.asarray(ratios, dtype=float)
    if len(ratios) != 3 or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError("Use three positive split ratios summing to one")
    ordered = np.random.RandomState(seed).permutation(sorted(ids)).tolist()
    n = len(ordered)
    nval, ntest = max(1, int(n * ratios[1])), max(1, int(n * ratios[2]))
    ntrain = n - nval - ntest
    if ntrain < 1:
        raise ValueError("Split leaves no training trajectories")
    return dict(zip(("train", "validation", "test"),
                    (ordered[:ntrain], ordered[ntrain:ntrain+nval], ordered[ntrain+nval:])))


def contact_sheet(rows, path, labels=None):
    if not rows:
        return
    h, w = rows[0][0].shape[:2]
    canvas = Image.new("RGB", (len(rows[0]) * w, len(rows) * (h + 24)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            canvas.paste(Image.fromarray(im), (c * w, r * (h + 24)))
        if labels:
            draw.text((2, r * (h + 24) + h), labels[r], fill="black")
    canvas.save(path)


def preview_corruptions(clean, cfg, out, patch_size=16):
    rng = np.random.RandomState(cfg["corruption_seed"])
    side = clean.shape[0] // patch_size
    mask = np.zeros(side * side, dtype=bool)
    mask[rng.permutation(len(mask))[:len(mask)//2]] = True
    mask = np.repeat(np.repeat(mask.reshape(side, side), patch_size, 0), patch_size, 1)
    rows, labels = [], []
    for kind in cfg["corruption_types"]:
        low, high = cfg["corruption"][kind + "_strength_range"]
        for strength in np.linspace(low, high, 3):
            im = corrupt_frames(clean, kind, float(strength), cfg["corruption_seed"])
            hidden = im.copy()
            hidden[mask] = 127
            rows.append([clean, im, hidden])
            labels.append(f"{kind} strength={strength:.4g}: clean / corrupt / 50% masked")
    contact_sheet(rows, out / "corruption_contact_sheet.png", labels)


def generate(cfg):
    dc = cfg["dataset"]
    if not dc["source_path"]:
        raise ValueError("Set dataset.source_path to the directory containing source_splits")
    if dc["frame_stride"] < 1 or any(dc[k] is not None and dc[k] < 1
                                    for k in ("max_states_per_trajectory", "max_total_states")):
        raise ValueError("Frame stride and state caps must be positive")
    out = absolute(dc["output_path"])
    out.mkdir(parents=True, exist_ok=True)
    if ((out / "manifest.json").exists() or (out / "samples.jsonl").exists()
            or ((out / "images").exists() and any((out / "images").iterdir()))):
        raise FileExistsError("Dataset output already exists; use a new output_path")
    if not cfg["generation"]["preview_only"] and not cfg["generation"]["ranges_reviewed"]:
        raise ValueError("Run preview_only first; review contact sheets, then set ranges_reviewed=true")
    trajectories = source_trajectories(dc)
    splits = split_ids([uid for uid, _ in trajectories], dc["trajectory_split"], dc["split_seed"])
    assignments = {uid: sp for sp, uids in splits.items() for uid in uids}
    sample_rng = np.random.RandomState(dc["state_sampling_seed"])
    nuisance_rng = np.random.RandomState(dc["nuisance_assignment_seed"])
    corrupt_rng = np.random.RandomState(dc["corruption_seed"])
    render_rng = np.random.RandomState(dc["dataset_seed"])
    candidates = []
    for uid, states in trajectories:
        times = np.arange(0, len(states), dc["frame_stride"])
        cap = dc["max_states_per_trajectory"]
        if cap is not None and len(times) > cap:
            times = np.sort(sample_rng.choice(times, cap, replace=False))
        candidates.extend((uid, int(t), states[t], len(states)) for t in times)
    cap = dc["max_total_states"]
    if cap is not None and len(candidates) > cap:
        candidates = [candidates[i] for i in sorted(sample_rng.choice(len(candidates), cap, replace=False))]
    if cfg["generation"]["preview_only"]:
        candidates = candidates[:cfg["generation"]["preview_states"]]
    if not candidates:
        raise ValueError("No states selected")
    types = dc["corruption_types"]
    if not types or any(k not in ("gaussian", "salt_pepper", "blur") for k in types):
        raise ValueError("Use canonical Track A corruption types")
    for kind in types:
        low, high = dc["corruption"][kind + "_strength_range"]
        if not 0 <= low <= high or not np.isfinite(high) or (kind == "salt_pepper" and high > 1):
            raise ValueError("Invalid corruption range: " + kind)
    kind_schedule = np.resize(np.array(types), len(candidates))
    corrupt_rng.shuffle(kind_schedule)
    records, rejects, sheets = [], [], []
    max_errors = {key: {f: 0. for f in ("agent_position", "block_position", "angle", "agent_velocity")}
                  for key in ("source", "canonical")}
    env = new_env(cfg["model"]["image_size"])
    try:
        for idx, (uid, t, source, length) in enumerate(candidates):
            rgb_a, meta_a = sample_colors(nuisance_rng, [CLEAN_RGB], dc["color_sampling"])
            rgb_b, meta_b = sample_colors(nuisance_rng, [CLEAN_RGB, rgb_a], dc["color_sampling"])
            render_seed = int(render_rng.randint(2**31-1))
            clean, canonical = render_state(env, source, CLEAN_RGB, render_seed)
            im_a, state_a = render_state(env, source, rgb_a, render_seed)
            im_b, state_b = render_state(env, source, rgb_b, render_seed)
            valid, errors = True, []
            for actual in (canonical, state_a, state_b):
                row_errors = {}
                for key, reference in (("source", source), ("canonical", canonical)):
                    ok, err = check_state(actual, reference, dc["reset_tolerances"][key])
                    valid = valid and ok
                    row_errors[key] = err
                    for field, value in err.items():
                        max_errors[key][field] = max(max_errors[key][field], value)
                errors.append(row_errors)
            if not valid:
                rejects.append({"trajectory_id": uid, "timestep": t, "segment": segment(t, length), "errors": errors})
                continue
            kind = str(kind_schedule[idx])
            strength = float(corrupt_rng.uniform(*dc["corruption"][kind + "_strength_range"]))
            cseed = int(corrupt_rng.randint(2**31-1))
            corr = corrupt_frames(clean, kind, strength, cseed)
            if len(sheets) < cfg["generation"]["preview_states"]:
                sheets.append([clean, im_a, im_b, corr])
            if not records:
                preview_corruptions(clean, dc, out, cfg["model"]["patch_size"])
            record = {"sample_id": idx, "trajectory_id": uid, "timestep": t, "trajectory_length": length,
                      "source_state": source.tolist(), "canonical_render_state": canonical.tolist(),
                      "eta_clean": CLEAN_RGB.tolist(), "eta_A": rgb_a.tolist(), "eta_B": rgb_b.tolist(),
                      "continuous_colors": {"A": meta_a, "B": meta_b},
                      "corruption_type": kind, "corruption_strength": strength, "corruption_seed": cseed,
                      "dataset_seed": dc["dataset_seed"], "nuisance_assignment_seed": dc["nuisance_assignment_seed"],
                      "render_seed": render_seed, "split": assignments[uid], "segment": segment(t, length),
                      "verified": True, "reset_errors": errors, "images": {}}
            if not cfg["generation"]["preview_only"]:
                for view, image in zip(VIEWS, (clean, im_a, im_b, corr)):
                    relative = f"images/{idx:08d}_{view}.png"
                    (out / "images").mkdir(exist_ok=True)
                    Image.fromarray(image).save(out / relative)
                    record["images"][view] = relative
            records.append(record)
    finally:
        env.close()
    contact_sheet(sheets, out / "four_view_contact_sheet.png",
                  ["clean / color A / color B / corruption"] * len(sheets))
    rejection_counts = {sp: {seg: 0 for seg in ("early", "middle", "late")}
                        for sp in splits}
    for reject in rejects:
        rejection_counts[assignments[reject["trajectory_id"]]][reject["segment"]] += 1
    write_json(out / "generation_report.json", {"accepted": len(records), "rejected": len(rejects),
                                                "max_reset_errors": max_errors, "rejections": rejects,
                                                "rejections_by_split_segment": rejection_counts,
                                                "config": cfg})
    if not records:
        raise ValueError("Every selected state failed reset verification; inspect generation_report.json")
    if cfg["generation"]["preview_only"]:
        return {"preview_path": str(out), "accepted": len(records)}
    if any(not any(row["split"] == s for row in records) for s in splits):
        raise ValueError("An accepted split is empty; inspect generation_report and regenerate in a fresh directory")
    with open(out / "samples.jsonl", "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    manifest = {"schema_version": dc["schema_version"], "config": cfg, "git": git_info(),
                "source_path": str(absolute(dc["source_path"])), "splits": splits,
                "sample_count": len(records), "image_count": 4 * len(records),
                "metadata_sha256": file_hash(out / "samples.jsonl"),
                "selected_states": [{"trajectory_id": uid, "timestep": t, "trajectory_length": length}
                                    for uid, t, _, length in candidates],
                "max_reset_errors": max_errors, "rejected_count": len(rejects)}
    write_json(out / "manifest.json", manifest)
    validate_dataset(out)
    return manifest


def validate_dataset(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    if (len(rows) != manifest["sample_count"] or manifest["image_count"] != 4 * len(rows)
            or file_hash(root / "samples.jsonl") != manifest["metadata_sha256"]):
        raise ValueError("Manifest/metadata integrity mismatch")
    sets = [set(manifest["splits"][s]) for s in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Trajectory split leakage")
    paths, samples = set(), set()
    for row in rows:
        uid, t = row["trajectory_id"], row["timestep"]
        if uid not in manifest["splits"][row["split"]] or not 0 <= t < row["trajectory_length"]:
            raise ValueError("Invalid trajectory/timestep assignment")
        if (uid, t) in samples:
            raise ValueError("Duplicate state sample")
        samples.add((uid, t))
        if not row["verified"] or set(row["images"]) != set(VIEWS):
            raise ValueError("Unverified or incomplete sample")
        for required in ("corruption_type", "corruption_strength", "corruption_seed", "eta_clean", "eta_A", "eta_B"):
            if required not in row:
                raise ValueError("Missing metadata: " + required)
        for relative in row["images"].values():
            p = (root / relative).resolve()
            if root.resolve() not in p.parents or not p.is_file():
                raise ValueError("Invalid image path")
            paths.add(p)
    if len(paths) != 4 * len(rows) or len(list((root / "images").glob("*.png"))) != len(paths):
        raise ValueError("Image count mismatch")
    return manifest, rows


class FourViewDataset(Dataset):
    def __init__(self, root, split, image_size):
        self.root = absolute(root)
        self.manifest, all_rows = validate_dataset(self.root)
        self.rows = [r for r in all_rows if r["split"] == split]
        self.image_size = image_size
        if not self.rows:
            raise ValueError(f"No accepted samples in {split}; adjust state caps or reset tolerances")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = {}
        for view in VIEWS:
            with Image.open(self.root / row["images"][view]) as im:
                array = np.array(im.convert("RGB"), copy=True)
            if array.shape != (self.image_size, self.image_size, 3):
                raise ValueError("Dataset resolution must match encoder config")
            item[view] = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1
        for eta in ("eta_clean", "eta_A", "eta_B"):
            item[eta] = torch.tensor(row[eta], dtype=torch.float32) / 255
        return item
