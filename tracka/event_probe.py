"""Frozen state and temporal probes for consecutive-frame RGB/Event encoders."""

import copy
import csv

import numpy as np
import torch
from torch.utils.data import default_collate

from planning.image_corruption import corrupt_frames
from .common import absolute, file_hash, git_info, load_checkpoint, seed_all, write_json
from .data import CLEAN_RGB, check_state, new_env, render_state
from .event_data import EventWindowDataset
from .event_model import EventTeacherStudent, representation_distance
from .event_train import spectrum_metrics
from .logging import RunLogger
from .probes import fit_probe, physical_targets, plot_results, state_metrics
from .train import device_for, to_device, unique_output


def as_tensor(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 127.5 - 1


def condition_pair(dataset, index, condition, seed, env=None, strength=None):
    row = dataset.rows[index]
    item = dataset[index]
    if condition in ("default", "stored_shift"):
        view = "clean" if condition == "default" else "shift"
        return item[view + "_previous"], item[view + "_current"]
    if condition in ("redBlock", "redAgent", "redAnchor"):
        palette = CLEAN_RGB.copy()
        palette[{"redBlock": 0, "redAgent": 1, "redAnchor": 2}[condition]] = [255, 0, 0]
        rendered = [render_state(env, state, palette, row["render_seed"])
                    for state in row["source_states"][:2]]
        if not all(check_state(actual, state, dataset.manifest["source_reset_tolerances"])[0]
                   for (_, actual), state in zip(rendered, row["source_states"][:2])):
            raise ValueError("Red-condition rendering changed source physics")
        if not check_state(rendered[1][1], row["canonical_render_state"],
                           dataset.manifest["canonical_reset_tolerances"])[0]:
            raise ValueError("Red-condition center differs from canonical render state")
        return tuple(as_tensor(image) for image, _ in rendered)
    if condition not in ("gaussian", "salt_pepper", "blur", "dark"):
        raise ValueError("Unknown event-probe condition")
    clean = np.stack([((item["clean_" + key].permute(1, 2, 0).numpy() + 1)
                       * 127.5).round().clip(0, 255).astype(np.uint8)
                      for key in ("previous", "current")])
    shifted = corrupt_frames(clean, condition, strength, seed + row["sample_id"])
    return tuple(as_tensor(image) for image in shifted)


@torch.no_grad()
def extract_features(encoder, dataset, condition, ec, device, strength=None):
    vectors, densities, gates = [], [], []
    env = new_env(encoder.image_size) if condition.startswith("red") else None
    try:
        for start in range(0, len(dataset), ec["batch_size"]):
            end = min(start + ec["batch_size"], len(dataset))
            pairs = [condition_pair(dataset, i, condition, ec["seed"], env, strength)
                     for i in range(start, end)]
            previous = torch.stack([pair[0] for pair in pairs]).to(device)
            current = torch.stack([pair[1] for pair in pairs]).to(device)
            result = encoder(previous, current)
            x = (result["global_token"] if ec["feature"] == "global"
                 else result["patch_tokens"].mean(1))
            vectors.append(x.cpu().numpy())
            densities.extend(result["patch_density"].mean((1, 2)).cpu().tolist())
            if result["gate"] is not None:
                gates.extend(result["gate"].mean((1, 2)).cpu().tolist())
    finally:
        if env is not None:
            env.close()
    return np.concatenate(vectors), float(np.mean(densities)), (
        float(np.mean(gates)) if gates else None)


@torch.no_grad()
def temporal_diagnostics(model, dataset, device, batch_size):
    keys = ("teacher_prediction", "student_prediction", "persistence", "patch_robustness")
    totals = {key: 0. for key in keys}
    mode, count = model.loss_cfg["distance"], 0
    for start in range(0, len(dataset), batch_size):
        batch = to_device(default_collate([dataset[i] for i in
                          range(start, min(start + batch_size, len(dataset)))]), device)
        now = model.teacher(batch["clean_previous"], batch["clean_current"])
        future = model.teacher(batch["clean_current"], batch["clean_next"])
        shift = model.student(batch["shift_previous"], batch["shift_current"])
        results = {
            "teacher_prediction": representation_distance(model.F(now["global_token"]), future["global_token"], mode),
            "student_prediction": representation_distance(model.F(shift["global_token"]), future["global_token"], mode),
            "persistence": representation_distance(now["global_token"], future["global_token"], mode),
            "patch_robustness": representation_distance(model.q(shift["patch_tokens"]), now["patch_tokens"], mode),
        }
        n = len(batch["clean_current"])
        for key, value in results.items():
            totals[key] += float(value) * n
        count += n
    return {"temporal/" + key: value / count for key, value in totals.items()}


def evaluate_event_encoder(cfg):
    cfg = copy.deepcopy(cfg)
    ec = cfg["evaluation"]
    if not ec["checkpoint"]:
        raise ValueError("evaluation.checkpoint is required")
    if ec["weights"] not in ("student", "teacher") or ec["feature"] not in ("global", "patch_mean"):
        raise ValueError("Unsupported evaluation.weights or evaluation.feature")
    seed_all(ec["seed"])
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    device = device_for(ec["device"])
    path = absolute(ec["checkpoint"])
    checkpoint = load_checkpoint(path)
    if checkpoint.get("schema_version") != "tracka-event-v1":
        raise ValueError("Checkpoint is not an RGB/Event encoder")
    trained = checkpoint["config"]
    model = EventTeacherStudent(trained["model"], trained["loss"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    encoder = getattr(model, ec["weights"])
    data = {split: EventWindowDataset(cfg["dataset"]["output_path"], split,
                                      trained["model"]["image_size"])
            for split in ("train", "validation", "test")}
    manifest_hash = file_hash(data["train"].root / "manifest.json")
    if checkpoint["manifest_sha256"] != manifest_hash:
        raise ValueError("Probe dataset differs from checkpoint training dataset")
    for dataset in data.values():
        cap = ec["max_states_per_split"]
        if cap and len(dataset.rows) > cap:
            indices = np.random.RandomState(ec["seed"]).choice(len(dataset.rows), cap, replace=False)
            dataset.rows = [dataset.rows[i] for i in sorted(indices)]
    out = unique_output(ec["output_dir"], "probe_event")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Event probe output directory is not empty")
    out.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["name"] = cfg["logging"]["name"] or (
        f"trackA-event-probe-{trained['model']['fusion']}-{ec['weights']}-{ec['feature']}")
    metadata = {"config": cfg, "checkpoint": str(path), "checkpoint_sha256": file_hash(path),
                "training_step": checkpoint["global_step"], "manifest_sha256": manifest_hash,
                "evaluated_samples": {split: [(r["trajectory_id"], r["timestep"])
                                              for r in ds.rows] for split, ds in data.items()}, **git_info()}
    with RunLogger(out, cfg["logging"], metadata) as logger:
        def features(split, condition="default", strength=None):
            return extract_features(encoder, data[split], condition, ec, device, strength)

        clean = {split: features(split)[0] for split in data}
        targets = {split: physical_targets(data[split].rows) for split in data}
        probe = fit_probe(clean["train"], targets["train"], clean["validation"],
                          targets["validation"], ec["ridge_alphas"])
        probe.save(out / "physical_probe.npz")
        test = data["test"]
        rows, baselines = [], {}

        def report(name, vectors, density, gate, kind=None, strength=None):
            for segment in ("full", "early", "middle", "late"):
                ids = (np.arange(len(test)) if segment == "full" else
                       np.array([i for i, row in enumerate(test.rows)
                                 if row["segment"] == segment], dtype=int))
                if not len(ids):
                    continue
                metrics = state_metrics(probe, vectors[ids], targets["test"][ids])
                if name == "default":
                    baselines[segment] = metrics["standardized_mse"]
                rows.append({"condition": name, "segment": segment, "samples": len(ids),
                             "corruption_type": kind, "strength": strength,
                             "event_density": density, "gate_mean": gate,
                             "latent_distance": float(np.sqrt(((vectors[ids] - clean["test"][ids]) ** 2).mean())),
                             "degradation": metrics["standardized_mse"] - baselines[segment], **metrics})
                if segment == "full":
                    logger.log({"train/global_step": checkpoint["global_step"],
                                **{f"probe/{name}/{key}": value for key, value in metrics.items()
                                   if value is not None},
                                f"event/probe_density_{name}": density})

        for condition in ("default", "stored_shift", "redBlock", "redAgent", "redAnchor"):
            report(condition, *features("test", condition),
                   kind="mixed" if condition == "stored_shift" else None)
        for kind, grid in (("gaussian", ec["gaussian_grid"]),
                           ("salt_pepper", ec["salt_pepper_grid"]),
                           ("blur", ec["blur_grid"]), ("dark", ec["dark_grid"])):
            for strength in grid:
                report(f"{kind}_{strength:g}", *features("test", kind, float(strength)),
                       kind, float(strength))
        diagnostics = temporal_diagnostics(model, test, device, ec["batch_size"])
        diagnostics.update(spectrum_metrics([torch.from_numpy(clean["test"])], "collapse/probe_feature"))
        logger.log({"train/global_step": checkpoint["global_step"], **diagnostics})
        logger.table("results", rows)
        write_json(out / "diagnostics.json", diagnostics)
        with (out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_results(rows, out / "robustness.png")
        logger.robustness_plots(rows, out / "robustness.png")
    print(f"Event probe results saved: {out}", flush=True)
    return out
