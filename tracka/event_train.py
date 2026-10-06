"""Training loop for the trajectory-contiguous RGB/Event experiment."""

import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.data import default_collate

from .common import (absolute, file_hash, git_info, load_checkpoint, restore_rng, rng_state,
                     seed_all, write_json)
from .event_data import EventWindowDataset
from .event_model import EventTeacherStudent
from .logging import RunLogger
from .train import device_for, to_device, unique_output


def active_encoder_parameters(encoder):
    """Count only modules on this fusion path, for control-efficiency reports."""
    modules = [encoder.shared_blocks, encoder.final_norm]
    standalone = [encoder.pos_embed]
    if encoder.cls_token is not None:
        standalone.append(encoder.cls_token)
    if encoder.fusion != "event_only":
        modules += [encoder.rgb_embed, encoder.rgb_blocks, encoder.rgb_norm]
    if encoder.fusion != "rgb_only":
        modules += [encoder.event_embed, encoder.event_blocks, encoder.event_norm]
    if encoder.gate is not None:
        modules.append(encoder.gate)
    return sum(p.numel() for p in standalone) + sum(
        p.numel() for module in modules for p in module.parameters())


def spectrum_metrics(features, prefix):
    """Effective rank on an evaluation *collection*, not one small minibatch."""
    x = torch.cat(features).double().cpu()
    if len(x) < 2:
        return {}
    x = x - x.mean(0, keepdim=True)
    per_dimension = x.square().mean(0)
    eigenvalues = torch.linalg.eigvalsh(x.T @ x / (len(x) - 1)).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 1e-12:
        return {prefix + "/effective_rank": 0., prefix + "/top1_fraction": 0.,
                prefix + "/samples": float(len(x)),
                prefix + "/mean_variance": float(per_dimension.mean())}
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0]
    effective = (-(nonzero * nonzero.log()).sum()).exp()
    return {prefix + "/effective_rank": float(effective),
            prefix + "/effective_rank_fraction": float(effective / min(len(x)-1, x.shape[1])),
            prefix + "/top1_fraction": float(probabilities[-1]),
            prefix + "/top5_fraction": float(probabilities[-5:].sum()),
            prefix + "/mean_variance": float(per_dimension.mean()),
            prefix + "/variance_dim_min": float(per_dimension.min()),
            prefix + "/variance_dim_median": float(per_dimension.median()),
            prefix + "/variance_dim_max": float(per_dimension.max()),
            prefix + "/samples": float(len(x))}


def spectrum_arrays(features):
    x = torch.cat(features).double().cpu()
    x -= x.mean(0, keepdim=True)
    return {"variance_per_dimension": x.square().mean(0).float(),
            "covariance_eigenvalues": torch.linalg.eigvalsh(
                x.T @ x / max(len(x) - 1, 1)).clamp_min(0).float(),
            "samples": len(x)}


@torch.no_grad()
def validate_event(model, dataset, cfg, device, details_path=None):
    prior = rng_state()
    was_training = model.training
    model.eval()
    seed_all(9871)
    accum, count = {}, 0
    global_features, patch_features = [], []
    kind_densities = {}
    clean_shift_distances = []
    try:
        limit = min(len(dataset), cfg["rank_samples"])
        batch_size = cfg["batch_size"]
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            batch = to_device(default_collate([dataset[i] for i in range(start, end)]), device)
            clean = model.student(batch["clean_previous"], batch["clean_current"])
            global_features.append(clean["global_token"].detach().cpu())
            patch_features.append(clean["patch_tokens"].mean(1).detach().cpu())
            shift = model.student(batch["shift_previous"], batch["shift_current"])
            clean_shift_distances.append(float((clean["global_token"] - shift["global_token"])
                                               .square().mean().sqrt()) * (end - start))
            per_item_density = shift["patch_density"].mean((1, 2)).detach().cpu().tolist()
            for row, density in zip(dataset.rows[start:end], per_item_density):
                kind_densities.setdefault(row["shift_kind"], []).append(density)
            if start < cfg["validation_batches"] * batch_size:
                _, logs = model.objective(batch)
                for key, value in logs.items():
                    if value.ndim == 0:
                        accum[key] = accum.get(key, 0.) + float(value) * (end - start)
                count += end - start
        metrics = {"val/" + key: value / count for key, value in accum.items()} if count else {}
        metrics.update(spectrum_metrics(global_features, "collapse/global"))
        metrics.update(spectrum_metrics(patch_features, "collapse/patch_mean"))
        metrics["latent/clean_corrupt_distance_val"] = sum(clean_shift_distances) / limit
        if details_path is not None:
            torch.save({"global": spectrum_arrays(global_features),
                        "patch_mean": spectrum_arrays(patch_features)}, details_path)
        for kind, densities in kind_densities.items():
            metrics["event/density_val_" + kind] = sum(densities) / len(densities)
        return metrics
    finally:
        model.train(was_training)
        restore_rng(prior)


def train_event_encoder(cfg):
    cfg = copy.deepcopy(cfg)
    tc = cfg["training"]
    if (tc["total_optimizer_updates"] < 1 or tc["batch_size"] < 2
            or tc["rank_samples"] < 2 or tc["validate_every"] < 1
            or tc["save_every"] < 1 or cfg["logging"]["log_every"] < 1):
        raise ValueError("Invalid Event training/logging budget")
    if not 0 <= tc["ema_momentum"] < 1:
        raise ValueError("training.ema_momentum must be in [0,1)")
    torch.set_num_threads(tc["cpu_threads"])
    seed_all(tc["seed"])
    device = device_for(tc["device"])
    data = EventWindowDataset(cfg["dataset"]["output_path"], "train", cfg["model"]["image_size"])
    valid = EventWindowDataset(cfg["dataset"]["output_path"], "validation", cfg["model"]["image_size"])
    manifest_hash = file_hash(data.root / "manifest.json")
    model = EventTeacherStudent(cfg["model"], cfg["loss"]).to(device)
    trainable = list(model.student.parameters()) + list(model.q.parameters()) + list(model.F.parameters())
    if model.sigreg_projector is not None:
        trainable += list(model.sigreg_projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=tc["lr"], weight_decay=tc["weight_decay"])
    checkpoint = load_checkpoint(absolute(tc["resume"])) if tc["resume"] else None
    step, draws, resume_id = 0, 0, None
    if checkpoint is not None:
        if checkpoint.get("schema_version") != "tracka-event-v1" or checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume checkpoint schema or dataset differs")
        for key in ("model", "loss"):
            if checkpoint["config"][key] != cfg[key]:
                raise ValueError("Resume Event configuration differs: " + key)
        for key in ("batch_size", "seed", "lr", "weight_decay", "ema_momentum"):
            if checkpoint["config"]["training"][key] != tc[key]:
                raise ValueError("Resume training configuration differs: " + key)
        for key in ("project", "entity"):
            if checkpoint["config"]["logging"][key] != cfg["logging"][key]:
                raise ValueError("Resume W&B project/entity differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, draws, resume_id = (checkpoint["global_step"], checkpoint["samples_drawn"],
                                  checkpoint.get("wandb_run_id"))
        out = Path(checkpoint["output_dir"])
        if tc["output_dir"] and Path(tc["output_dir"]).resolve() != out:
            raise ValueError("Resume must use its original output directory")
        log_file = out / "metrics.jsonl"
        if log_file.exists() and max((json.loads(s).get("train/global_step", 0)
                                      for s in log_file.read_text().splitlines()), default=0) > step:
            raise ValueError("Checkpoint is older than durable metrics")
    else:
        out = unique_output(tc["output_dir"], "event_encoder")
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Event training output directory is not empty")
    if step >= tc["total_optimizer_updates"]:
        raise ValueError("Resume budget must exceed saved step")
    out.mkdir(parents=True, exist_ok=True)
    cfg["logging"]["name"] = cfg["logging"]["name"] or (
        f"trackA-event-{cfg['model']['fusion']}-{cfg['model']['rgb_input_mode']}"
        f"-{cfg['model']['event_mode']}-seed{tc['seed']}")
    write_json(out / "config.json", cfg)
    metadata = {"config": cfg, "manifest_sha256": manifest_hash,
                "output_dir": str(out), "schema_version": "tracka-event-v1",
                "normalization": "uint8 / 127.5 - 1",
                "active_student_encoder_parameters": active_encoder_parameters(model.student),
                "student_encoder_parameters_total": sum(p.numel() for p in model.student.parameters()),
                "q_parameters": sum(p.numel() for p in model.q.parameters()),
                "F_parameters": sum(p.numel() for p in model.F.parameters()), **git_info()}
    started = time.monotonic()
    with RunLogger(out, cfg["logging"], metadata, resume_id) as logger:
        if checkpoint is not None:
            restore_rng(checkpoint["rng"])
        model.train()
        while step < tc["total_optimizer_updates"]:
            indices = torch.randint(len(data), (tc["batch_size"],)).tolist()
            batch = to_device(default_collate([data[i] for i in indices]), device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = model.objective(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite RGB/Event training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, tc["grad_clip"],
                                                       error_if_nonfinite=True)
            optimizer.step()
            model.update_teacher(tc["ema_momentum"])
            step += 1
            draws += tc["batch_size"]
            if step <= 2 or step % cfg["logging"]["log_every"] == 0 or step == tc["total_optimizer_updates"]:
                values = {key: float(value.detach()) for key, value in logs.items()}
                values.update({"train/global_step": step, "train/samples_drawn": draws,
                               "train/elapsed_seconds": time.monotonic() - started,
                               "train/grad_norm": float(grad_norm),
                               "train/lr": optimizer.param_groups[0]["lr"],
                               "teacher/ema_momentum": tc["ema_momentum"]})
                logger.log(values)
                print(f"step={step} loss={float(loss):.6f} robust={float(logs['loss/robust']):.6f}"
                      f" temporal={float(logs['loss/temporal']):.6f}", flush=True)
            if step % tc["validate_every"] == 0 or step == tc["total_optimizer_updates"]:
                logger.log({"train/global_step": step, **validate_event(
                    model, valid, tc, device, out / f"collapse_{step:08d}.pt")})
            if step % tc["save_every"] == 0 or step == tc["total_optimizer_updates"]:
                state = {"schema_version": "tracka-event-v1", "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(), "config": cfg,
                         "manifest_sha256": manifest_hash, "global_step": step,
                         "samples_drawn": draws, "output_dir": str(out),
                         "rng": rng_state(), "wandb_run_id": logger.run_id, **git_info()}
                temporary = out / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(out / "checkpoint_latest.pt")
    print(f"Event encoder run saved: {out}", flush=True)
    return out
