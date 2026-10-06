import copy
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .common import (absolute, file_hash, git_info, load_checkpoint, restore_rng,
                     rng_state, seed_all, write_json)
from .data import FourViewDataset
from .logging import RunLogger
from .model import RobustMAE, resolved_mask_ratios, resolved_variance_config


def device_for(choice):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if choice == "auto" else torch.device(choice)


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def objective_for(objective, step):
    return ("render_only" if step % 2 == 0 else "corruption_only") if objective == "robust_alternating" else objective


def unique_output(cfg, prefix):
    return absolute(cfg) if cfg else absolute(f"tracka_outputs/{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}")


@torch.no_grad()
def validate(model, dataset, cfg, device):
    prior = rng_state()
    was_training = model.training
    model.eval()
    seed_all(9871)
    values = {}
    try:
        modes = ["render_only", "corruption_only"] if cfg["objective"] == "robust_alternating" else [cfg["objective"]]
        count = min(len(dataset), cfg["validation_batches"] * cfg["batch_size"])
        for start in range(0, count, cfg["batch_size"]):
            batch = to_device(default_collate([dataset[i] for i in range(start, min(start + cfg["batch_size"], count))]), device)
            for mode in modes:
                _, logs = model.objective(batch, mode)
                for key, value in logs.items():
                    values.setdefault("val/" + mode + "/" + key, []).append((float(value), len(batch["clean"])))
        return {k: sum(v*n for v,n in rows)/sum(n for _,n in rows) for k, rows in values.items()}
    finally:
        model.train(was_training)
        restore_rng(prior)


def train(cfg):
    cfg = copy.deepcopy(cfg)
    tc = cfg["training"]
    objective = tc["objective"]
    if objective not in ("clean_mae", "render_only", "corruption_only", "robust_alternating"):
        raise ValueError("Unknown training.objective")
    if tc["total_optimizer_updates"] < 1 or tc["batch_size"] < 1:
        raise ValueError("Training update count and batch size must be positive")
    loss_cfg = resolved_variance_config(cfg["loss"])
    if (loss_cfg["lambda_variance"] or loss_cfg["lambda_sigreg"]) and tc["batch_size"] < 2:
        raise ValueError("Variance/SIGReg regularization requires training.batch_size >= 2")
    if objective == "robust_alternating" and (tc["total_optimizer_updates"] % 2 or not tc["equal_render_corruption_schedule"]):
        raise ValueError("robust_alternating requires an even update budget and equal schedule")
    for n in (tc["save_every"], tc["validate_every"], cfg["logging"]["log_every"], cfg["logging"]["image_every"]):
        if n < 1:
            raise ValueError("Logging/checkpoint intervals must be positive")
    torch.set_num_threads(tc["cpu_threads"])
    seed_all(tc["seed"])
    device = device_for(tc["device"])
    data = FourViewDataset(cfg["dataset"]["output_path"], "train", cfg["model"]["image_size"])
    valid = FourViewDataset(cfg["dataset"]["output_path"], "validation", cfg["model"]["image_size"])
    manifest_hash = file_hash(data.root / "manifest.json")
    model = RobustMAE(cfg["model"], cfg["decoder"], cfg["loss"], cfg["mask"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    checkpoint = load_checkpoint(absolute(tc["resume"])) if tc["resume"] else None
    step, draws, encoded_images, decoded_images, resume_id = 0, 0, 0, 0, None
    if checkpoint:
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume dataset manifest differs")
        for key in ("model", "decoder"):
            if checkpoint["config"][key] != cfg[key]:
                raise ValueError("Resume configuration differs: " + key)
        if resolved_variance_config(checkpoint["config"]["loss"]) != resolved_variance_config(cfg["loss"]):
            raise ValueError("Resume configuration differs: loss")
        if resolved_mask_ratios(checkpoint["config"]["mask"]) != resolved_mask_ratios(cfg["mask"]):
            raise ValueError("Resume configuration differs: mask")
        for key in ("objective", "batch_size", "seed", "lr", "weight_decay"):
            if checkpoint["config"]["training"][key] != tc[key]:
                raise ValueError("Resume training configuration differs: " + key)
        for key in ("project", "entity"):
            if checkpoint["config"]["logging"][key] != cfg["logging"][key]:
                raise ValueError("Resume must use the original W&B " + key)
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, draws, resume_id = checkpoint["global_step"], checkpoint["samples_drawn"], checkpoint.get("wandb_run_id")
        encoded_images = checkpoint.get("encoded_images", 0)
        decoded_images = checkpoint.get("decoded_images", 0)
        out = Path(checkpoint["output_dir"])
        if tc["output_dir"] and absolute(tc["output_dir"]) != out:
            raise ValueError("Resume must use original output directory")
        # Reject a stale checkpoint if newer durable metrics already exist.
        log_path = out / "metrics.jsonl"
        if log_path.exists():
            last = max((json.loads(line).get("train/global_step", 0) for line in log_path.read_text().splitlines()), default=0)
            if last > step:
                raise ValueError("Checkpoint predates logged steps; use the latest checkpoint")
    else:
        out = unique_output(tc["output_dir"], objective)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Training directory not empty; use resume or a new output_dir")
    if step >= tc["total_optimizer_updates"]:
        raise ValueError("Resume budget must exceed checkpoint global_step")
    out.mkdir(parents=True, exist_ok=True)
    run_name = f"trackA-{objective}-{cfg['decoder']['conditioning']}-mask{cfg['mask']['ratio']*100:g}-seed{tc['seed']}"
    ratios = resolved_mask_ratios(cfg["mask"])
    if ratios["render_only"] != ratios["clean_mae"] or ratios["corruption_only"] != ratios["clean_mae"]:
        run_name = (f"trackA-{objective}-{cfg['decoder']['conditioning']}"
                    f"-render{ratios['render_only']*100:g}-corr{ratios['corruption_only']*100:g}-seed{tc['seed']}")
    if loss_cfg["lambda_sigreg"]:
        run_name += f"-sigreg-{loss_cfg['sigreg_feature']}-w{loss_cfg['lambda_sigreg']:g}"
    cfg["logging"]["name"] = cfg["logging"]["name"] or run_name
    write_json(out / "config.json", cfg)
    metadata = {"config": cfg, **git_info(), "manifest_sha256": manifest_hash,
                "output_dir": str(out), "normalization": "uint8 / 127.5 - 1"}
    started = time.monotonic()
    with RunLogger(out, cfg["logging"], metadata, resume_id) as logger:
        if checkpoint:
            restore_rng(checkpoint["rng"])
        model.train()
        while step < tc["total_optimizer_updates"]:
            mode = objective_for(objective, step)
            ids = torch.randint(len(data), (tc["batch_size"],)).tolist()
            batch = to_device(default_collate([data[i] for i in ids]), device)
            optimizer.zero_grad(set_to_none=True)
            loss, logs = model.objective(batch, mode)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            draws += tc["batch_size"]
            encoded_images += tc["batch_size"] * {"clean_mae": 1, "render_only": 3, "corruption_only": 2}[mode]
            decoded_images += tc["batch_size"] * (4 if mode == "render_only" else 1)
            # Sample each alternating objective equally, even with an even log interval.
            mode_step = (step + 1) // 2 if objective == "robust_alternating" else step
            if step <= 2 or mode_step % cfg["logging"]["log_every"] == 0 or step >= tc["total_optimizer_updates"] - 1:
                values = {k: float(v.detach()) for k, v in logs.items()}
                values.update({"train/global_step": step, "train/cycle": step // 2 if objective == "robust_alternating" else 0,
                               "train/lr": optimizer.param_groups[0]["lr"], "train/grad_norm": float(grad_norm),
                               "train/samples_drawn": draws, "train/elapsed_seconds": time.monotonic()-started})
                values.update({"train/encoded_images": encoded_images, "train/decoded_images": decoded_images})
                logger.log(values)
                print(f"step={step} objective={mode} loss={float(loss.detach()):.6f}", flush=True)
            if step % tc["validate_every"] == 0 or step == tc["total_optimizer_updates"]:
                logger.log({"train/global_step": step, **validate(model, valid, tc, device)})
            if step % cfg["logging"]["image_every"] == 0 or step == tc["total_optimizer_updates"]:
                prior = rng_state()
                model.eval()
                preview = model.preview({k: v[:1] for k, v in batch.items()}, mode=mode)[0]
                image = ((preview.clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
                logger.image(image, step)
                model.train()
                restore_rng(prior)
            if step % tc["save_every"] == 0 or step == tc["total_optimizer_updates"]:
                state = {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
                         "optimizer": optimizer.state_dict(), "epoch": draws // len(data),
                         "global_step": step, "cycle": step // 2 if objective == "robust_alternating" else 0,
                         "next_update_type": objective_for(objective, step), "samples_drawn": draws,
                         "encoded_images": encoded_images, "decoded_images": decoded_images,
                         "config": cfg, "schema_version": data.manifest["schema_version"],
                         "manifest_sha256": manifest_hash, **git_info(), "rng": rng_state(),
                         "wandb_run_id": logger.run_id, "output_dir": str(out)}
                temporary = out / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(out / "checkpoint_latest.pt")
    print(f"Track A run saved: {out}", flush=True)
    return out
