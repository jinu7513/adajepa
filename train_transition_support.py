"""Train query-only, causal support-conditioned AdaJEPA from a released checkpoint."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

from models.support_transition import SupportFiLMPredictor, SupportVWorldModel


def make_model(source, source_cfg, experiment):
    encoder = source["encoder"]
    action_encoder = source["action_encoder"]
    proprio_encoder = source["proprio_encoder"]
    reference = source["predictor"]
    source_predictor_cfg = OmegaConf.to_container(source_cfg.predictor, resolve=True)
    source_predictor_cfg.pop("_target_", None)
    visual_dim = encoder.emb_dim
    action_dim = action_encoder.emb_dim * int(source_cfg.num_action_repeat)
    predictor = SupportFiLMPredictor(
        num_patches=reference.pos_embedding.shape[1] // int(source_cfg.num_hist),
        num_frames=int(source_cfg.num_hist),
        dim=reference.pos_embedding.shape[-1],
        visual_dim=visual_dim,
        action_dim=action_dim,
        context_dim=int(experiment.context_dim),
        feature_mode=str(experiment.feature_mode),
        film_layers=int(experiment.film_layers),
        affine_type=str(experiment.affine_type),
        **source_predictor_cfg,
    )
    missing, unexpected = predictor.load_state_dict(reference.state_dict(), strict=False)
    if unexpected or any(not key.startswith(("feature_encoder.", "support_aggregator.", "film.")) for key in missing):
        raise ValueError(f"Source predictor mismatch: missing={missing}, unexpected={unexpected}")
    if experiment.feature_mode == "residual":
        predictor.attach_reference(reference)
    if experiment.identity_only:
        if experiment.feature_mode != "delta":
            raise ValueError("Identity-only control uses the delta architecture without conditioning")
        for name, parameter in predictor.named_parameters():
            parameter.requires_grad_(not name.startswith(("feature_encoder.", "support_aggregator.", "film.", "reference_predictor.")))
    elif not experiment.train_base_predictor:
        for name, parameter in predictor.named_parameters():
            parameter.requires_grad_(name.startswith(("feature_encoder.", "support_aggregator.", "film.")))
    else:
        for name, parameter in predictor.named_parameters():
            parameter.requires_grad_(not name.startswith("reference_predictor."))
    for module in (encoder, action_encoder, proprio_encoder):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    model = SupportVWorldModel(
        image_size=int(source_cfg.img_size), num_hist=int(source_cfg.num_hist),
        num_pred=int(source_cfg.num_pred), encoder=encoder,
        proprio_encoder=proprio_encoder, action_encoder=action_encoder,
        decoder=None, predictor=predictor,
        proprio_dim=int(source_cfg.proprio_emb_dim),
        action_dim=int(source_cfg.action_emb_dim),
        concat_dim=int(source_cfg.concat_dim),
        num_action_repeat=int(source_cfg.num_action_repeat),
        num_proprio_repeat=int(source_cfg.num_proprio_repeat),
        train_encoder=False, train_predictor=True, train_decoder=False,
        support_k=8,
    )
    return model


def make_data(source_cfg, data_root, batch_size, workers):
    from datasets.pusht_dset import load_pusht_slice_train_val
    transform = hydra.utils.instantiate(source_cfg.env.dataset.transform)
    datasets, _ = load_pusht_slice_train_val(
        transform=transform, n_rollout=None,
        data_path=str(data_root / "pusht_noise"),
        normalize_action=bool(source_cfg.normalize_action),
        num_hist=9, num_pred=1, frameskip=int(source_cfg.frameskip),
        with_velocity=True, color_domains=("default", "blue", "green"),
    )
    return {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=workers > 0)
        for split, dataset in datasets.items()
    }


def query_loss(model, observations, actions, support_k, z=None):
    # The final image is the detached target. Support ends at the current query input.
    if z is None:
        with torch.no_grad():
            z = model.encode(observations, actions)
    context, valid = model.support_context_from_sequence(z, support_k)
    prediction = model.predict(z[:, -4:-1], support_context=context, context_valid=valid)
    target = z[:, -1].detach()
    end = -model.action_dim
    loss = torch.nn.functional.mse_loss(prediction[:, -1, :, :end], target[:, :, :end])
    visual_dim = model.predictor.visual_dim
    visual_loss = torch.nn.functional.mse_loss(
        prediction[:, -1, :, :visual_dim], target[:, :, :visual_dim]
    )
    return loss, visual_loss, context, valid


def reference_query_loss(model, z):
    """Same query and frozen target, with identity/zero support FiLM."""
    context = z.new_zeros(z.shape[0], model.predictor.context_dim)
    valid = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    predicted = model.predict(z[:, -4:-1], support_context=context, context_valid=valid)
    return torch.nn.functional.mse_loss(
        predicted[:, -1, :, :-model.action_dim], z[:, -1, :, :-model.action_dim]
    )


def metric_write(path, payload):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val, cfg, source_dir,
                    resume_epoch=None, next_batch=0, train_sum=0.0, train_steps=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch, "encoder": model.encoder,
        "predictor": model.predictor,
        "action_encoder": model.action_encoder,
        "proprio_encoder": model.proprio_encoder,
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val": best_val,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "experiment_config": OmegaConf.to_container(cfg, resolve=True),
        "source_dir": str(source_dir),
        "checkpoint_version": 1,
        "resume_epoch": epoch + 1 if resume_epoch is None else resume_epoch,
        "next_batch": next_batch,
        "train_sum": train_sum,
        "train_steps": train_steps,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "conf/transition_support.yaml")
    parser.add_argument("--feature-mode", choices=("delta", "residual"))
    parser.add_argument("--affine-type", choices=("mlp", "linear"))
    parser.add_argument("--train-base-predictor", action="store_true")
    parser.add_argument("--no-train-base-predictor", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--wandb-name")
    args = parser.parse_args()
    if args.train_base_predictor and args.no_train_base_predictor:
        parser.error("Base predictor trainability flags conflict")
    cfg = OmegaConf.load(args.config)
    for key in ("feature_mode", "affine_type", "epochs", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.train_base_predictor:
        cfg.train_base_predictor = True
    if args.no_train_base_predictor:
        cfg.train_base_predictor = False
    if args.identity_only:
        cfg.identity_only = True
        cfg.train_base_predictor = True
        cfg.support_k_values = [0]
    source_dir = args.source_dir.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    if run_dir == source_dir or source_dir in run_dir.parents:
        raise ValueError("Run output must not overwrite the released source checkpoint")
    source_cfg = OmegaConf.load(source_dir / "hydra.yaml")
    if (int(source_cfg.num_hist), int(source_cfg.num_pred), int(source_cfg.frameskip),
            int(source_cfg.concat_dim)) != (3, 1, 5, 1):
        raise ValueError("Expected the released PushT 3/1/5/concat1 temporal layout")
    if not (data_dir / "pusht_noise/transition_color_manifest.json").is_file():
        raise FileNotFoundError("Paired colors were not prepared by the A/B/C prepare stage")
    latest_source = source_dir / "checkpoints/model_latest.pth"
    source = torch.load(latest_source, map_location="cpu", weights_only=False)
    required = {"encoder", "predictor", "action_encoder", "proprio_encoder"}
    if required - source.keys():
        raise ValueError(f"Source checkpoint is missing {sorted(required - source.keys())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    latest = checkpoint_dir / "model_latest.pth"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        if saved["experiment_config"] != OmegaConf.to_container(cfg, resolve=True) or saved["source_dir"] != str(source_dir):
            raise ValueError("Resume configuration or source checkpoint differs")
        model = SupportVWorldModel(
            image_size=int(source_cfg.img_size), num_hist=int(source_cfg.num_hist), num_pred=1,
            encoder=saved["encoder"], proprio_encoder=saved["proprio_encoder"],
            action_encoder=saved["action_encoder"], decoder=None, predictor=saved["predictor"],
            proprio_dim=int(source_cfg.proprio_emb_dim), action_dim=int(source_cfg.action_emb_dim),
            concat_dim=1, num_action_repeat=int(source_cfg.num_action_repeat),
            num_proprio_repeat=int(source_cfg.num_proprio_repeat), train_encoder=False,
            train_predictor=True, train_decoder=False,
        )
        start_epoch = int(saved.get("resume_epoch", int(saved["epoch"]) + 1))
        start_batch = int(saved.get("next_batch", 0))
        prior_sum = float(saved.get("train_sum", 0.0))
        prior_steps = int(saved.get("train_steps", 0))
        best_val = float(saved["best_val"])
    else:
        model = make_model(source, source_cfg, cfg)
        start_epoch = 1
        start_batch = 0
        prior_sum = 0.0
        prior_steps = 0
        best_val = float("inf")
    # This is the config read by plan.py.load_model; keep the source observation/action layout.
    plan_cfg = OmegaConf.create(OmegaConf.to_container(source_cfg, resolve=True))
    with open_dict(plan_cfg):
        plan_cfg.model._target_ = "models.support_transition.SupportVWorldModel"
        plan_cfg.model.support_k = 0 if cfg.identity_only else 8
        plan_cfg.model.train_encoder = False
        plan_cfg.model.train_predictor = True
        plan_cfg.model.train_decoder = False
        plan_cfg.has_decoder = False
        plan_cfg.predictor._target_ = "models.support_transition.SupportFiLMPredictor"
    (run_dir / "hydra.yaml").write_text(OmegaConf.to_yaml(plan_cfg, resolve=True), encoding="utf-8")
    (run_dir / "experiment.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Support training must run on an allocated CUDA compute node")
    model = model.to(device)
    params = [p for p in model.predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=str(cfg.precision) == "fp16")
    if str(cfg.precision) not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be bf16, fp16 or fp32")
    amp_enabled = str(cfg.precision) != "fp32"
    amp_dtype = torch.bfloat16 if str(cfg.precision) == "bf16" else torch.float16
    if latest.exists():
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
    # Trajectory windows and per-epoch row order must survive a restarted job.
    np.random.seed(seed)
    loaders = make_data(source_cfg, data_dir, int(cfg.batch_size), int(cfg.num_workers))
    if latest.exists():
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    import wandb
    identity = run_dir / "wandb_identity.json"
    wandb_id = json.loads(identity.read_text(encoding="utf-8"))["run_id"] if identity.exists() else None
    run = wandb.init(project=str(cfg.wandb_project), name=args.wandb_name or run_dir.name,
                     config={**OmegaConf.to_container(cfg, resolve=True), "source_dir": str(source_dir)},
                     dir=str(run_dir), id=wandb_id, resume="allow")
    if not identity.exists():
        identity.write_text(json.dumps({"run_id": run.id}) + "\n", encoding="utf-8")
    metric_path = run_dir / "metrics.jsonl"
    accumulation = int(cfg.gradient_accumulation)
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    try:
        for epoch in range(start_epoch, int(cfg.epochs) + 1):
            started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = prior_sum if epoch == start_epoch else 0.0
            steps = prior_steps if epoch == start_epoch else 0
            first_batch = start_batch if epoch == start_epoch else 0
            sample_count = len(loaders["train"].dataset)
            batch_size = int(cfg.batch_size)
            original_batches = len(loaders["train"])
            order = torch.randperm(
                sample_count, generator=torch.Generator().manual_seed(seed + epoch)
            ).tolist()
            remaining = Subset(loaders["train"].dataset, order[first_batch * batch_size:])
            epoch_loader = DataLoader(
                remaining, batch_size=batch_size, shuffle=False,
                num_workers=int(cfg.num_workers), pin_memory=True,
                persistent_workers=int(cfg.num_workers) > 0,
                generator=torch.Generator().manual_seed(seed + epoch + 10_000),
            )
            train_k_values = ([0] if cfg.identity_only
                              else [int(k) for k in cfg.support_k_values if int(k) > 0])
            if not train_k_values:
                raise ValueError("At least one nonzero support K is required for conditioned training")
            for index, batch in enumerate(epoch_loader, start=first_batch):
                observations, actions, _ = batch
                observations = {key: value.to(device, non_blocking=True) for key, value in observations.items()}
                actions = actions.to(device, non_blocking=True)
                k = train_k_values[index % len(train_k_values)]
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                    loss, visual_loss, context, valid = query_loss(model, observations, actions, int(k))
                group_start = index - (index % accumulation)
                group_size = min(accumulation, original_batches - group_start)
                scaler.scale(loss / group_size).backward()
                optimizer_boundary = (index + 1) % accumulation == 0 or index + 1 == original_batches
                if optimizer_boundary:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.max_grad_norm))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach())
                steps += 1
                if optimizer_boundary and (index + 1) % int(cfg.save_every_batches) == 0:
                    save_checkpoint(latest, model, optimizer, scaler, epoch - 1, best_val,
                                    cfg, source_dir, resume_epoch=epoch, next_batch=index + 1,
                                    train_sum=total, train_steps=steps)
                if index % 20 == 0:
                    print(f"epoch={epoch} batch={index}/{original_batches} k={k} loss={float(loss):.6f}", flush=True)
            model.eval()
            validation = {k: [] for k in cfg.support_k_values}
            reference_losses = []
            diagnostic_context = None
            with torch.no_grad():
                for index, batch in enumerate(loaders["valid"]):
                    if index >= int(cfg.validation_batches):
                        break
                    observations, actions, _ = batch
                    observations = {key: value.to(device, non_blocking=True) for key, value in observations.items()}
                    actions = actions.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                        z = model.encode(observations, actions)
                        reference_losses.append(float(reference_query_loss(model, z)))
                        for k in cfg.support_k_values:
                            value, _, current_context, _ = query_loss(model, observations, actions, int(k), z=z)
                            validation[int(k)].append(float(value))
                            if int(k) == 8:
                                diagnostic_context = current_context.detach()
            val_metrics = {f"val/loss_k{k}": sum(v) / len(v) for k, v in validation.items() if v}
            if not val_metrics:
                raise ValueError("Validation dataset yielded no batches")
            val_mean = sum(val_metrics.values()) / len(val_metrics)
            with torch.no_grad():
                if diagnostic_context is None:
                    diagnostic_context = current_context.detach()
                scales = [projection(diagnostic_context).chunk(2, dim=-1)
                          for projection in model.predictor.film.values()]
                film_gamma = sum(float(gamma.abs().mean()) for gamma, _ in scales) / len(scales)
                film_beta = sum(float(beta.abs().mean()) for _, beta in scales) / len(scales)
            row = {"epoch": epoch, "train/loss": total / steps, "val/loss": val_mean,
                   "val/reference_query_loss": sum(reference_losses) / len(reference_losses),
                   "train/last_grad_norm": float(grad_norm),
                   "film/mean_abs_gamma": film_gamma, "film/mean_abs_beta": film_beta,
                   "duration_seconds": time.monotonic() - started,
                   "trainable_parameters": sum(p.numel() for p in params), **val_metrics}
            metric_write(metric_path, row)
            run.log(row, step=epoch)
            best_val = min(best_val, val_mean)
            save_checkpoint(checkpoint_dir / f"model_{epoch}.pth", model, optimizer, scaler,
                            epoch, best_val, cfg, source_dir)
            save_checkpoint(latest, model, optimizer, scaler, epoch, best_val, cfg, source_dir)
            if val_mean == best_val:
                save_checkpoint(checkpoint_dir / "model_best.pth", model, optimizer, scaler,
                                epoch, best_val, cfg, source_dir)
            print(json.dumps(row, sort_keys=True), flush=True)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
