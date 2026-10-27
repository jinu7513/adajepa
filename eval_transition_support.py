"""Paired PushT latent prediction diagnostics on seen and held-out colors.

Only ``train`` videos enter training. This script reads the separate ``val``
episodes, including redBlock/purpleBlock rendered by the held-out stage.
"""

import argparse
import csv
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets.pusht_dset import FixedColorPushTDataset
from datasets.traj_dset import TrajSlicerDataset


DOMAINS = ("default", "blue", "green", "redBlock", "purpleBlock")


def load_model(model_dir, device):
    cfg = OmegaConf.load(model_dir / "hydra.yaml")
    checkpoint = torch.load(model_dir / "checkpoints/model_latest.pth", map_location="cpu", weights_only=False)
    required = ("encoder", "predictor", "action_encoder", "proprio_encoder")
    if any(key not in checkpoint for key in required):
        raise ValueError("Checkpoint lacks required model modules")
    model = hydra.utils.instantiate(
        cfg.model, encoder=checkpoint["encoder"], predictor=checkpoint["predictor"],
        action_encoder=checkpoint["action_encoder"],
        proprio_encoder=checkpoint["proprio_encoder"], decoder=None,
        proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=cfg.num_proprio_repeat,
    ).to(device).eval()
    experiment_path = model_dir / "experiment.yaml"
    model.identity_only = bool(OmegaConf.load(experiment_path).get("identity_only", False)) if experiment_path.exists() else False
    return model, cfg


def make_loader(cfg, root, domain, batch_size, workers, max_episodes):
    transform = hydra.utils.instantiate(cfg.env.dataset.transform)
    dataset = FixedColorPushTDataset(
        color_domain=domain, n_rollout=max_episodes, transform=transform,
        data_path=str(root / "pusht_noise/val"), normalize_action=True,
        with_velocity=True,
    )
    sliced = TrajSlicerDataset(dataset, num_frames=12, frameskip=int(cfg.frameskip))
    return DataLoader(sliced, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def shuffled_context(model, z, k):
    """Break transition pairings, not ordering (mean aggregation ignores order)."""
    if k == 0:
        return model.support_context_from_sequence(z[:, :10], 0)
    query_index = 8
    visual = z[..., :model.predictor.visual_dim].mean(dim=2)
    action = z[:, :, 0, -model.action_dim:]
    previous = visual[:, query_index - k:query_index]
    following = visual[:, query_index - k + 1:query_index + 1]
    permuted = torch.roll(following, shifts=1, dims=1)
    reference = None
    if model.predictor.feature_mode == "residual":
        reference = torch.stack([
            model._reference_prediction([z[:, j] for j in range(i + 1)])
            for i in range(query_index - k, query_index)
        ], dim=1)
    return model.predictor.context_from_support(
        previous, action[:, query_index - k:query_index], permuted,
        torch.ones(z.shape[0], k, dtype=torch.bool, device=z.device), reference,
    )


def prediction_errors(model, z, action, context, valid):
    # At t=8, frames 6:9 are available. Frames 9:12 are targets only.
    history = z[:, 6:9].clone()
    end = -model.action_dim
    errors = []
    for horizon in range(3):
        if getattr(model.predictor, "uses_support_film", False):
            predicted = model.predict(history, support_context=context, context_valid=valid)[:, -1]
        else:
            predicted = model.predict(history)[:, -1]
        target = z[:, 9 + horizon]
        errors.append(((predicted[..., :end] - target[..., :end]) ** 2).mean(dim=(1, 2)))
        if horizon < 2:
            next_action = action[:, 9 + horizon:10 + horizon]
            with_action = model.replace_actions_from_z(predicted[:, None].clone(), next_action)
            history = torch.cat([history[:, 1:], with_action], dim=1)
    return torch.stack(errors, dim=1)


def evaluate(model, cfg, data_root, max_batches, batch_size, workers, max_episodes):
    results = []
    support_model = getattr(model.predictor, "uses_support_film", False) and not model.identity_only
    device = next(model.parameters()).device
    for domain_index, domain in enumerate(DOMAINS):
        other = DOMAINS[(domain_index + 1) % len(DOMAINS)]
        # Identical physical windows because each slicer uses the same RNG seed.
        np.random.seed(701)
        loader = make_loader(cfg, data_root, domain, batch_size, workers, max_episodes)
        np.random.seed(701)
        swapped_loader = make_loader(cfg, data_root, other, batch_size, workers, max_episodes)
        if not np.array_equal(loader.dataset.slices, swapped_loader.dataset.slices):
            raise ValueError("Paired-color evaluation windows differ")
        accumulators = {}
        with torch.no_grad():
            for batch_index, (batch, swapped_batch) in enumerate(zip(loader, swapped_loader)):
                if batch_index >= max_batches:
                    break
                observations, actions, _ = batch
                swapped_observations, swapped_actions, _ = swapped_batch
                if not torch.equal(actions, swapped_actions):
                    raise ValueError("Paired-color actions differ")
                observations = {key: value.to(device) for key, value in observations.items()}
                swapped_observations = {key: value.to(device) for key, value in swapped_observations.items()}
                actions = actions.to(device)
                z = model.encode(observations, actions)
                swapped_z = model.encode(swapped_observations, actions) if support_model else None
                k_values = (0, 2, 4, 8) if support_model else (0,)
                for k in k_values:
                    correct_context, correct_valid = (
                        model.support_context_from_sequence(z[:, :10], k)
                        if support_model else (None, None)
                    )
                    modes = ("correct", "swapped", "shuffled", "identity") if k > 0 else ("identity",)
                    for mode in modes:
                        if mode == "identity":
                            context = z.new_zeros(z.shape[0], model.predictor.context_dim) if support_model else None
                            valid = torch.zeros(z.shape[0], dtype=torch.bool, device=device) if support_model else None
                        elif mode == "correct":
                            context, valid = correct_context, correct_valid
                        elif mode == "swapped":
                            context, valid = model.support_context_from_sequence(swapped_z[:, :10], k)
                        else:
                            context, valid = shuffled_context(model, z, k)
                        errors = prediction_errors(model, z, actions, context, valid)
                        key = (k, mode)
                        if key not in accumulators:
                            accumulators[key] = {"sum": torch.zeros(3, dtype=torch.float64),
                                                 "context_norm": 0.0, "context_distance": 0.0,
                                                 "n": 0}
                        accumulators[key]["sum"] += errors.double().sum(dim=0).cpu()
                        if support_model:
                            accumulators[key]["context_norm"] += float(context.float().norm(dim=-1).sum())
                            accumulators[key]["context_distance"] += float(
                                (context.float() - correct_context.float()).norm(dim=-1).sum()
                            )
                        accumulators[key]["n"] += errors.shape[0]
        for (k, mode), item in accumulators.items():
            mean = item["sum"] / item["n"]
            results.append({"domain": domain, "domain_split": "heldout" if domain in DOMAINS[3:] else "seen",
                            "support_k": k, "support_mode": mode, "samples": item["n"],
                            "one_step_mse": float(mean[0]), "open_loop_h2_mse": float(mean[1]),
                            "open_loop_h3_mse": float(mean[2]),
                            "open_loop_mean_mse": float(mean.mean()),
                            "context_norm": item["context_norm"] / item["n"] if support_model else 0.0,
                            "context_distance_to_correct": item["context_distance"] / item["n"] if support_model else 0.0})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=50)
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    if args.max_batches < 1 or args.max_episodes < 1:
        parser.error("max-batches and max-episodes must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model_dir.resolve(strict=True), device)
    results = evaluate(model, cfg, args.data_dir.resolve(strict=True), args.max_batches,
                       args.batch_size, args.num_workers, args.max_episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=f"offline-{args.model_dir.name}",
                         dir=str(args.output_dir), config={"model_dir": str(args.model_dir),
                                                           "max_batches": args.max_batches,
                                                           "max_episodes": args.max_episodes})
        for row in results:
            label = f"{row['domain']}/k{row['support_k']}/{row['support_mode']}"
            run.summary[f"{label}/one_step_mse"] = row["one_step_mse"]
            run.summary[f"{label}/open_loop_mean_mse"] = row["open_loop_mean_mse"]
        run.finish()
    print(f"Saved {len(results)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
