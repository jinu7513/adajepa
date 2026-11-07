"""Test whether frozen PushT features retain pose under a red-block shift.

The paired red/gray images and the oracle adapter are diagnostic only. They
must never be used as inputs to the proposed online MPC adapter.
"""

import argparse
from contextlib import closing
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from datasets.img_transforms import default_transform
from scripts.generate_transition_colors import paired_frames


COLORS = {"default": "LightSlateGray", "redBlock": "Red"}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_states(data_root, split, count, seed):
    root = data_root / split
    states = torch.load(root / "states.pth", map_location="cpu", weights_only=False).numpy()
    velocities = torch.load(root / "velocities.pth", map_location="cpu", weights_only=False).numpy()
    with open(root / "seq_lengths.pkl", "rb") as handle:
        lengths = pickle.load(handle)
    shapes_path = root / "shapes.pkl"
    if shapes_path.is_file():
        with open(shapes_path, "rb") as handle:
            shapes = pickle.load(handle)
    else:
        shapes = ["T"] * len(lengths)
    if not (len(states) == len(velocities) == len(lengths) == len(shapes)):
        raise ValueError(f"Inconsistent episode counts in {root}")
    if states.shape[-1] < 5 or velocities.shape[-1] < 2:
        raise ValueError(f"Expected PushT state/velocity features in {root}")
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(lengths), size=min(count, len(lengths)), replace=False)
    samples = []
    for episode in chosen:
        frame = int(rng.randint(0, int(lengths[episode])))
        physical = np.concatenate(
            [states[episode, frame, :5], velocities[episode, frame, :2]]
        ).astype(np.float32)
        samples.append((int(episode), frame, str(shapes[episode]), physical))
    return samples


def render_pairs(samples, image_size):
    gray, red, physical, ids = [], [], [], []
    changed_fraction = []
    for episode, frame, shape, state in samples:
        with closing(paired_frames(state[None], shape, image_size, COLORS)) as pairs:
            pair = next(pairs)
        g, r = pair["default"], pair["redBlock"]
        if g.shape != r.shape or g.dtype != np.uint8 or r.dtype != np.uint8:
            raise ValueError("Paired renders must be matching uint8 RGB arrays")
        changed_fraction.append(float(np.mean(np.any(g != r, axis=-1))))
        gray.append(g)
        red.append(r)
        physical.append(state[:5])
        ids.append({"episode": episode, "frame": frame, "shape": shape})
    if not changed_fraction or min(changed_fraction) <= 0:
        raise ValueError("At least one paired render did not change color")
    return gray, red, np.stack(physical), ids, changed_fraction


def encode_images(encoder, images, image_size, device, batch_size):
    transform = default_transform(image_size)
    result = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([
                transform(torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0)
                for img in images[start : start + batch_size]
            ]).to(device)
            latent = encoder(batch)
            if latent.ndim != 3 or latent.shape[1] != 1:
                raise ValueError(f"Expected one global visual token, got {tuple(latent.shape)}")
            result.append(latent[:, 0].float().cpu())
    return torch.cat(result).numpy()


def pose_targets(states):
    angle = states[:, 4]
    return np.column_stack([
        states[:, :4], np.sin(angle), np.cos(angle)
    ]).astype(np.float64)


def ridge_predict(train_x, train_y, test_x, ridge=1.0):
    """A fixed linear readout; labels are used only for representation diagnosis."""
    mean = train_x.mean(axis=0)
    scale = np.maximum(train_x.std(axis=0), 1e-6)
    x = (train_x - mean) / scale
    query = (test_x - mean) / scale
    x = np.column_stack([x, np.ones(len(x))])
    query = np.column_stack([query, np.ones(len(query))])
    penalty = np.eye(x.shape[1]) * ridge
    penalty[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ train_y)
    return query @ weights


def pose_metrics(pred, actual):
    if pred.shape != actual.shape or pred.shape[1] != 6:
        raise ValueError("Pose arrays must have columns agent xy, block xy, sin, cos")
    pred_angle = np.arctan2(pred[:, 4], pred[:, 5])
    true_angle = np.arctan2(actual[:, 4], actual[:, 5])
    angle_error = np.angle(np.exp(1j * (pred_angle - true_angle)))
    return {
        "agent_position_rmse_px": float(np.sqrt(np.mean((pred[:, :2] - actual[:, :2]) ** 2))),
        "block_position_rmse_px": float(np.sqrt(np.mean((pred[:, 2:4] - actual[:, 2:4]) ** 2))),
        "block_angle_mae_deg": float(np.mean(np.abs(angle_error)) * 180 / np.pi),
    }


def paired_metrics(red, gray):
    distances = np.linalg.norm(red - gray, axis=1)
    all_distances = np.linalg.norm(red[:, None, :] - gray[None, :, :], axis=-1)
    order = np.argsort(all_distances, axis=1)
    truth = np.arange(len(red))[:, None]
    return {
        "paired_mse": float(np.mean((red - gray) ** 2)),
        "paired_l2_mean": float(np.mean(distances)),
        "cross_color_retrieval_top1": float(np.mean(order[:, :1] == truth)),
        "cross_color_retrieval_top5": float(np.mean(np.any(order[:, :5] == truth, axis=1))),
    }


class ResidualAdapter(nn.Module):
    def __init__(self, dim, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return z + self.net(z)


def fit_oracle(red_train, gray_train, dim, width, steps, lr, device, seed):
    """Privileged paired alignment upper bound, not an online adaptation method."""
    torch.manual_seed(seed)
    adapter = ResidualAdapter(dim, width).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    red = torch.from_numpy(red_train).to(device)
    gray = torch.from_numpy(gray_train).to(device)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ids = torch.randint(len(red), (min(64, len(red)),), generator=generator).to(device)
        # Keep gray features near identity so the oracle does not ruin default.
        loss = F.mse_loss(adapter(red[ids]), gray[ids])
        loss = loss + 0.1 * F.mse_loss(adapter(gray[ids]), gray[ids])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    adapter.eval()
    return adapter


def apply_adapter(adapter, values, device):
    with torch.no_grad():
        return adapter(torch.from_numpy(values).to(device)).cpu().numpy()


def save_contact_sheet(path, gray, red, limit=8):
    strips = [np.concatenate((g, r), axis=1) for g, r in zip(gray[:limit], red[:limit])]
    sheet = np.concatenate(strips, axis=0)
    if not cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-dir", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="Parent of train/ and val/ in pusht_noise")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--oracle-steps", type=int, default=200)
    parser.add_argument("--oracle-width", type=int, default=32)
    parser.add_argument("--oracle-lr", type=float, default=1e-3)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args(argv)
    if min(args.train_samples, args.val_samples, args.oracle_width, args.encode_batch_size) <= 0:
        parser.error("Sample counts, oracle width, and encoding batch size must be positive")
    if args.oracle_steps < 0 or args.oracle_lr <= 0:
        parser.error("Oracle steps must be nonnegative and learning rate positive")
    model_dir = args.source_model_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = model_dir / "checkpoints" / "model_latest.pth"
    for path in (model_dir / "hydra.yaml", checkpoint_path,
                 data_root / "train" / "states.pth", data_root / "val" / "states.pth"):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(model_dir / "hydra.yaml")
    if cfg.env.name not in ("pusht", "pushobj"):
        raise ValueError(f"Expected PushT/PushObj source checkpoint, got {cfg.env.name}")
    if cfg.env.dataset.transform._target_ != "datasets.img_transforms.default_transform":
        raise ValueError("Source checkpoint uses an unexpected image transform")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = checkpoint["encoder"].to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    del checkpoint
    image_size = int(cfg.img_size)
    features = {}
    states = {}
    metadata = {}
    previews = None
    for split, count, offset in (("train", args.train_samples, 0),
                                 ("val", args.val_samples, 100000)):
        samples = sample_states(data_root, split, count, args.seed + offset)
        gray, red, states[split], ids, changed = render_pairs(samples, image_size)
        if split == "val":
            previews = (gray, red)
        features[split] = {
            "gray": encode_images(encoder, gray, image_size, device, args.encode_batch_size),
            "red": encode_images(encoder, red, image_size, device, args.encode_batch_size),
        }
        metadata[split] = {"samples": ids, "changed_pixel_fraction_mean": float(np.mean(changed))}
        print(f"Encoded {split}: {len(samples)} paired physical states", flush=True)
    save_contact_sheet(output_dir / "paired_render_preview.png", *previews)
    del encoder

    train, val = features["train"], features["val"]
    if train["gray"].shape != train["red"].shape or val["gray"].shape != val["red"].shape:
        raise ValueError("Gray/red feature shapes differ")
    dim = train["gray"].shape[1]
    y_train, y_val = pose_targets(states["train"]), pose_targets(states["val"])
    metrics = {
        "feature_dim": dim,
        "gray_feature_std_mean": float(np.mean(np.std(val["gray"], axis=0))),
        "red_feature_std_mean": float(np.mean(np.std(val["red"], axis=0))),
        "raw": paired_metrics(val["red"], val["gray"]),
        "gray_probe_on_gray": pose_metrics(
            ridge_predict(train["gray"], y_train, val["gray"]), y_val
        ),
        "gray_probe_on_red": pose_metrics(
            ridge_predict(train["gray"], y_train, val["red"]), y_val
        ),
        "red_probe_on_red": pose_metrics(
            ridge_predict(train["red"], y_train, val["red"]), y_val
        ),
    }
    mean_shift = np.mean(train["gray"] - train["red"], axis=0)
    metrics["mean_shift"] = paired_metrics(val["red"] + mean_shift, val["gray"])
    metrics["gray_probe_on_mean_shift_red"] = pose_metrics(
        ridge_predict(train["gray"], y_train, val["red"] + mean_shift), y_val
    )
    adapter = fit_oracle(train["red"], train["gray"], dim,
                         args.oracle_width, args.oracle_steps, args.oracle_lr,
                         device, args.seed)
    aligned_val = apply_adapter(adapter, val["red"], device)
    metrics["oracle"] = paired_metrics(aligned_val, val["gray"])
    metrics["oracle"]["train_paired_mse"] = float(np.mean(
        (apply_adapter(adapter, train["red"], device) - train["gray"]) ** 2
    ))
    metrics["oracle"]["gray_identity_mse"] = float(np.mean(
        (apply_adapter(adapter, val["gray"], device) - val["gray"]) ** 2
    ))
    metrics["gray_probe_on_oracle_red"] = pose_metrics(
        ridge_predict(train["gray"], y_train, aligned_val), y_val
    )
    output = {
        "experiment": "redBlock frozen-encoder alignment feasibility; oracle uses paired colors",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "data_root": str(data_root),
        "seed": args.seed,
        "oracle_width": args.oracle_width,
        "oracle_steps": args.oracle_steps,
        "oracle_lr": args.oracle_lr,
        "metrics": metrics,
        "sample_metadata": metadata,
    }
    result_path = output_dir / "results.json"
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name="redBlock-latent-alignment-oracle-seed{}".format(args.seed),
            config={k: output[k] for k in ("experiment", "source_checkpoint_sha256", "seed",
                                          "oracle_width", "oracle_steps", "oracle_lr")},
            dir=str(output_dir),
        )
        flattened = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                flattened.update({key + "/" + subkey: val for subkey, val in value.items()})
            else:
                flattened[key] = value
        run.log(flattened)
        run.log({"paired_render_preview": wandb.Image(str(output_dir / "paired_render_preview.png"))})
        run.finish()
    print(json.dumps({"metrics": metrics, "results": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
