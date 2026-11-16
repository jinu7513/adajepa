"""Frozen state probe on paired PushT color videos; labels are evaluation-only."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader

from datasets.img_transforms import default_transform


TRAIN_COLORS = ("default", "blue", "green")
EVAL_COLORS = ("default", "purpleBlock", "redBlock")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episodes(root, split, limit):
    folder = root / split
    states = torch.load(folder / "states.pth", map_location="cpu").numpy()
    with (folder / "seq_lengths.pkl").open("rb") as stream:
        lengths = pickle.load(stream)
    return [(i, int(lengths[i]) // 2, states[i, int(lengths[i]) // 2, :5])
            for i in range(min(limit, len(lengths)))]


def encode_color(encoder, transform, root, split, selected, color, batch_size, device):
    features = []
    for start in range(0, len(selected), batch_size):
        images = []
        for episode, frame, _ in selected[start:start + batch_size]:
            if color == "source_default":
                path = root / split / "obses" / "episode_{:03d}.mp4".format(episode)
            else:
                path = root / split / "colors" / color / "obses" / "episode_{:03d}.mp4".format(episode)
            reader = VideoReader(str(path), num_threads=1)
            if frame >= len(reader):
                raise ValueError("Video shorter than physical trajectory: {}".format(path))
            image = reader[frame]
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image.asnumpy())
            tensor = image.permute(2, 0, 1).float() / 255.0
            images.append(transform(tensor))
        with torch.no_grad():
            output = encoder(torch.stack(images).to(device))
        features.append(output.detach().float().mean(dim=1).cpu().numpy())
    return np.concatenate(features)


def targets(selected):
    state = np.stack([item[2] for item in selected]).astype(np.float64)
    angle = state[:, 4]
    return np.column_stack([state[:, :4], np.sin(angle), np.cos(angle)])


def contact_labels(root, split, selected):
    """Label only agent-block geometry collisions, never wall/goal collisions."""
    from env.pusht.pusht_env import PushTEnv
    folder = root / split
    shapes_path = folder / "shapes.pkl"
    if shapes_path.exists():
        with shapes_path.open("rb") as stream:
            shapes = pickle.load(stream)
    else:
        shapes = ["T"] * (max(item[0] for item in selected) + 1)
    labels = []
    for episode, _, state in selected:
        environment = PushTEnv(shape=shapes[episode], reset_to_state=state, with_velocity=False)
        try:
            environment.reset()
            labels.append(int(any(
                bool(agent_shape.shapes_collide(block_shape).points)
                for agent_shape in environment.agent.shapes
                for block_shape in environment.block.shapes
            )))
        finally:
            environment.close()
    return np.asarray(labels, dtype=np.int64)


def average_precision(scores, labels):
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float((precision * ordered).sum() / positives)


def bootstrap_interval(values, aggregate, draws=500):
    rng = np.random.default_rng(0)
    estimates = []
    for _ in range(draws):
        selected = values[rng.integers(0, len(values), len(values))]
        estimates.append(float(aggregate(selected)))
    return [float(v) for v in np.percentile(estimates, (2.5, 97.5))]


def fit_ridge(features, labels, alpha=1.0):
    x_mean = features.mean(0)
    x_std = np.maximum(features.std(0), 1e-6)
    y_mean = labels.mean(0)
    y_std = np.maximum(labels.std(0), 1e-6)
    x = (features - x_mean) / x_std
    y = (labels - y_mean) / y_std
    x = x - x.mean(0)
    y = y - y.mean(0)
    weight = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    return x_mean, x_std, y_mean, y_std, weight


def predict_ridge(fitted, features):
    x_mean, x_std, y_mean, y_std, weight = fitted
    return ((features - x_mean) / x_std) @ weight * y_std + y_mean


def metrics(pred, truth, features, clean=None):
    agent = float(np.sqrt(np.mean(np.sum((pred[:, :2] - truth[:, :2]) ** 2, axis=1))))
    block = float(np.sqrt(np.mean(np.sum((pred[:, 2:4] - truth[:, 2:4]) ** 2, axis=1))))
    delta = np.arctan2(pred[:, 4], pred[:, 5]) - np.arctan2(truth[:, 4], truth[:, 5])
    angle = float(np.degrees(np.mean(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))))
    std = features.std(0)
    centered = features - features.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False) ** 2
    probability = singular / max(singular.sum(), 1e-20)
    effective_rank = float(np.exp(-(probability * np.log(probability + 1e-20)).sum()))
    result = {"agent_rmse_px": agent, "block_rmse_px": block,
              "block_angle_mae_deg": angle, "latent_std_mean": float(std.mean()),
              "latent_norm_mean": float(np.linalg.norm(features, axis=1).mean()),
              "latent_effective_rank": effective_rank}
    agent_sq = np.sum((pred[:, :2] - truth[:, :2]) ** 2, axis=1)
    block_sq = np.sum((pred[:, 2:4] - truth[:, 2:4]) ** 2, axis=1)
    angle_abs = np.degrees(np.abs(np.arctan2(np.sin(delta), np.cos(delta))))
    result["agent_rmse_ci95_px"] = bootstrap_interval(agent_sq, lambda x: np.sqrt(np.mean(x)))
    result["block_rmse_ci95_px"] = bootstrap_interval(block_sq, lambda x: np.sqrt(np.mean(x)))
    result["block_angle_mae_ci95_deg"] = bootstrap_interval(angle_abs, np.mean)
    if clean is not None:
        paired = np.linalg.norm(features - clean, axis=1)
        between = np.linalg.norm(clean - np.roll(clean, 1, axis=0), axis=1)
        result["same_state_color_distance"] = float(paired.mean())
        result["between_state_clean_distance"] = float(between.mean())
        result["normalized_color_distance"] = float(paired.mean() / max(between.mean(), 1e-12))
        cosine = np.sum(features * clean, axis=1) / np.maximum(
            np.linalg.norm(features, axis=1) * np.linalg.norm(clean, axis=1), 1e-12)
        result["paired_cosine_mean"] = float(cosine.mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=256)
    parser.add_argument("--val-episodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-colors", nargs="+", default=list(TRAIN_COLORS))
    parser.add_argument("--eval-colors", nargs="+", default=list(EVAL_COLORS))
    parser.add_argument("--contact-probe", action="store_true")
    args = parser.parse_args()
    if min(args.train_episodes, args.val_episodes, args.batch_size) < 2:
        raise ValueError("Episode limits and batch size must be at least 2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = checkpoint["encoder"].to(device).eval()
    encoder.requires_grad_(False)
    transform = default_transform(224)
    train = episodes(args.data_root, "train", args.train_episodes)
    val = episodes(args.data_root, "val", args.val_episodes)
    y_train = targets(train)
    y_val = targets(val)
    x_train = np.concatenate([
        encode_color(encoder, transform, args.data_root, "train", train, color,
                     args.batch_size, device) for color in args.train_colors])
    fitted = fit_ridge(x_train.astype(np.float64), np.tile(y_train, (len(args.train_colors), 1)))
    contact_fit = None
    if args.contact_probe:
        train_contact = contact_labels(args.data_root, "train", train)
        val_contact = contact_labels(args.data_root, "val", val)
        contact_fit = fit_ridge(x_train.astype(np.float64),
                                np.tile(train_contact[:, None], (len(args.train_colors), 1)))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    encoded = {color: encode_color(encoder, transform, args.data_root, "val", val, color,
                                   args.batch_size, device).astype(np.float64)
               for color in args.eval_colors}
    clean_color = args.eval_colors[0]
    results = {}
    with (output / "per_episode.jsonl").open("w", encoding="utf-8") as stream:
        for color, features in encoded.items():
            pred = predict_ridge(fitted, features)
            results[color] = metrics(pred, y_val, features, encoded[clean_color] if color != clean_color else None)
            contact_score = None
            if contact_fit is not None:
                contact_score = predict_ridge(contact_fit, features)[:, 0]
                results[color]["agent_block_contact_ap"] = average_precision(contact_score, val_contact)
                results[color]["agent_block_contact_positives"] = int(val_contact.sum())
            for j, (episode, frame, _) in enumerate(val):
                row = {"color": color, "episode": episode, "frame": frame,
                       "prediction": pred[j].tolist(), "target": y_val[j].tolist()}
                if contact_score is not None:
                    row["agent_block_contact_label"] = int(val_contact[j])
                    row["agent_block_contact_score"] = float(contact_score[j])
                stream.write(json.dumps(row) + "\n")
    payload = {"checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_sha256": sha256(args.checkpoint),
               "data_root": str(args.data_root.resolve()),
               "train_episodes": len(train), "val_episodes": len(val),
               "train_colors": args.train_colors, "eval_colors": args.eval_colors,
               "ridge_alpha": 1.0, "results": results}
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
