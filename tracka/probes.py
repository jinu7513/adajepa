"""Clean-trained linear probes; shifted test data never changes probe fitting."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from planning.image_corruption import corrupt_frames
from .common import absolute, file_hash, git_info, load_checkpoint, seed_all, write_json
from .data import (CLEAN_RGB, FourViewDataset, check_state, new_env,
                   render_state, sample_colors)
from .logging import RunLogger
from .model import ScratchEncoder
from .train import device_for, unique_output


class LinearProbe:
    """Ridge-regularized affine probe, trained in standardized feature/target units."""
    def fit(self, x, y, alpha):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.x_mean, self.x_std = x.mean(0), np.maximum(x.std(0), 1e-6)
        self.y_mean, self.y_std = y.mean(0), np.maximum(y.std(0), 1e-6)
        xn = (x - self.x_mean) / self.x_std
        yn = (y - self.y_mean) / self.y_std
        if len(xn) < xn.shape[1]:
            self.weight = xn.T @ np.linalg.solve(xn @ xn.T + alpha * np.eye(len(xn)), yn)
        else:
            self.weight = np.linalg.solve(xn.T @ xn + alpha * np.eye(xn.shape[1]), xn.T @ yn)
        self.alpha = float(alpha)
        return self

    def predict(self, x):
        return (((x - self.x_mean) / self.x_std) @ self.weight) * self.y_std + self.y_mean

    def save(self, path):
        np.savez(path, weight=self.weight, x_mean=self.x_mean, x_std=self.x_std,
                 y_mean=self.y_mean, y_std=self.y_std, alpha=self.alpha)


def fit_probe(train_x, train_y, val_x, val_y, alphas):
    if len(train_x) < 2 or len(val_x) < 1:
        raise ValueError("Insufficient probe train/validation data")
    best, score = None, float("inf")
    for alpha in alphas:
        if alpha <= 0:
            raise ValueError("Ridge alpha must be positive")
        probe = LinearProbe().fit(train_x, train_y, alpha)
        mse = np.mean(((probe.predict(val_x) - val_y) / probe.y_std) ** 2)
        if mse < score:
            best, score = probe, mse
    return best


def physical_targets(rows):
    s = np.asarray([r["canonical_render_state"] for r in rows])
    return np.column_stack([s[:, :4], np.cos(s[:, 4]), np.sin(s[:, 4]), s[:, 5:7]])


def regression_metrics(pred, target):
    error = (pred - target) ** 2
    denom = ((target - target.mean(0)) ** 2).sum(0)
    r2 = [float(1 - error[:, j].sum() / d) if d > 1e-12 else None for j, d in enumerate(denom)]
    return {"mse": float(error.mean()), **{f"r2_{j}": value for j, value in enumerate(r2)}}


def state_metrics(probe, features, targets):
    pred = probe.predict(features)
    angle = np.arctan2(pred[:, 5], pred[:, 4]) - np.arctan2(targets[:, 5], targets[:, 4])
    angle = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
    error = (pred - targets) ** 2
    metrics = regression_metrics(pred, targets)
    metrics.pop("mse")  # Do not mix physical units into a headline raw-state MSE.
    metrics.update({"standardized_mse": float((error / probe.y_std**2).mean()),
                    "agent_position_mse": float(error[:, :2].mean()),
                    "block_position_mse": float(error[:, 2:4].mean()),
                    "velocity_mse": float(error[:, 6:8].mean()),
                    "angle_mae_rad": float(angle.mean()),
                    "angle_mae_deg": float(np.degrees(angle).mean())})
    return metrics


def load_reference(cfg, device):
    ec = cfg["evaluation"]
    feature = ec.get("feature", "patch_mean")
    if feature not in ("patch_mean", "cls"):
        raise ValueError("evaluation.feature must be patch_mean or cls")
    if feature == "cls" and ec["encoder"] == "dino":
        raise ValueError("evaluation.feature=cls is only supported for scratch checkpoint or random encoders")
    if ec["encoder"] == "checkpoint":
        if not ec["checkpoint"]:
            raise ValueError("evaluation.checkpoint is required")
        path = absolute(ec["checkpoint"])
        state = load_checkpoint(path)
        encoder = ScratchEncoder(**state["config"]["model"])
        encoder.load_state_dict(state["encoder"])
        image_size = encoder.image_size
        metadata = {"checkpoint_path": str(path), "checkpoint_sha256": file_hash(path),
                    "training_objective": state["config"]["training"]["objective"],
                    "training_seed": state["config"]["training"]["seed"],
                    "training_model": state["config"]["model"],
                    "training_decoder": state["config"]["decoder"],
                    "training_mask": state["config"]["mask"],
                    "training_loss": state["config"]["loss"],
                    "training_updates": state["global_step"],
                    "training_manifest_sha256": state["manifest_sha256"],
                    "encoder_input_resolution": image_size, "patch_size": encoder.patch_size,
                    "tokens": encoder.num_patches}
    elif ec["encoder"] == "random":
        encoder = ScratchEncoder(**cfg["model"])
        image_size = encoder.image_size
        metadata = {"training_objective": "random", "encoder_input_resolution": image_size,
                    "patch_size": encoder.patch_size, "tokens": encoder.num_patches,
                    "training_model": cfg["model"], "training_seed": ec["seed"]}
    elif ec["encoder"] == "dino":
        from models.dino import DinoV2Encoder
        from torchvision.transforms import Resize
        base = DinoV2Encoder(name="dinov2_vits14", feature_key="x_norm_patchtokens")
        image_size = cfg["model"]["image_size"]
        resolution = (image_size // 16) * base.patch_size

        class DinoReference(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = base
                self.resize = Resize(resolution)

            def forward(self, x):
                return self.base(self.resize(x))

        encoder = DinoReference()
        metadata = {"training_objective": "dino_frozen", "encoder_input_resolution": resolution,
                    "patch_size": base.patch_size, "tokens": (resolution // base.patch_size)**2}
    else:
        raise ValueError("evaluation.encoder must be checkpoint, random, or dino")
    if feature == "cls" and encoder.cls_token is None:
        raise ValueError("evaluation.feature=cls requires an encoder trained with model.use_cls=true")
    encoder = encoder.to(device).eval()
    encoder.requires_grad_(False)
    metadata.update({"loader_normalization": "uint8 / 127.5 - 1", "loader_image_size": image_size,
                     "pooling": "mean over patch tokens" if feature == "patch_mean" else "CLS token",
                     "feature": feature, "encoder": ec["encoder"]})
    return encoder, image_size, metadata


@torch.no_grad()
def extract_features(encoder, images, device, batch_size, feature="patch_mean"):
    if feature not in ("patch_mean", "cls"):
        raise ValueError("evaluation.feature must be patch_mean or cls")
    if feature == "cls" and (not isinstance(encoder, ScratchEncoder) or encoder.cls_token is None):
        raise ValueError("CLS feature extraction requires a scratch encoder with model.use_cls=true")
    result = []
    for start in range(0, len(images), batch_size):
        array = np.stack(images[start:start + batch_size])
        x = torch.from_numpy(array).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1
        if feature == "cls":
            representation = encoder.forward_features(x, return_cls=True)["cls_token"]
        else:
            representation = encoder(x).mean(1)
        result.append(representation.cpu().numpy())
    return np.concatenate(result)


class ImageSequence:
    """Lazy sequence: evaluate one image batch at a time, never retain all RGBs."""
    def __init__(self, length, getter):
        self.length, self.getter = length, getter

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.getter(i) for i in range(*index.indices(self.length))]
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.getter(index)


def read_images(dataset, view):
    def get(index):
        with Image.open(dataset.root / dataset.rows[index]["images"][view]) as im:
            image = np.array(im.convert("RGB"), copy=True)
        if image.shape != (dataset.image_size, dataset.image_size, 3):
            raise ValueError("Probe image resolution differs from encoder contract")
        return image
    return ImageSequence(len(dataset), get)


def plot_results(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = [r for r in rows if r["segment"] == "full"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    color_rows = [r for r in full if r["corruption_type"] is None]
    axes[0].bar([r["condition"] for r in color_rows], [r["standardized_mse"] for r in color_rows])
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Standardized state MSE (lower is better)")
    for kind in ("gaussian", "salt_pepper", "blur"):
        points = [r for r in full if r["corruption_type"] == kind]
        # Each strength uses different units; use a separate normalized grid index.
        axes[1].plot(range(len(points)), [r["standardized_mse"] for r in points], marker="o", label=kind)
    axes[1].set_xlabel("Severity grid index (actual strengths in results.csv)")
    axes[1].set_ylabel("Standardized state MSE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate(cfg):
    cfg = copy.deepcopy(cfg)
    ec = cfg["evaluation"]
    seed_all(ec["seed"])
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    device = device_for(ec["device"])
    encoder, image_size, metadata = load_reference(cfg, device)
    data = {s: FourViewDataset(cfg["dataset"]["output_path"], s, image_size)
            for s in ("train", "validation", "test")}
    manifest_hash = file_hash(data["train"].root / "manifest.json")
    if metadata.get("training_manifest_sha256", manifest_hash) != manifest_hash:
        raise ValueError("Probe dataset differs from the encoder training manifest")
    for ds in data.values():
        cap = ec["max_states_per_split"]
        if cap and len(ds.rows) > cap:
            ids = np.random.RandomState(ec["seed"]).choice(len(ds.rows), cap, replace=False)
            ds.rows = [ds.rows[i] for i in sorted(ids)]
    out = unique_output(ec["output_dir"], "probe_" + metadata["training_objective"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Evaluation output is not empty")
    default_name = f"trackA-probe-{metadata['training_objective']}-seed{ec['seed']}"
    if metadata["feature"] == "cls":
        default_name += "-cls"
    cfg["logging"]["name"] = cfg["logging"]["name"] or default_name
    metadata.update({"manifest_sha256": manifest_hash, "config": cfg, **git_info(),
                     "evaluated_samples": {s: [(r["trajectory_id"], r["timestep"]) for r in ds.rows] for s, ds in data.items()}})
    def features(images):
        return extract_features(encoder, images, device, ec["batch_size"], metadata["feature"])
    with RunLogger(out, cfg["logging"], metadata) as logger:
        images = {s: read_images(ds, "clean") for s, ds in data.items()}
        clean = {s: features(ims) for s, ims in images.items()}
        targets = {s: physical_targets(ds.rows) for s, ds in data.items()}
        probe = fit_probe(clean["train"], targets["train"], clean["validation"], targets["validation"], ec["ridge_alphas"])
        probe.save(out / "physical_probe.npz")
        rows, test = [], data["test"]
        baseline = {}

        def report(name, x, kind=None, strength=None):
            for seg in ("full", "early", "middle", "late"):
                ids = np.arange(len(test)) if seg == "full" else np.array([i for i, r in enumerate(test.rows) if r["segment"] == seg], dtype=int)
                if not len(ids):
                    continue
                metrics = state_metrics(probe, x[ids], targets["test"][ids])
                if name == "default":
                    baseline[seg] = metrics["standardized_mse"]
                rows.append({"condition": name, "segment": seg, "samples": len(ids),
                             "corruption_type": kind, "strength": strength,
                             "latent_distance": float(np.sqrt(((x[ids] - clean["test"][ids]) ** 2).mean())),
                             "degradation": metrics["standardized_mse"] - baseline[seg], **metrics})
                if seg == "full":
                    logger.log({f"probe/{name}/{k}": v for k, v in metrics.items() if v is not None})

        report("default", clean["test"])
        nuisance_x, nuisance_y = {}, {}
        # RGB regressors need variable-color A/B images; clean RGB alone is constant.
        for s, ds in data.items():
            x_a, x_b = features(read_images(ds, "A")), features(read_images(ds, "B"))
            nuisance_x[s] = np.concatenate([x_a, x_b])
            nuisance_y[s] = np.concatenate([np.asarray([r["eta_" + q] for r in ds.rows]).reshape(len(ds), 9)/255
                                            for q in ("A", "B")])
            if s == "test":
                report("colorA", x_a)
                report("colorB", x_b)
        diagnostics = []
        for j, obj in enumerate(("block", "agent", "goal")):
            ys = {s: y[:, j*3:(j+1)*3] for s, y in nuisance_y.items()}
            rgb_probe = fit_probe(nuisance_x["train"], ys["train"], nuisance_x["validation"], ys["validation"], ec["ridge_alphas"])
            rgb_probe.save(out / f"rgb_{obj}_probe.npz")
            diagnostics.append({"diagnostic": "rgb_" + obj, **regression_metrics(rgb_probe.predict(nuisance_x["test"]), ys["test"])})
        # New render conditions use source_state with the original render seed, and
        # verify against stored canonical state. Never reset from canonical again.
        color_cfg = copy.deepcopy(data["train"].manifest["config"]["dataset"]["color_sampling"])
        rng = np.random.RandomState(ec["seed"])
        env = new_env(image_size)
        eval_meta = []
        try:
            for condition in ("redBlock", "redAgent", "redAnchor", "random_inside", "random_outside"):
                colors = []
                for row in test.rows:
                    if condition.startswith("red"):
                        rgb = CLEAN_RGB.copy()
                        rgb[("redBlock", "redAgent", "redAnchor").index(condition)] = [255, 0, 0]
                    else:
                        cc = copy.deepcopy(color_cfg)
                        if condition == "random_outside":
                            cc.update(ec["outside_color_sampling"])
                        rgb, _ = sample_colors(rng, [CLEAN_RGB], cc)
                    colors.append(rgb)
                    eval_meta.append({"condition": condition, "sample_id": row["sample_id"], "rgb": rgb.tolist()})
                def get_shifted(index):
                    row = test.rows[index]
                    image, state = render_state(env, row["source_state"], colors[index], row["render_seed"])
                    ok, error = check_state(state, row["canonical_render_state"], data["train"].manifest["config"]["dataset"]["reset_tolerances"]["canonical"])
                    if not ok:
                        raise ValueError(f"Evaluation renderer state changed: {error}")
                    return image
                report(condition, features(ImageSequence(len(test), get_shifted)))
        finally:
            env.close()
        for kind, grid in ec["corruption_grid"].items():
            for strength in grid:
                ims = ImageSequence(len(test), lambda i: corrupt_frames(images["test"][i], kind, strength, ec["seed"] + i))
                report(f"{kind}_{strength:g}", features(ims), kind, float(strength))
        # Metadata-only diagnostics from the one fixed offline corruption per state.
        corr = {s: features(read_images(ds, "corrupt")) for s, ds in data.items()}
        kinds = data["train"].manifest["config"]["dataset"]["corruption_types"]
        labels = {s: np.array([kinds.index(r["corruption_type"]) for r in ds.rows]) for s, ds in data.items()}
        classifier = fit_probe(corr["train"], np.eye(len(kinds))[labels["train"]], corr["validation"],
                               np.eye(len(kinds))[labels["validation"]], ec["ridge_alphas"])
        predicted = classifier.predict(corr["test"]).argmax(-1)
        counts = np.bincount(labels["train"], minlength=len(kinds))
        diagnostics.append({"diagnostic": "corruption_type", "accuracy": float((predicted == labels["test"]).mean()),
                            "uniform_chance": 1 / len(kinds), "majority_baseline": float((labels["test"] == counts.argmax()).mean())})
        for j, kind in enumerate(kinds):
            ids = {s: np.where(labels[s] == j)[0] for s in data}
            if any(len(ids[s]) < (2 if s == "train" else 1) for s in data):
                diagnostics.append({"diagnostic": "strength_" + kind, "status": "insufficient_samples"})
                continue
            ys = {s: np.array([[data[s].rows[i]["corruption_strength"]] for i in ids[s]]) for s in data}
            p = fit_probe(corr["train"][ids["train"]], ys["train"], corr["validation"][ids["validation"]], ys["validation"], ec["ridge_alphas"])
            diagnostics.append({"diagnostic": "strength_" + kind, **regression_metrics(p.predict(corr["test"][ids["test"]]), ys["test"])})
        logger.table("results", rows)
        logger.table("diagnostics", diagnostics)
        write_json(out / "evaluation_views.json", eval_meta)
        with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_results(rows, out / "robustness.png")
        logger.robustness_plots(rows, out / "robustness.png")
    print(f"Probe results saved: {out}", flush=True)
    return out
