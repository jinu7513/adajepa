"""Same-color one-step and autoregressive horizon errors on real PushT videos."""

import argparse
import csv
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from datasets.pusht_dset import FixedColorPushTDataset, PushTDataset
from datasets.img_transforms import default_transform
from plan import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--source-default", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.model_dir / "hydra.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_dir / "checkpoints" / "model_{}.pth".format(args.epoch),
                       cfg, cfg.num_action_repeat, device).eval()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for color in ("default", "purpleBlock", "redBlock"):
        kwargs = dict(n_rollout=None, transform=default_transform(cfg.img_size),
                      data_path=str(args.data_root / "val"), normalize_action=cfg.normalize_action,
                      with_velocity=True)
        if color == "default" and args.source_default:
            dataset = PushTDataset(**kwargs)
        else:
            dataset = FixedColorPushTDataset(color_domain=color, **kwargs)
        count = 0
        for episode in range(len(dataset)):
            if count >= args.episodes:
                break
            if dataset.get_seq_length(episode) < 40:
                continue
            obs, raw_actions, _, _ = dataset.get_frames(episode, range(40))
            obs = {key: value[::5].unsqueeze(0).to(device) for key, value in obs.items()}
            actions = raw_actions.reshape(1, 8, 10).to(device)
            with torch.no_grad():
                prediction, _, _, _, _ = model(
                    {key: value[:, :4] for key, value in obs.items()}, actions[:, :4]
                )
                target = model.encode_obs({key: value[:, :4] for key, value in obs.items()})["visual"]
                visual_dim = model.encoder.emb_dim
                one_step = torch.mean((prediction[..., :visual_dim] - target[:, 1:4]) ** 2)
                rolled, _ = model.rollout(
                    {key: value[:, :3] for key, value in obs.items()},
                    actions, known_observed_actions=True
                )
                targets = model.encode_obs(obs)["visual"]
                for horizon in range(1, 6):
                    index = 2 + horizon
                    raw = torch.mean((rolled["visual"][:, index] - targets[:, index]) ** 2)
                    scale = torch.mean(targets[:, index] ** 2).clamp_min(1e-8)
                    rows.append({"condition": color, "episode": episode,
                                 "horizon": horizon, "visual_mse": float(raw.item()),
                                 "visual_relative_mse": float((raw / scale).item()),
                                 "teacher_forced_one_step_visual_mse": float(one_step.item())})
            count += 1
    if not rows:
        raise ValueError("No valid evaluation windows")
    with (output / "per_episode_horizon.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in ("default", "purpleBlock", "redBlock"):
        summary[condition] = {}
        for horizon in range(1, 6):
            values = [r for r in rows if r["condition"] == condition and r["horizon"] == horizon]
            summary[condition][str(horizon)] = {
                "n": len(values),
                "visual_mse": sum(r["visual_mse"] for r in values) / len(values),
                "visual_relative_mse": sum(r["visual_relative_mse"] for r in values) / len(values),
                "teacher_forced_one_step_visual_mse": sum(r["teacher_forced_one_step_visual_mse"] for r in values) / len(values),
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved {}".format(output / "summary.json"))


if __name__ == "__main__":
    main()
