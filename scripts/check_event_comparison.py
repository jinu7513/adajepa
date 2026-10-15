"""Read-only check that A/B/C predictor runs use comparable PushT inputs."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


MATCHED_FIELDS = (
    "encoder._target_", "encoder.checkpoint_path", "env.name",
    "env.dataset.data_path", "env.dataset.include_previous_frames",
    "env.dataset.n_rollout", "env.dataset.window_block_size",
    "frameskip", "num_hist", "num_pred",
    "training.seed", "training.batch_size", "training.stop_grad",
    "training.encoder_lr",
    "training.predictor_lr", "training.action_encoder_lr",
    "training.mixed_precision", "has_predictor", "has_decoder",
    "model.train_predictor", "model.train_decoder",
)


def audit(directories, epoch):
    configs = {}
    problems = []
    for label, directory in zip("ABC", directories):
        folder = Path(directory).expanduser().resolve()
        config_path = folder / "hydra.yaml"
        checkpoint_path = folder / "checkpoints" / f"model_{epoch}.pth"
        if not config_path.is_file() or not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
            problems.append(f"{label}: missing or empty "
                            f"{config_path if not config_path.is_file() else checkpoint_path}")
            continue
        config = OmegaConf.load(config_path)
        configs[label] = config
        if OmegaConf.select(config, "encoder._target_") != "models.tracka_event_patch.TrackAEventPatchEncoder":
            problems.append(f"{label}: not an Event-patch predictor checkpoint")
        training = bool(OmegaConf.select(config, "model.train_encoder"))
        scope = OmegaConf.select(config, "training.encoder_train_scope")
        if label == "A" and training:
            problems.append("A: encoder should be frozen")
        if label == "B" and (not training or scope != "full"):
            problems.append("B: expected full joint encoder training")
        if label == "C" and (not training or scope != "final_shared"):
            problems.append("C: expected final_shared encoder training")

    if len(configs) == 3:
        for key in MATCHED_FIELDS:
            values = {label: OmegaConf.select(config, key) for label, config in configs.items()}
            if key in ("encoder.checkpoint_path", "env.dataset.data_path"):
                values = {label: str(Path(value).resolve()) if value else value
                          for label, value in values.items()}
            if len(set(map(str, values.values()))) != 1:
                problems.append(f"{key} differs: {values}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("A", help="existing frozen predictor run directory")
    parser.add_argument("B", help="full joint predictor run directory")
    parser.add_argument("C", help="final-shared predictor run directory")
    parser.add_argument("--epoch", type=int, default=3)
    args = parser.parse_args()
    problems = audit((args.A, args.B, args.C), args.epoch)
    if problems:
        for problem in problems:
            print("MISMATCH:", problem)
        raise SystemExit(1)
    print(f"A/B/C inputs match and each has model_{args.epoch}.pth;"
          " encoder trainability differs as intended.")


if __name__ == "__main__":
    main()
