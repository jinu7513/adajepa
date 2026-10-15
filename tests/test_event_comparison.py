from omegaconf import OmegaConf

from scripts.check_event_comparison import audit


def test_audit_finds_training_mismatch(tmp_path):
    runs = []
    for label, train_encoder, scope in (("A", False, "frozen"),
                                        ("B", True, "full"),
                                        ("C", True, "final_shared")):
        directory = tmp_path / label
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "checkpoints" / "model_3.pth").write_bytes(b"checkpoint")
        config = OmegaConf.create({
            "encoder": {"_target_": "models.tracka_event_patch.TrackAEventPatchEncoder",
                        "checkpoint_path": str(tmp_path / "event.pt")},
            "env": {"name": "pusht", "dataset": {
                "data_path": str(tmp_path / "pusht_noise"),
                "include_previous_frames": True, "n_rollout": None,
                "window_block_size": 2}},
            "frameskip": 5, "num_hist": 3, "num_pred": 1,
            "training": {"seed": 0, "batch_size": 32, "stop_grad": True,
                         "encoder_lr": 1e-5, "predictor_lr": 5e-4,
                         "action_encoder_lr": 5e-4, "mixed_precision": "bf16",
                         "encoder_train_scope": scope},
            "has_predictor": True, "has_decoder": False,
            "model": {"train_encoder": train_encoder, "train_predictor": True,
                      "train_decoder": False},
        })
        OmegaConf.save(config, directory / "hydra.yaml")
        runs.append(directory)
    assert audit(runs, 3) == []
    config = OmegaConf.load(runs[2] / "hydra.yaml")
    config.training.batch_size = 64
    OmegaConf.save(config, runs[2] / "hydra.yaml")
    assert "training.batch_size differs" in "\n".join(audit(runs, 3))
