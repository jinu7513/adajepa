"""Fast tests for the frozen Track A patch-token world-model path."""

import pickle
import torch
from torch import nn
from omegaconf import OmegaConf
import pytest

from models.tracka_patch import TrackAPatchEncoder
from models.visual_world_model import VWorldModel
from models.vit import ViTPredictor
from tracka.model import ScratchEncoder


def make_tracka_checkpoint(tmp_path, step=10000):
    model_cfg = {"image_size": 32, "patch_size": 8, "embed_dim": 24,
                 "depth": 1, "heads": 3, "use_cls": True}
    encoder = ScratchEncoder(**model_cfg).eval()
    path = tmp_path / "tracka.pt"
    torch.save({"encoder": encoder.state_dict(), "config": {"model": model_cfg},
                "global_step": step, "manifest_sha256": "test-manifest"}, path)
    return path, encoder


def test_tracka_patch_adapter_matches_original(tmp_path):
    path, original = make_tracka_checkpoint(tmp_path)
    adapter = TrackAPatchEncoder(str(path), expected_step=10000).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = original(images)
        actual = adapter(images)
    assert actual.shape == (2, 16, 24)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(adapter.agg(actual), expected.mean(1))
    assert adapter.source_step == 10000
    assert adapter.source_manifest_sha256 == "test-manifest"


def test_tracka_patch_adapter_rejects_wrong_artifact(tmp_path):
    path, _ = make_tracka_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        TrackAPatchEncoder("relative/checkpoint.pt")
    with pytest.raises(ValueError, match="Expected Track A step"):
        TrackAPatchEncoder(str(path), expected_step=1000)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        TrackAPatchEncoder(str(path), expected_sha256="wrong")


class TinyEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.linear(x)


def test_world_model_receives_full_tracka_patch_grid(tmp_path):
    path, _ = make_tracka_checkpoint(tmp_path)
    adapter = TrackAPatchEncoder(str(path), expected_step=10000).eval()
    adapter.requires_grad_(False)
    model = VWorldModel(
        image_size=32, num_hist=2, num_pred=1, encoder=adapter,
        proprio_encoder=TinyEmbedding(4, 8), action_encoder=TinyEmbedding(2, 8),
        decoder=None, predictor=ViTPredictor(num_patches=16, num_frames=2, dim=40,
                                            depth=1, heads=2, mlp_dim=64),
        proprio_dim=8, action_dim=8,
        concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=False, train_predictor=True, train_decoder=False,
    )
    obs = {"visual": torch.randn(2, 3, 3, 32, 32),
           "proprio": torch.randn(2, 3, 4)}
    actions = torch.randn(2, 3, 2)
    visual = model.encode_obs(obs)["visual"]
    full = model.encode(obs, actions)
    assert visual.shape == (2, 3, 16, 24)
    assert full.shape == (2, 3, 16, 40)
    predicted, decoded, reconstructed, loss, parts = model(obs, actions)
    assert predicted.shape == (2, 2, 16, 40)
    assert decoded is None and reconstructed is None
    assert torch.isfinite(loss)
    assert torch.isfinite(parts["z_visual_loss"])
    loss.backward()
    assert any(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in adapter.parameters())


def test_tracka_preflight_requires_full_push_t_trajectories(tmp_path):
    from train import validate_tracka_predictor_inputs

    encoder_path, _ = make_tracka_checkpoint(tmp_path)
    root = tmp_path / "pusht_noise"
    cfg = OmegaConf.create({
        "encoder": {"_target_": "models.tracka_patch.TrackAPatchEncoder",
                    "checkpoint_path": str(encoder_path)},
        "env": {"name": "pusht", "dataset": {"data_path": str(root), "n_rollout": 2}},
        "model": {"train_encoder": False, "train_predictor": True, "train_decoder": False},
        "has_decoder": False,
        "training": {"save_frozen_encoder": True},
    })
    with pytest.raises(FileNotFoundError, match="rel_actions.pth"):
        validate_tracka_predictor_inputs(cfg)
    for split in ("train", "val"):
        for name in ("states.pth", "rel_actions.pth", "seq_lengths.pkl",
                     "velocities.pth", "obses/episode_000.mp4", "obses/episode_001.mp4"):
            path = root / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "seq_lengths.pkl":
                with path.open("wb") as stream:
                    pickle.dump([10, 10], stream)
            else:
                path.touch()
    validate_tracka_predictor_inputs(cfg)
    (root / "val" / "obses" / "episode_001.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="episode_001.mp4"):
        validate_tracka_predictor_inputs(cfg)


def test_plan_loader_accepts_self_contained_tracka_checkpoint(tmp_path):
    from plan import load_ckpt, load_model

    path, _ = make_tracka_checkpoint(tmp_path)
    encoder = TrackAPatchEncoder(str(path), expected_step=10000)
    world_model_path = tmp_path / "model_latest.pth"
    torch.save({"encoder": encoder,
                "predictor": ViTPredictor(num_patches=16, num_frames=2, dim=40,
                                          depth=1, heads=2, mlp_dim=64),
                "proprio_encoder": TinyEmbedding(4, 8),
                "action_encoder": TinyEmbedding(2, 8),
                "epoch": 1}, world_model_path)
    loaded = load_ckpt(world_model_path, torch.device("cpu"),
                       encoder_target="models.tracka_patch.TrackAPatchEncoder")
    assert loaded["epoch"] == 1
    assert loaded["encoder"].source_step == 10000
    assert loaded["predictor"] is not None
    train_cfg = OmegaConf.create({
        "encoder": {"_target_": "models.tracka_patch.TrackAPatchEncoder"},
        "model": {"_target_": "models.visual_world_model.VWorldModel",
                  "image_size": 32, "num_hist": 2, "num_pred": 1,
                  "train_encoder": False, "train_predictor": True,
                  "train_decoder": False},
        "has_decoder": False, "proprio_emb_dim": 8, "action_emb_dim": 8,
        "concat_dim": 1, "num_proprio_repeat": 1,
    })
    model = load_model(world_model_path, train_cfg, num_action_repeat=1,
                       device=torch.device("cpu"))
    obs = {"visual": torch.randn(1, 2, 3, 32, 32),
           "proprio": torch.randn(1, 2, 4)}
    assert model.encode_obs(obs)["visual"].shape == (1, 2, 16, 24)
