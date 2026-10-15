"""CPU checks for frozen Event patch tokens in the PushT predictor."""

import pickle

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from models.tracka_event_patch import TrackAEventPatchEncoder
from models.visual_world_model import VWorldModel
from models.vit import ViTPredictor
from tracka.event_model import EventTeacherStudent


MODEL_CFG = {
    "image_size": 32, "patch_size": 8, "embed_dim": 24, "heads": 3,
    "rgb_depth": 0, "event_depth": 0, "shared_depth": 1,
    "event_mode": "hard_two_channel", "event_threshold": .2,
    "event_eps": .001, "shift_previous": False, "fusion": "fixed_sum",
    "global_representation": "cls", "rgb_input_mode": "current",
}
LOSS_CFG = {"distance": "cosine", "anti_collapse": "none",
            "lambda_rob": 1., "lambda_temp": 1., "lambda_ac": 0.}


def make_event_checkpoint(tmp_path, step=10000, patch_size=8):
    model_cfg = dict(MODEL_CFG, patch_size=patch_size)
    model = EventTeacherStudent(model_cfg, LOSS_CFG).eval()
    path = tmp_path / "event.pt"
    torch.save({"schema_version": "tracka-event-v1", "model": model.state_dict(),
                "config": {"model": model_cfg, "loss": LOSS_CFG},
                "global_step": step, "manifest_sha256": "test-manifest"}, path)
    return path, model


class TinyEmbedding(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, values):
        return self.linear(values)


def test_event_adapter_uses_exact_student_patch_tokens(tmp_path):
    path, original = make_event_checkpoint(tmp_path)
    encoder = TrackAEventPatchEncoder(str(path), expected_step=10000).eval()
    previous = torch.randn(2, 3, 3, 32, 32)
    current = torch.randn(2, 3, 3, 32, 32)
    with torch.no_grad():
        expected = original.student(previous.flatten(0, 1), current.flatten(0, 1))["patch_tokens"]
        actual = encoder.forward_sequence(previous, current)
    assert actual.shape == (2, 3, 16, 24)
    torch.testing.assert_close(actual.flatten(0, 1), expected)
    assert encoder.source_step == 10000
    assert encoder.source_manifest_sha256 == "test-manifest"
    embedded = tmp_path / "embedded_model.pt"
    torch.save({"encoder": encoder}, embedded)
    path.rename(tmp_path / "moved_event.pt")
    restored = torch.load(embedded, map_location="cpu", weights_only=False)["encoder"].eval()
    with torch.no_grad():
        torch.testing.assert_close(restored.forward_sequence(previous, current), actual)
    with pytest.raises(ValueError, match="absolute"):
        TrackAEventPatchEncoder("relative/event.pt")
    with pytest.raises(ValueError, match="Expected Event step"):
        TrackAEventPatchEncoder(str(tmp_path / "moved_event.pt"), expected_step=5000)


def test_event_world_model_predictor_backward_and_missing_previous(tmp_path):
    path, _ = make_event_checkpoint(tmp_path)
    encoder = TrackAEventPatchEncoder(str(path)).eval().requires_grad_(False)
    world = VWorldModel(
        image_size=32, num_hist=2, num_pred=1, encoder=encoder,
        proprio_encoder=TinyEmbedding(4, 8), action_encoder=TinyEmbedding(2, 8),
        decoder=None, predictor=ViTPredictor(num_patches=16, num_frames=2, dim=40,
                                            depth=1, heads=2, mlp_dim=64),
        proprio_dim=8, action_dim=8, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=False, train_predictor=True, train_decoder=False,
    )
    obs = {"visual": torch.randn(2, 3, 3, 32, 32),
           "visual_previous": torch.randn(2, 3, 3, 32, 32),
           "proprio": torch.randn(2, 3, 4)}
    act = torch.randn(2, 3, 2)
    with pytest.raises(ValueError, match="visual_previous"):
        world.encode_obs({key: value for key, value in obs.items() if key != "visual_previous"})
    assert world.encode_obs(obs)["visual"].shape == (2, 3, 16, 24)
    predicted, decoded, reconstructed, loss, parts = world(obs, act)
    assert predicted.shape == (2, 2, 16, 40)
    assert decoded is None and reconstructed is None
    assert torch.isfinite(loss) and torch.isfinite(parts["z_visual_loss"])
    loss.backward()
    assert any(param.grad is not None for param in world.predictor.parameters())
    assert all(param.grad is None for param in encoder.parameters())


@pytest.mark.parametrize("scope", ["full", "final_shared"])
def test_event_joint_training_reaches_selected_encoder_and_predictor(tmp_path, scope):
    from train import Trainer

    path, _ = make_event_checkpoint(tmp_path)
    encoder = TrackAEventPatchEncoder(str(path))
    trainer = Trainer.__new__(Trainer)
    trainer.encoder = encoder
    trainer.train_encoder = True
    trainer.cfg = OmegaConf.create({"training": {"encoder_train_scope": scope}})
    trainer._configure_encoder_trainability()
    trainable = {name for name, param in encoder.named_parameters() if param.requires_grad}
    assert trainable
    if scope == "full":
        assert trainable == {name for name, _ in encoder.named_parameters()}
    else:
        assert all(name.startswith(("backbone.shared_blocks.0.", "backbone.final_norm."))
                   for name in trainable)
        assert any(name.startswith("backbone.shared_blocks.0.") for name in trainable)

    world = VWorldModel(
        image_size=32, num_hist=2, num_pred=1, encoder=encoder,
        proprio_encoder=TinyEmbedding(4, 8), action_encoder=TinyEmbedding(2, 8),
        decoder=None, predictor=ViTPredictor(num_patches=16, num_frames=2, dim=40,
                                            depth=1, heads=2, mlp_dim=64),
        proprio_dim=8, action_dim=8, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=True, train_predictor=True, train_decoder=False, stop_grad=True,
    )
    obs = {"visual": torch.randn(2, 3, 3, 32, 32),
           "visual_previous": torch.randn(2, 3, 3, 32, 32),
           "proprio": torch.randn(2, 3, 4)}
    world(obs, torch.randn(2, 3, 2))[3].backward()
    assert any(param.grad is not None for name, param in encoder.named_parameters()
               if name in trainable)
    assert all(param.grad is None for name, param in encoder.named_parameters()
               if name not in trainable)
    assert any(param.grad is not None for param in world.predictor.parameters())


def test_event_video_reads_true_previous_frame(monkeypatch, tmp_path):
    from datasets.pusht_dset import PushTDataset
    import datasets.pusht_dset as pusht

    class FakeReader:
        def __init__(self, *args, **kwargs):
            pass

        def get_batch(self, frames):
            return torch.stack([torch.full((4, 4, 3), int(frame), dtype=torch.uint8)
                                for frame in frames])

    monkeypatch.setattr(pusht, "VideoReader", FakeReader)
    data = PushTDataset.__new__(PushTDataset)
    data.data_path = tmp_path
    data.actions = torch.zeros(1, 12, 2)
    data.states = torch.zeros(1, 12, 7)
    data.proprios = torch.zeros(1, 12, 4)
    data.shapes = ["T"]
    data.transform = None
    data.include_previous_frames = True
    obs, _, _, _ = data.get_frames(0, [0, 5, 10])
    torch.testing.assert_close(obs["visual"][:, 0, 0, 0], torch.tensor([0., 5., 10.]) / 255)
    torch.testing.assert_close(obs["visual_previous"][:, 0, 0, 0],
                               torch.tensor([0., 4., 9.]) / 255)


def test_event_slicer_keeps_actions_dense_but_images_sparse():
    from datasets.traj_dset import TrajSlicerDataset

    class TinyTrajectory:
        include_previous_frames = True
        proprio_dim = 4
        action_dim = 2
        state_dim = 7

        def __init__(self):
            self.actions = torch.arange(12 * 2, dtype=torch.float32).reshape(1, 12, 2)
            self.requested = None

        def __len__(self):
            return 1

        def get_seq_length(self, index):
            return 12

        def get_frames(self, index, frames):
            self.requested = list(frames)
            values = torch.tensor(self.requested, dtype=torch.float32)
            obs = {"visual": values[:, None],
                   "visual_previous": (values - 1).clamp_min(0)[:, None],
                   "proprio": torch.zeros(len(values), 4)}
            return obs, self.actions[index, self.requested], torch.zeros(len(values), 7), {}

    source = TinyTrajectory()
    sliced = TrajSlicerDataset(source, num_frames=3, frameskip=2)
    _, start, end = sliced.slices[0]
    obs, act, _ = sliced[0]
    assert source.requested == list(range(start, end, 2))
    assert act.shape == (3, 4)
    torch.testing.assert_close(act[0], source.actions[0, start:start + 2].flatten())
    torch.testing.assert_close(obs["visual_previous"][:, 0],
                               (obs["visual"][:, 0] - 1).clamp_min(0))


def test_event_preflight_requires_previous_and_full_trajectories(tmp_path):
    from train import validate_tracka_predictor_inputs

    path, _ = make_event_checkpoint(tmp_path, patch_size=16)
    root = tmp_path / "pusht_noise"
    cfg = OmegaConf.create({
        "encoder": {"_target_": "models.tracka_event_patch.TrackAEventPatchEncoder",
                    "checkpoint_path": str(path), "expected_step": 10000},
        "env": {"name": "pusht", "dataset": {"data_path": str(root),
                                          "n_rollout": 1, "include_previous_frames": False}},
        "model": {"train_encoder": False, "train_predictor": True, "train_decoder": False},
        "has_decoder": False, "img_size": 32,
        "training": {"save_frozen_encoder": True},
    })
    with pytest.raises(ValueError, match="include_previous_frames"):
        validate_tracka_predictor_inputs(cfg)
    cfg.env.dataset.include_previous_frames = True
    with pytest.raises(FileNotFoundError, match="rel_actions.pth"):
        validate_tracka_predictor_inputs(cfg)
    for split in ("train", "val"):
        for name in ("states.pth", "rel_actions.pth", "seq_lengths.pkl",
                     "velocities.pth", "obses/episode_000.mp4"):
            destination = root / split / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if name == "seq_lengths.pkl":
                with destination.open("wb") as stream:
                    pickle.dump([10], stream)
            else:
                destination.touch()
    validate_tracka_predictor_inputs(cfg)
    cfg.model.train_encoder = True
    cfg.training.encoder_train_scope = "full"
    validate_tracka_predictor_inputs(cfg)
    cfg.training.encoder_train_scope = "final_shared"
    validate_tracka_predictor_inputs(cfg)
    cfg.model.train_encoder = False
    with pytest.raises(ValueError, match="encoder_train_scope=frozen"):
        validate_tracka_predictor_inputs(cfg)


def test_event_resume_forks_when_metrics_are_ahead(tmp_path):
    from tracka.event_train import resume_destination

    old = tmp_path / "original"
    old.mkdir()
    metrics = old / "metrics.jsonl"
    metrics.write_text('{"train/global_step": 5000}\n{"train/global_step": 5060}\n')
    checkpoint = {"output_dir": str(old), "global_step": 5000,
                  "wandb_run_id": "original-wandb-id"}
    training = {"resume": str(old / "checkpoint_latest.pt"),
                "output_dir": None, "fork_from_checkpoint": False}
    with pytest.raises(ValueError, match="older than durable metrics"):
        resume_destination(checkpoint, training)
    fresh = tmp_path / "new-run"
    training.update(output_dir=str(fresh), fork_from_checkpoint=True)
    destination, run_id, source = resume_destination(checkpoint, training)
    assert destination == fresh
    assert run_id is None  # A fork must get a fresh W&B identity.
    assert source == str(old / "checkpoint_latest.pt")
    assert metrics.read_text().endswith('{"train/global_step": 5060}\n')
    fresh.mkdir()
    (fresh / "existing.txt").touch()
    with pytest.raises(FileExistsError, match="new and empty"):
        resume_destination(checkpoint, training)
