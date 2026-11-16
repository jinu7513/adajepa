"""Cross-color and inverse branches remain independent and correctly aligned."""

import multiprocessing
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from datasets.pusht_dset import ColorPairPushTDataset, PushTDataset
from models.inverse_action import InverseActionHead
from models.visual_world_model import VWorldModel
from torch import nn


@pytest.mark.parametrize("mode", ("same_fixed", "same_window_random", "cross_fixed", "cross_window_random"))
def test_window_pair_assignment_is_reproducible_and_respects_scheme(mode):
    def instance():
        obj = ColorPairPushTDataset.__new__(ColorPairPushTDataset)
        obj.color_domains = ("default", "blue", "green")
        obj.color_pair_mode = mode
        obj.color_seed = 11
        obj.epoch_shared = multiprocessing.Value("i", 0)
        return obj

    first, second = instance(), instance()
    first.set_epoch(1)
    second.set_epoch(1)
    pairs = [first.pair_for(e, s) for e in range(6) for s in range(10)]
    assert pairs == [second.pair_for(e, s) for e in range(6) for s in range(10)]
    assert all((a == b) == mode.startswith("same") for a, b in pairs)
    if mode.endswith("window_random"):
        first.set_epoch(2)
        assert pairs != [first.pair_for(e, s) for e in range(6) for s in range(10)]
    else:
        first.set_epoch(2)
        assert pairs == [first.pair_for(e, s) for e in range(6) for s in range(10)]
    if mode == "cross_fixed":
        assert len(set(pairs)) == 6


def test_paired_renders_share_actions_states_and_hold_each_color_within_window(tmp_path, monkeypatch):
    obj = ColorPairPushTDataset.__new__(ColorPairPushTDataset)
    obj.data_path = tmp_path
    obj.transform = None
    obj.color_domains = ("default", "blue", "green")
    obj.color_pair_mode = "cross_fixed"
    obj.color_seed = 0
    obj.epoch_shared = multiprocessing.Value("i", 1)
    input_color, target_color = obj.pair_for(0, 0)
    for color in (input_color, target_color):
        path = tmp_path / "colors" / color / "obses" / "episode_000.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def fake_base_get_frames(self, episode, frames):
        value = {"default": 0.1, "blue": 0.2, "green": 0.3}[input_color]
        return ({"visual": torch.full((20, 3, 2, 2), value),
                 "proprio": torch.zeros(20, 4)},
                torch.ones(20, 2), torch.arange(100).reshape(20, 5), {})

    class FakeReader:
        def __init__(self, path, num_threads):
            self.value = {"default": 25, "blue": 51, "green": 76}[Path(path).parts[-3]]

        def get_batch(self, frames):
            return torch.full((len(frames), 2, 2, 3), self.value, dtype=torch.uint8)

    monkeypatch.setattr(PushTDataset, "get_frames", fake_base_get_frames)
    monkeypatch.setattr("datasets.pusht_dset.VideoReader", FakeReader)
    obs, actions, states, _ = obj.get_frames(0, range(20))
    assert actions.shape == (20, 2) and torch.equal(actions, torch.ones_like(actions))
    assert torch.equal(states, torch.arange(100).reshape(20, 5))
    assert torch.equal(obs["proprio"], torch.zeros(20, 4))
    assert torch.allclose(obs["visual"], obs["visual"][0, 0, 0, 0].expand_as(obs["visual"]))
    assert torch.allclose(obs["visual_target"], obs["visual_target"][0, 0, 0, 0].expand_as(obs["visual_target"]))
    assert not torch.equal(obs["visual"], obs["visual_target"])


def test_same_color_pair_uses_identical_source_and_target_images(tmp_path, monkeypatch):
    obj = ColorPairPushTDataset.__new__(ColorPairPushTDataset)
    obj.data_path = tmp_path
    obj.transform = None
    obj.color_domains = ("default", "blue", "green")
    obj.color_pair_mode = "same_fixed"
    obj.color_seed = 0
    obj.epoch_shared = multiprocessing.Value("i", 0)

    def fake_base_get_frames(self, episode, frames):
        return ({"visual": torch.randn(20, 3, 2, 2),
                 "proprio": torch.randn(20, 4)},
                torch.randn(20, 2), torch.randn(20, 5), {})

    monkeypatch.setattr(PushTDataset, "get_frames", fake_base_get_frames)
    obs, _, _, _ = obj.get_frames(0, range(20))
    assert torch.equal(obs["visual"], obs["visual_target"])


def test_inverse_uses_actual_endpoint_and_all_intervening_action_chunks():
    torch.manual_seed(5)
    head = InverseActionHead(visual_dim=4, action_dim=2, horizons=(1, 2, 3), hidden_dim=8)
    visual = torch.randn(2, 4, 1, 4, requires_grad=True)
    actions = torch.randn(2, 4, 2)
    loss, horizons = head(visual, actions)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)
    before = head.heads["3"][-1].weight.detach().clone()
    optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(loss) and set(horizons) == {1, 2, 3}
    assert visual.grad[:, 0].abs().sum() > 0
    assert visual.grad[:, 3].abs().sum() > 0
    assert head.heads["3"][-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, head.heads["3"][-1].weight)
    assert actions.grad is None
    with pytest.raises(ValueError, match="endpoint"):
        head(visual[:, :3], actions)


def test_track_configs_never_combine_cross_color_and_inverse():
    root = Path(__file__).resolve().parents[1] / "conf"
    for name in ("same_fixed", "same_window_random", "cross_fixed", "cross_window_random"):
        cfg = OmegaConf.load(root / "experiment" / "cross_color_jepa" / (name + ".yaml"))
        assert cfg.training.research_track == "cross_color_jepa"
        assert cfg.training.inverse_weight == 0
        assert cfg.training.color_pair_mode == name
    for name in ("default_jepa_baseline", "default_jepa_plus_inverse"):
        cfg = OmegaConf.load(root / "experiment" / "mi_jepa" / (name + ".yaml"))
        assert cfg.training.research_track == "mi_jepa"
        assert "color_pair_mode" not in cfg.training
    env = OmegaConf.load(root / "env" / "pusht_mi.yaml")
    assert "color_domains" not in env.dataset and "color_pair_mode" not in env.dataset


def test_cross_color_target_is_stop_gradient_but_input_encoder_updates():
    class TinyVisual(nn.Module):
        emb_dim = 4
        latent_ndim = 1
        name = "tiny_visual"

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.7))

        def forward(self, images):
            value = images.flatten(1).mean(dim=1, keepdim=True) * self.scale
            return value.repeat(1, 4).unsqueeze(1)

    class TinyEmbedding(nn.Module):
        emb_dim = 2

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    encoder = TinyVisual()
    model = VWorldModel(
        image_size=1, num_hist=3, num_pred=1, encoder=encoder,
        proprio_encoder=TinyEmbedding(), action_encoder=TinyEmbedding(),
        decoder=None, predictor=nn.Linear(8, 8), proprio_dim=2, action_dim=2,
        concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=True, train_predictor=True, train_decoder=False,
        stop_grad=True, encoder_eval_mode=True,
    )
    input_visual = torch.randn(2, 4, 1, 1, 1, requires_grad=True)
    target_visual = torch.randn(2, 4, 1, 1, 1, requires_grad=True)
    obs = {"visual": input_visual, "visual_target": target_visual,
           "proprio": torch.randn(2, 4, 2)}
    actions = torch.randn(2, 4, 2)
    _, _, _, loss, _ = model(obs, actions)
    loss.backward()
    assert encoder.scale.grad is not None and encoder.scale.grad.abs() > 0
    assert input_visual.grad[:, :3].abs().sum() > 0
    assert target_visual.grad is None or target_visual.grad.abs().sum() == 0
    assert input_visual.grad[:, 3].abs().sum() == 0
