"""Small checks of the existing one-step AdaJEPA gradient path."""

import pytest
import torch
from torch import nn

from models.visual_world_model import VWorldModel
from models.vit import generate_mask_matrix
from datasets import pusht_dset
from train import Trainer


class VisualEncoder(nn.Module):
    name = "tiny_visual"
    emb_dim = 4
    latent_ndim = 1

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))

    def forward(self, images):
        value = images.flatten(1).mean(1, keepdim=True) * self.scale
        return value.repeat(1, self.emb_dim).unsqueeze(1)


class SmallEmbedding(nn.Module):
    emb_dim = 2

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


@pytest.mark.parametrize("mode", ["predictor_only", "joint"])
def test_predictive_loss_gradient_routing_and_future_stop_grad(mode):
    torch.manual_seed(7)
    encoder = VisualEncoder()
    if mode == "predictor_only":
        encoder.requires_grad_(False)
    predictor = nn.Linear(8, 8)
    model = VWorldModel(
        image_size=1, num_hist=3, num_pred=1, encoder=encoder,
        proprio_encoder=SmallEmbedding(), action_encoder=SmallEmbedding(),
        decoder=None, predictor=predictor, proprio_dim=2, action_dim=2,
        concat_dim=1, num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=(mode == "joint"), train_predictor=True,
        train_decoder=False, stop_grad=True, encoder_eval_mode=True,
    )
    model.train()
    assert not model.encoder.training
    assert model.predictor.training
    visual = torch.randn(2, 4, 1, 1, 1, requires_grad=True)
    actions = torch.randn(2, 4, 2, requires_grad=True)
    obs = {"visual": visual, "proprio": torch.randn(2, 4, 2)}
    _, _, _, loss, metrics = model(obs, actions)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["z_loss"].requires_grad
    assert predictor.weight.grad is not None
    assert predictor.weight.grad.norm() > 0
    if mode == "joint":
        assert encoder.scale.grad is not None
        assert encoder.scale.grad.abs() > 0
    else:
        assert encoder.scale.grad is None
    # Frame 3 is a target only. Its visual feature and action cannot receive
    # gradients through the stop-grad target branch.
    assert torch.count_nonzero(visual.grad[:, 3]) == 0
    assert torch.count_nonzero(actions.grad[:, 3]) == 0
    assert torch.count_nonzero(visual.grad[:, :3]) > 0


def test_heldout_validation_color_uses_disjoint_video_domain(monkeypatch):
    def fake_init(self, **kwargs):
        self.data_path = __import__("pathlib").Path(kwargs["data_path"])
        self.seq_lengths = [20]
        self.proprio_dim = self.action_dim = self.state_dim = 2

    monkeypatch.setattr(pusht_dset.PushTDataset, "__init__", fake_init)
    datasets, trajectories = pusht_dset.load_pusht_slice_train_val(
        transform=None, data_path="/unused/pusht_noise", num_hist=3,
        num_pred=1, frameskip=5,
        color_domains=["default", "blue", "green"],
        validation_color_domain="purpleBlock",
    )
    assert isinstance(trajectories["train"], pusht_dset.ColorPushTDataset)
    assert isinstance(trajectories["valid"], pusht_dset.FixedColorPushTDataset)
    assert trajectories["valid"].color_domain == "purpleBlock"
    assert trajectories["train"].data_path.name == "train"
    assert trajectories["valid"].data_path.name == "val"
    assert len(datasets["train"]) == len(datasets["valid"]) == 1


def test_predictor_attention_is_causal_over_three_frame_slots():
    mask = generate_mask_matrix(npatch=1, nwindow=3)[0, 0]
    assert torch.equal(mask, torch.tril(torch.ones(3, 3)))


def test_gradient_resume_uses_total_epoch_budget():
    assert list(range(1, Trainer._last_epoch_exclusive(0, 3, "joint"))) == [1, 2, 3]
    assert list(range(3, Trainer._last_epoch_exclusive(2, 3, "joint"))) == [3]
    assert list(range(4, Trainer._last_epoch_exclusive(3, 3, "joint"))) == []
    assert list(range(3, Trainer._last_epoch_exclusive(2, 3, None))) == [3, 4, 5]
