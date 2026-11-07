import numpy as np
import torch
from torch import nn

from planning.latent_adapter import LatentAdapterTrainer, VisualLatentAdapter


class _Encoder(nn.Module):
    emb_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


class _WorldModel(nn.Module):
    concat_dim = 1
    action_dim = 1
    num_hist = 1

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.predictor = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.predictor.weight.copy_(torch.eye(3) * 0.5)
        for param in self.parameters():
            param.requires_grad_(False)

    def encode(self, obs, act):
        visual = obs["visual"] * self.encoder.weight
        if getattr(self, "visual_adapter_enabled", True):
            visual = self.visual_adapter(visual)
        return torch.cat([visual.unsqueeze(2), act.unsqueeze(2)], dim=-1)

    def predict(self, z):
        return self.predictor(z)


def test_adapter_starts_as_identity_and_updates_without_changing_world_model():
    wm = _WorldModel()
    trainer = LatentAdapterTrainer(wm, "mlp", 3, 1e-2, 3)
    x = torch.randn(4, 2)
    torch.testing.assert_close(trainer.adapter(x), x)
    before = {k: v.detach().clone() for k, v in wm.predictor.state_dict().items()}
    initial = [p.detach().clone() for p in trainer.adapter.parameters()]
    obs = {"visual": torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])}
    act = torch.ones(1, 2, 1)
    losses = trainer.finetune([obs], [act])
    assert len(losses) == 3 and all(np.isfinite(losses))
    assert any(not torch.equal(p, q) for p, q in zip(trainer.adapter.parameters(), initial))
    for key, value in before.items():
        torch.testing.assert_close(wm.predictor.state_dict()[key], value)
    trainer.reset()
    for param, value in zip(trainer.adapter.parameters(), initial):
        torch.testing.assert_close(param, value)


def test_bias_adapter_and_score_are_finite():
    wm = _WorldModel()
    trainer = LatentAdapterTrainer(wm, "bias", 3, 1e-2, 1)
    obs = {"visual": torch.tensor([[[1., 2.], [2., 3.]]])}
    act = torch.ones(1, 1, 1)
    assert len(trainer.finetune([obs], [act])) == 1
    assert np.isfinite(trainer.score_segments([obs], [act])[0])


def test_visual_adapter_works_on_patch_axis():
    adapter = VisualLatentAdapter(384, "mlp", 32)
    z = torch.randn(2, 3, 4, 384)
    torch.testing.assert_close(adapter(z), z)
