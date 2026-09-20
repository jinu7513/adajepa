import pytest
import torch
import torch.nn as nn

from planning.adajepa import AdaJEPATrainer
from models.encoder.resnet import SmallResNet


class ToyEncoder(nn.Module):
    """Small ordered encoder matching the top-level block contract used by AdaJEPA."""

    def __init__(self, dim=4):
        super().__init__()
        self.rb1 = nn.Linear(dim, dim)
        self.rb2 = nn.Linear(dim, dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        return self.projection(self.rb2(self.rb1(x)))


class ToyTransformer(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.norm(x)


class ToyPredictor(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.transformer = ToyTransformer(dim)

    def forward(self, x):
        return self.transformer(x)


class ToyWorldModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.encoder = ToyEncoder(dim)
        self.predictor = ToyPredictor(dim)
        self.num_hist = 2
        self.concat_dim = 0
        self.action_dim = dim
        self.stop_grad = True

    def encode(self, obs, act):
        visual = self.encoder(obs["visual"]).unsqueeze(2)
        proprio = obs["proprio"].unsqueeze(2)
        action = act.unsqueeze(2)
        return torch.cat([visual, proprio, action], dim=2)

    def predict(self, z):
        return self.predictor(z)


def make_segment(transitions=7, dim=4):
    generator = torch.Generator().manual_seed(7)
    obs = {
        "visual": torch.randn(2, transitions + 1, dim, generator=generator),
        "proprio": torch.randn(2, transitions + 1, dim, generator=generator),
    }
    actions = torch.randn(2, transitions, dim, generator=generator)
    return obs, actions


def make_trainer(mode="last_block", multi_step=False, horizons=None, weights=None):
    return AdaJEPATrainer(
        wm=ToyWorldModel(),
        lr=1e-3,
        steps=1,
        encoder_lr=1e-3,
        encoder_adapt_mode=mode,
        multi_step_adaptation=multi_step,
        adapt_horizons=horizons,
        horizon_weights=weights,
    )


def test_released_small_resnet_parameter_block_order():
    encoder = SmallResNet(dim=4)
    block_names = [
        name
        for name, module in encoder.named_children()
        if any(True for _ in module.parameters())
    ]
    assert block_names == ["rb1", "rb2", "rb3", "rb4", "rb5", "projection"]


@pytest.mark.parametrize(
    "mode,expected_blocks",
    [
        ("freeze_all", set()),
        ("last_block", {"projection"}),
        ("last_2_blocks", {"rb2", "projection"}),
        ("full_encoder", {"rb1", "rb2", "projection"}),
    ],
)
def test_encoder_modes_have_only_expected_gradients(mode, expected_blocks):
    trainer = make_trainer(mode=mode)
    obs, actions = make_segment(transitions=3)
    trainer.finetune([obs], [actions])

    for name, module in trainer.wm.encoder.named_children():
        gradients = [parameter.grad for parameter in module.parameters()]
        if name in expected_blocks:
            assert gradients and all(gradient is not None for gradient in gradients)
        else:
            assert all(gradient is None for gradient in gradients)
    assert all(not parameter.requires_grad for parameter in trainer.wm.encoder.parameters())

    selected_predictor_ids = {id(p) for p in trainer._ada_predictor_params}
    for parameter in trainer.wm.predictor.parameters():
        if id(parameter) in selected_predictor_ids:
            assert parameter.grad is not None
        else:
            assert parameter.grad is None
        assert not parameter.requires_grad


def test_one_step_disabled_uses_original_loss_exactly():
    trainer = make_trainer(
        mode="last_block", multi_step=False, horizons=[1, 2, 4], weights=[1, 0.5, 0.25]
    )
    obs, actions = make_segment(transitions=4)
    prepared_obs, prepared_actions = trainer._prepare_segment(obs, actions)
    expected = trainer._prediction_loss(
        trainer.wm.encode(prepared_obs, prepared_actions),
        detach_src=False,
        detach_tgt=True,
    ).item()
    actual = trainer.finetune([obs], [actions])[0]
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_multi_step_loss_contains_every_requested_horizon():
    trainer = make_trainer(
        multi_step=True, horizons=[1, 2, 4], weights=[1.0, 0.5, 0.25]
    )
    obs, actions = make_segment(transitions=7)
    prepared_obs, prepared_actions = trainer._prepare_segment(obs, actions)
    z = trainer.wm.encode(prepared_obs, prepared_actions)
    total, components = trainer._multi_step_prediction_loss(
        z, detach_src=False, detach_tgt=True
    )
    assert set(components) == {1, 2, 4}
    expected = components[1] + 0.5 * components[2] + 0.25 * components[4]
    assert torch.allclose(total, expected)
    assert all(component.item() > 0 for component in components.values())


def test_horizon_weight_length_must_match():
    with pytest.raises(ValueError, match="same length"):
        make_trainer(multi_step=True, horizons=[1, 2, 4], weights=[1.0])


def test_short_trajectory_skips_unavailable_horizon_explicitly(caplog):
    trainer = make_trainer(
        multi_step=True, horizons=[1, 4], weights=[1.0, 0.25]
    )
    obs, actions = make_segment(transitions=2)
    prepared_obs, prepared_actions = trainer._prepare_segment(obs, actions)
    _, components = trainer._multi_step_prediction_loss(
        trainer.wm.encode(prepared_obs, prepared_actions)
    )
    assert set(components) == {1}
    assert "Skipping adaptation horizon 4" in caplog.text


def test_short_trajectory_fails_if_no_horizon_is_available():
    trainer = make_trainer(multi_step=True, horizons=[4], weights=[1.0])
    obs, actions = make_segment(transitions=2)
    prepared_obs, prepared_actions = trainer._prepare_segment(obs, actions)
    with pytest.raises(ValueError, match="none of adapt_horizons"):
        trainer._multi_step_prediction_loss(trainer.wm.encode(prepared_obs, prepared_actions))
