import json
from pathlib import Path

import torch
from torch import nn

from models.vit import ViTPredictor, TransitionFiLMPredictor
from models.visual_world_model import VWorldModel


def tiny_predictors():
    args = dict(num_patches=2, num_frames=3, dim=8, depth=3,
                heads=2, mlp_dim=16, dropout=0.0, emb_dropout=0.0, pool="mean")
    torch.manual_seed(7)
    baseline = ViTPredictor(**args)
    film = TransitionFiLMPredictor(**args, visual_dim=4, action_dim=2, context_dim=12)
    missing, unexpected = film.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert all(name.startswith(("context_net.", "film.")) for name in missing)
    return baseline.eval(), film.eval()


def test_zero_initialized_film_is_exact_baseline_even_with_valid_context():
    baseline, film = tiny_predictors()
    x = torch.randn(2, 6, 8)
    context = torch.randn(2, 3, 10)
    valid = torch.tensor([[False, True, True], [True, True, False]])
    with torch.no_grad():
        torch.testing.assert_close(baseline(x), film(x, context, valid), atol=1e-6, rtol=1e-6)


def test_future_tokens_and_context_cannot_affect_earlier_slot():
    _, film = tiny_predictors()
    with torch.no_grad():
        film.film["2"].weight.fill_(0.01)
    x = torch.randn(1, 6, 8)
    context = torch.randn(1, 3, 10)
    valid = torch.tensor([[False, True, True]])
    x_future = x.clone()
    x_future[:, 2:] += 100
    context_future = context.clone()
    context_future[:, 1:] += 100
    with torch.no_grad():
        first = film(x, context, valid)[:, :2]
        changed = film(x_future, context_future, valid)[:, :2]
    torch.testing.assert_close(first, changed, atol=1e-5, rtol=1e-5)


def test_transition_context_uses_previous_action_chunk_and_no_target():
    model = object.__new__(VWorldModel)
    nn.Module.__init__(model)
    model.concat_dim = 1
    model.proprio_dim = 2
    model.action_dim = 2
    z = torch.randn(1, 3, 2, 8)
    context, valid = model._transition_context(z)
    assert valid.tolist() == [[False, True, True]]
    torch.testing.assert_close(context[:, 0], torch.zeros_like(context[:, 0]))
    torch.testing.assert_close(context[:, 1, 4:6], z[:, 0, 0, -2:])
    torch.testing.assert_close(context[:, 2, 4:6], z[:, 1, 0, -2:])
    future = z.clone()
    future[:, 2] += 20
    new_context, _ = model._transition_context(future)
    torch.testing.assert_close(context[:, :2], new_context[:, :2])


class FakeEncoder(nn.Module):
    name = "fake"
    emb_dim = 4

    def forward(self, image):
        value = image.mean(dim=(1, 2, 3))[:, None, None]
        return value.expand(-1, 2, 4)


class FakeEmbedding(nn.Module):
    emb_dim = 2

    def forward(self, value):
        return value[..., :2]


def test_mpc_rollout_holds_only_real_observed_context():
    _, film = tiny_predictors()
    model = VWorldModel(image_size=4, num_hist=3, num_pred=1,
                        encoder=FakeEncoder(), proprio_encoder=FakeEmbedding(),
                        action_encoder=FakeEmbedding(), decoder=None,
                        predictor=film, proprio_dim=2, action_dim=2,
                        concat_dim=1, num_action_repeat=1,
                        num_proprio_repeat=1, train_encoder=False,
                        train_predictor=True, train_decoder=False).eval()
    pair = {"visual": torch.rand(1, 2, 3, 4, 4), "proprio": torch.zeros(1, 2, 2)}
    model.set_observed_transition(pair, torch.tensor([[[0.3, 0.4]]]))
    saved = model._observed_transition_context.clone()
    captured = []
    original = film.forward

    def recording_forward(x, transition_context=None, context_valid=None):
        captured.append((transition_context.detach().clone(), context_valid.clone()))
        return original(x, transition_context, context_valid)

    film.forward = recording_forward
    current = {key: value[:, -1:] for key, value in pair.items()}
    with torch.no_grad():
        model.rollout(current, torch.randn(1, 4, 2))
    assert captured
    for context, valid in captured:
        assert valid.all()
        torch.testing.assert_close(context, saved[:, None].expand_as(context))
    torch.testing.assert_close(saved, model._observed_transition_context)
    model.clear_observed_transition()
    assert model._observed_transition_context is None


def test_color_table_excludes_red_from_training():
    from scripts.generate_transition_colors import COLORS
    assert list(COLORS) == ["default", "blue", "green"]
    assert "Red" not in COLORS.values()


def test_film_parameters_and_base_predictor_receive_gradients():
    _, film = tiny_predictors()
    x = torch.randn(2, 6, 8)
    context = torch.randn(2, 3, 10)
    valid = torch.tensor([[False, True, True], [False, True, True]])
    film(x, context, valid).square().mean().backward()
    assert film.film["2"].weight.grad is not None
    assert film.film["2"].weight.grad.abs().sum() > 0
    assert film.transformer.layers[0][0].to_qkv.weight.grad is not None
    # The last projection is zero-initialized, so the context MLP receives
    # gradients only after at least one FiLM projection update.
    with torch.no_grad():
        film.film["2"].weight.normal_(std=0.01)
    film.zero_grad(set_to_none=True)
    film(x, context, valid).square().mean().backward()
    assert film.context_net[1].weight.grad is not None
    assert film.context_net[1].weight.grad.abs().sum() > 0


def test_aggregation_uses_paper_curve_not_console_text(tmp_path):
    from scripts.aggregate_transition_film import read_curve
    log = tmp_path / "logs.json"
    log.write_text(
        '\n'.join([
            json.dumps({"mpc/success_rate": 0.75, "step": 1}),
            json.dumps({"paper/mpc_step": 1, "paper/success_rate_pct": 12.0}),
            json.dumps({"paper/mpc_step": 2, "paper/success_rate_pct": 24.0}),
        ]) + '\n', encoding="utf-8"
    )
    assert read_curve(log) == {1: 12.0, 2: 24.0}
