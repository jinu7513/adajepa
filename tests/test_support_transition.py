import torch
from torch import nn

from models.vit import ViTPredictor
from models.support_transition import SupportFiLMPredictor, SupportVWorldModel


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


def tiny_model(mode="delta", frozen_base=True):
    args = dict(num_patches=2, num_frames=3, dim=8, depth=3,
                heads=2, mlp_dim=16, dropout=0, emb_dropout=0, pool="mean")
    torch.manual_seed(12)
    reference = ViTPredictor(**args)
    predictor = SupportFiLMPredictor(**args, visual_dim=4, action_dim=2,
                                     context_dim=12, feature_mode=mode)
    missing, unexpected = predictor.load_state_dict(reference.state_dict(), strict=False)
    assert not unexpected
    assert all(name.startswith(("feature_encoder.", "support_aggregator.", "film.")) for name in missing)
    if mode == "residual":
        predictor.attach_reference(reference)
    if frozen_base:
        for name, parameter in predictor.named_parameters():
            parameter.requires_grad_(name.startswith(("feature_encoder.", "support_aggregator.", "film.")))
    model = SupportVWorldModel(
        image_size=4, num_hist=3, num_pred=1,
        encoder=FakeEncoder(), proprio_encoder=FakeEmbedding(),
        action_encoder=FakeEmbedding(), decoder=None, predictor=predictor,
        proprio_dim=2, action_dim=2, concat_dim=1,
        num_action_repeat=1, num_proprio_repeat=1,
        train_encoder=False, train_predictor=True, train_decoder=False,
    )
    return reference.eval(), model.eval()


def fake_sequence(batch=2, frames=10):
    observations = {"visual": torch.rand(batch, frames, 3, 4, 4),
                    "proprio": torch.rand(batch, frames, 2)}
    actions = torch.rand(batch, frames, 2)
    return observations, actions


def test_film_is_exact_identity_for_zero_projection_and_k0():
    reference, model = tiny_model()
    x = torch.randn(2, 6, 8)
    context = torch.randn(2, 12)
    valid = torch.ones(2, dtype=torch.bool)
    with torch.no_grad():
        torch.testing.assert_close(reference(x), model.predictor(x, context, valid), atol=1e-6, rtol=1e-6)
        context0 = torch.zeros_like(context)
        torch.testing.assert_close(reference(x), model.predictor(x, context0, ~valid), atol=1e-6, rtol=1e-6)


def test_k_sweep_and_query_target_cannot_change_context():
    _, model = tiny_model()
    obs, act = fake_sequence()
    z = model.encode(obs, act)
    future = z.clone()
    future[:, -1] += 50
    for k in (0, 2, 4, 8):
        context, valid = model.support_context_from_sequence(z, k)
        changed, new_valid = model.support_context_from_sequence(future, k)
        assert context.shape == (2, 12)
        assert valid.tolist() == [k > 0] * 2
        torch.testing.assert_close(context, changed)
        torch.testing.assert_close(valid, new_valid)
    try:
        model.set_active_support_k(9)
        assert False, "Unsupported K accepted"
    except ValueError:
        pass


def test_zero_initialization_delays_context_gradients_but_not_film_gradients():
    _, model = tiny_model()
    obs, act = fake_sequence()
    z = model.encode(obs, act).detach()
    context, valid = model.support_context_from_sequence(z, 4)
    model.predict(z[:, -4:-1], support_context=context, context_valid=valid).square().mean().backward()
    projection = model.predictor.film["2"][-1]
    assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0
    with torch.no_grad():
        projection.weight.normal_(std=0.01)
    model.zero_grad(set_to_none=True)
    context, valid = model.support_context_from_sequence(z, 4)
    model.predict(z[:, -4:-1], support_context=context, context_valid=valid).square().mean().backward()
    weight = model.predictor.feature_encoder.net[1].weight
    assert weight.grad is not None and weight.grad.abs().sum() > 0
    assert model.predictor.transformer.layers[0][0].to_qkv.weight.grad is None


def test_completed_support_only_and_reset_between_episodes():
    _, model = tiny_model()
    obs, act = fake_sequence(batch=1, frames=4)
    assert not model._support_entries
    with torch.no_grad():
        model.observe_rollout(obs, act[:, :3])
    assert len(model._support_entries) == 3
    snapshot = model.get_observed_support_state()
    captured = []
    original = model.predictor.forward

    def recording(x, context=None, context_valid=None):
        captured.append(context.detach().clone())
        return original(x, context, context_valid)

    model.predictor.forward = recording
    current = {key: value[:, -1:] for key, value in obs.items()}
    with torch.no_grad():
        model.rollout(current, torch.rand(1, 4, 2))
    assert len(captured) >= 2
    for context in captured[1:]:
        torch.testing.assert_close(context, captured[0])
    assert len(model._support_entries) == 3
    model.clear_observed_transition()
    assert not model._support_entries
    model.restore_observed_support_state(snapshot)
    assert len(model._support_entries) == 3


def test_residual_requires_frozen_reference_and_checkpoint_roundtrip(tmp_path):
    _, model = tiny_model("residual")
    obs, act = fake_sequence()
    z = model.encode(obs, act)
    context, valid = model.support_context_from_sequence(z, 8)
    assert context.shape == (2, 12) and valid.all()
    assert all(not p.requires_grad for p in model.predictor.reference_predictor.parameters())
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 1, "predictor": model.predictor}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)["predictor"]
    x = torch.randn(1, 6, 8)
    with torch.no_grad():
        torch.testing.assert_close(model.predictor(x), loaded(x))


def test_matched_window_identity_trains_only_base_predictor():
    _, model = tiny_model(frozen_base=False)
    for name, parameter in model.predictor.named_parameters():
        parameter.requires_grad_(not name.startswith(("feature_encoder.", "support_aggregator.", "film.")))
    obs, act = fake_sequence()
    z = model.encode(obs, act).detach()
    context, valid = model.support_context_from_sequence(z, 0)
    loss = (model.predict(z[:, -4:-1], support_context=context, context_valid=valid)[:, -1]
            - z[:, -1]).square().mean()
    loss.backward()
    assert model.predictor.transformer.layers[0][0].to_qkv.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.predictor.film.parameters())
