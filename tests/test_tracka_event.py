import numpy as np
import pytest
import torch
from PIL import Image

from tracka.data import CLEAN_RGB
from tracka.event import event_from_rgb
from tracka.event_data import EventWindowDataset, generate_event_windows, validate_event_windows
from tracka.event_model import EventTeacherStudent, RGBEventEncoder, representation_distance
from tracka.model import patchify


def small_model(**overrides):
    model = dict(image_size=16, patch_size=8, embed_dim=24, heads=3,
                 rgb_depth=1, event_depth=1, shared_depth=1,
                 event_mode="hard_two_channel", event_threshold=.2, event_eps=.001,
                 shift_previous=False, fusion="fixed_sum",
                 global_representation="cls", rgb_input_mode="current")
    model.update(overrides)
    return model


def small_loss(**overrides):
    loss = dict(lambda_rob=1., lambda_temp=1., distance="cosine",
                anti_collapse="none", lambda_ac=0., sigreg_hidden_dim=32,
                sigreg_dim=16, sigreg_directions=8, sigreg_knots=5)
    loss.update(overrides)
    return loss


def test_event_transform_shapes_polarity_and_static():
    old = torch.full((2, 3, 16, 16), -1.)
    new = old.clone()
    new[:, :, :8, :8] = 1.
    two, diag = event_from_rgb(old, new, mode="hard_two_channel", patch_size=8,
                               threshold=.2)
    assert two.shape == (2, 2, 16, 16)
    assert torch.all(two[:, 0, :8, :8] == 1)
    assert torch.all(two[:, 1] == 0)
    assert diag["patch_density"].shape == (2, 4, 1)
    assert diag["event_density"].item() == pytest.approx(.25)
    for mode, channels in (("hard_signed", 1), ("delta_log", 1), ("soft_clipped", 1)):
        event, _ = event_from_rgb(old, new, mode=mode, patch_size=8)
        assert event.shape == (2, channels, 16, 16)
    static, _ = event_from_rgb(old, old, patch_size=8)
    assert static.count_nonzero() == 0


def test_fixed_fusion_and_teacher_student_gradients():
    torch.manual_seed(7)
    encoder = RGBEventEncoder(**small_model(rgb_depth=0, event_depth=0, shared_depth=0))
    previous = torch.rand(2, 3, 16, 16) * 2 - 1
    current = torch.rand_like(previous) * 2 - 1
    event, _ = event_from_rgb(previous, current, patch_size=8)
    rgb = encoder.rgb_norm(encoder.rgb_embed(patchify(current, 8)) + encoder.pos_embed)
    evt = encoder.event_norm(encoder.event_embed(patchify(event, 8)) + encoder.pos_embed)
    expected = encoder.final_norm(rgb + evt)
    actual = encoder(previous, current)
    assert actual["patch_tokens"].shape == (2, 4, 24)
    assert actual["global_token"].shape == (2, 24)
    torch.testing.assert_close(actual["patch_tokens"], expected)

    model = EventTeacherStudent(small_model(), small_loss())
    batch = {f"{view}_{frame}": torch.rand(2, 3, 16, 16) * 2 - 1
             for view in ("clean", "shift") for frame in ("previous", "current", "next")}
    loss, diagnostics = model.objective(batch)
    loss.backward()
    assert loss.isfinite()
    assert diagnostics["loss/robust"].isfinite()
    assert all(p.grad is None for p in model.teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.student.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.q.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.F.parameters())
    assert not hasattr(model, "decoder")
    before = next(model.teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(model.student.parameters()).add_(1.)
    model.update_teacher(.9)
    assert torch.allclose(next(model.teacher.parameters()), before + .1)


def test_gated_and_rgb_pair_ablation_shapes():
    previous = torch.randn(2, 3, 16, 16)
    current = torch.randn_like(previous)
    gated = RGBEventEncoder(**small_model(fusion="gated_sum"))
    output = gated(previous, current)
    assert output["gate"].shape == (2, 4, 1)
    assert 0 < output["gate"].mean() < .5
    rgb_pair = RGBEventEncoder(**small_model(fusion="rgb_only", rgb_input_mode="pair"))
    assert rgb_pair(previous, current)["patch_tokens"].shape == (2, 4, 24)
    event_only = RGBEventEncoder(**small_model(fusion="event_only", global_representation="mean_pool"))
    assert event_only(previous, current)["global_token"].shape == (2, 24)
    with pytest.raises(ValueError):
        RGBEventEncoder(**small_model(fusion="fixed_sum", rgb_input_mode="pair"))
    for distance in ("cosine", "normalized_mse", "mse"):
        assert representation_distance(current.flatten(1), current.flatten(1), distance) < 1e-6


def test_event_windows_use_original_neighbor_indices(tmp_path, monkeypatch):
    import tracka.event_data as module

    root = tmp_path / "pairs"
    root.mkdir()
    (root / "manifest.json").write_text("{}", encoding="utf-8")
    palette = CLEAN_RGB.tolist()
    states = {}
    rows = []
    splits = {"train": ["train/0"], "validation": ["train/1"], "test": ["train/2"]}
    for i, split in enumerate(splits):
        uid = splits[split][0]
        trajectory = np.zeros((5, 7), dtype=np.float32)
        trajectory[:, 0] = np.arange(5) + i * 5
        states[uid] = trajectory
        center = np.full((16, 16, 3), 4 + i * 5, dtype=np.uint8)
        Image.fromarray(center).save(root / f"{i}.png")
        rows.append({"sample_id": i, "trajectory_id": uid, "timestep": 2,
                     "split": split, "segment": "middle", "render_seed": 4,
                     "canonical_render_state": trajectory[2].tolist(),
                     "eta_A": palette, "eta_B": palette,
                     "images": {"clean": f"{i}.png"}})
    base = {"config": {"model": {"image_size": 16}, "dataset": {
        "source_path": "unused", "reset_tolerances": {
            "source": dict(agent_position=5., block_position=5., angle=.1, agent_velocity=.1),
            "canonical": dict(agent_position=0., block_position=0., angle=0., agent_velocity=0.)}}},
            "splits": splits}
    monkeypatch.setattr(module, "validate_dataset", lambda _: (base, rows))
    monkeypatch.setattr(module, "source_trajectories", lambda _: list(states.items()))

    class FakeEnv:
        def close(self):
            pass

    monkeypatch.setattr(module, "new_env", lambda _: FakeEnv())

    def fake_render(_, state, colors, seed):
        value = int(state[0]) + 2
        return np.full((16, 16, 3), value, dtype=np.uint8), np.array(state)

    monkeypatch.setattr(module, "render_state", fake_render)
    cfg = {"model": small_model(), "dataset": {
        "source_pairs_path": str(root), "output_path": str(tmp_path / "windows"),
        "shift_seed": 31, "shift_kinds": ["blur"],
        "corruption_strength": {"blur": [0., 0.]}},
        "generation": {"preview_only": False, "ranges_reviewed": True,
                       "preview_windows": 1}}
    result = generate_event_windows(cfg)
    assert result["sample_count"] == 3
    assert (tmp_path / "windows" / "window_contact_sheet.png").is_file()
    assert (tmp_path / "windows" / "event_contact_sheet.png").is_file()
    manifest, generated = validate_event_windows(tmp_path / "windows")
    assert manifest["splits"] == splits
    assert generated[0]["source_states"][0][0] == 1  # t-1, not another pair row.
    assert generated[0]["source_states"][2][0] == 3
    dataset = EventWindowDataset(tmp_path / "windows", "train", 16)
    assert dataset[0]["clean_previous"].shape == (3, 16, 16)
