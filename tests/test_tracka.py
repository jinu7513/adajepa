import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from planning.image_corruption import corrupt_frames
from tracka.common import load_checkpoint
from tracka.data import (CLEAN_RGB, FourViewDataset, check_state, generate, new_env,
                         render_state, sample_colors, state_errors, validate_dataset, valid_colors)
from tracka.logging import RunLogger
from tracka.model import (RobustMAE, ScratchEncoder, gather_tokens, patchify,
                          resolved_mask_ratios, resolved_variance_config, sample_mask,
                          unpatchify, variance_hinge)
from tracka.probes import LinearProbe, evaluate, physical_targets
from tracka.train import train


@pytest.fixture
def cfg(tmp_path):
    cfg = OmegaConf.to_container(OmegaConf.load(Path(__file__).parents[1] / "conf/tracka.yaml"), resolve=False)
    cfg.pop("hydra")
    cfg["model"].update(image_size=32, patch_size=8, embed_dim=24, depth=1, heads=3)
    cfg["decoder"].update(dim=12, depth=1, heads=3, rgb_embed_dim=8)
    cfg["dataset"].update(source_path=str(tmp_path / "source"), output_path=str(tmp_path / "pairs"),
                           frame_stride=1, max_states_per_trajectory=3)
    cfg["generation"].update(preview_only=False, ranges_reviewed=True, preview_states=2)
    cfg["training"].update(total_optimizer_updates=2, batch_size=2, device="cpu", cpu_threads=1,
                            validate_every=2, save_every=1, validation_batches=1)
    cfg["logging"].update(mode="disabled", log_every=1, image_every=2)
    cfg["evaluation"].update(device="cpu", batch_size=4, max_states_per_split=6)
    return cfg


@pytest.fixture
def generated(cfg):
    root = Path(cfg["dataset"]["source_path"]) / "train"
    root.mkdir(parents=True)
    states = torch.zeros(9, 3, 5)
    for i in range(9):
        for t in range(3):
            states[i, t] = torch.tensor([60 + i * 4 + t, 70 + i + t, 250 + t * 4, 280 + i, .2 + i*.03])
    torch.save(states, root / "states.pth")
    torch.save(torch.zeros(9, 3, 2), root / "velocities.pth")
    with open(root / "seq_lengths.pkl", "wb") as f:
        pickle.dump([3] * 9, f)
    generate(cfg)
    return cfg


def random_batch(cfg, b=2):
    size = cfg["model"]["image_size"]
    batch = {k: torch.randn(b, 3, size, size) for k in ("clean", "A", "B", "corrupt")}
    batch.update({k: torch.rand(b, 3, 3) for k in ("eta_clean", "eta_A", "eta_B")})
    return batch


def test_corruption_semantics_and_legacy():
    x = np.full((512, 512, 3), 127, dtype=np.uint8)
    y = corrupt_frames(x, "salt_pepper", .2, 3)
    assert abs((y[..., 0] != 127).mean() - .2) < .004
    assert abs((y[..., 0] == 255).mean() - .1) < .004
    assert np.array_equal(y, corrupt_frames(x, "salt_pepper", .2, 3))
    assert np.array_equal(x, corrupt_frames(x, "blur", 0))
    for kind in ("gaussian", "blur", "snp1", "snp5", "dark"):
        assert corrupt_frames(x, kind).shape == x.shape
    # Lock the historical draw algorithm, including repeats and overwrite order.
    rng = np.random.RandomState(7)
    expected = x.copy()
    n = int(.01 * 512 * 512)
    ys = rng.randint(0, 512, n); xs = rng.randint(0, 512, n)
    expected[ys, xs] = 255
    ys = rng.randint(0, 512, n); xs = rng.randint(0, 512, n)
    expected[ys, xs] = 0
    assert np.array_equal(expected, corrupt_frames(x, "snp1", seed=7))
    with pytest.raises(ValueError):
        corrupt_frames(x, "salt_pepper", 2)


def test_mask_restore_and_original_positions(cfg):
    torch.manual_seed(4)
    enc = ScratchEncoder(**cfg["model"]).eval()
    x = torch.randn(2, 3, 32, 32)
    patches = patchify(x, 8)
    assert torch.equal(unpatchify(patches, 8), x)
    keep, restore, mask = sample_mask(2, 16, .5, "cpu")
    assert keep.shape == (2, 8) and torch.all(mask.sum(1) == 8)
    visible = gather_tokens(patches, keep)
    joined = torch.cat([visible, torch.zeros_like(visible)], 1)
    assert torch.equal(gather_tokens(joined, restore), patches * (1-mask[..., None]))
    # With no transformer blocks, visible features equal gathering full features.
    enc.blocks = torch.nn.ModuleList([])
    assert torch.allclose(enc.forward_visible(x, keep), gather_tokens(enc(x), keep), atol=1e-6)


def test_objective_specific_mask_ratios(cfg, monkeypatch):
    import tracka.model as model_module

    cfg["mask"].update(render_ratio=.5, corruption_ratio=.25)
    model = RobustMAE(cfg["model"], cfg["decoder"], cfg["loss"], cfg["mask"])
    batch = random_batch(cfg)
    observed = []
    original = model_module.sample_mask

    def record_mask(batch_size, patches, ratio, device, generator=None):
        observed.append(ratio)
        return original(batch_size, patches, ratio, device, generator)

    monkeypatch.setattr(model_module, "sample_mask", record_mask)
    for mode, expected in (("render_only", .5), ("corruption_only", .25), ("clean_mae", .5)):
        loss, logs = model.objective(batch, mode)
        assert torch.isfinite(loss)
        assert observed[-1] == expected
        assert logs["mask/ratio"].item() == expected
    model.preview(batch, mode="corruption_only")
    assert observed[-1] == .25
    assert resolved_mask_ratios({"ratio": .5}) == resolved_mask_ratios(
        {"ratio": .5, "render_ratio": None, "corruption_ratio": None})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        resolved_mask_ratios({"ratio": .5, "corruption_ratio": 0})


@pytest.mark.parametrize("conditioning", ["additive", "cross_attention"])
@pytest.mark.parametrize("mode", ["clean_mae", "render_only", "corruption_only"])
def test_objective_backward(cfg, conditioning, mode):
    torch.set_num_threads(1)
    cfg["decoder"]["conditioning"] = conditioning
    model = RobustMAE(cfg["model"], cfg["decoder"], cfg["loss"], cfg["mask"])
    batch = random_batch(cfg)
    loss, logs = model.objective(batch, mode)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0
    assert model.decoder.rgb_mlp[0].weight.grad.abs().sum() > 0
    if mode != "clean_mae":
        model.zero_grad()
        _, logs = model.objective(batch, mode)
        prefix = "render" if mode == "render_only" else "corruption"
        logs[prefix + "/invariance"].backward()
        assert model.encoder.patch_embed.weight.grad.abs().sum() > 0
    # Metadata and state never enter the model interface.
    assert all("state" not in key for key in batch)


def test_default_shapes_bidirectional_and_wrapper():
    torch.set_num_threads(1)
    enc = ScratchEncoder(depth=1).eval()
    x = torch.randn(1, 3, 224, 224)
    assert enc(x).shape == (1, 196, 384)
    keep, _, _ = sample_mask(1, 196, .5, "cpu")
    assert enc.forward_visible(x, keep).shape == (1, 98, 384)
    a = enc(x)[:, 0].detach()
    x[:, :, -16:, -16:] += 2
    assert not torch.allclose(a, enc(x)[:, 0])  # Last patch can affect first patch.
    from models.visual_world_model import VWorldModel
    world = VWorldModel(image_size=224, num_hist=1, num_pred=1, encoder=enc,
                       proprio_encoder=torch.nn.Identity(), action_encoder=torch.nn.Identity(),
                       decoder=None, predictor=None)
    obs = world.encode_obs({"visual": x[:, None], "proprio": torch.zeros(1, 1, 4)})
    assert obs["visual"].shape == (1, 1, 196, 384)


def test_cls_contract_and_global(cfg):
    cfg["model"]["use_cls"] = True
    cfg["decoder"]["use_cls_global_decoder"] = True
    cfg["loss"]["lambda_cls_global"] = .1
    model = RobustMAE(cfg["model"], cfg["decoder"], cfg["loss"], cfg["mask"])
    batch = random_batch(cfg)
    assert model.encoder(batch["clean"]).shape == (2, 16, 24)
    assert model.encoder.forward_features(batch["clean"], return_cls=True)["cls_token"].shape == (2, 24)
    loss, _ = model.objective(batch, "render_only")
    loss.backward()
    assert model.decoder.global_proj.weight.grad.abs().sum() > 0


def test_label_free_variance_regularization(cfg, monkeypatch):
    import tracka.model as model_module

    flat, flat_std = variance_hinge(torch.ones(4, 8), .1)
    varied, varied_std = variance_hinge(torch.tensor([[0.], [1.], [2.], [3.]]), .1)
    assert flat > 0 and flat_std < .1
    assert varied == 0 and varied_std > .1
    cfg["model"]["use_cls"] = True
    cfg["loss"].update(lambda_variance=.1, variance_target_std=2.0,
                       variance_feature="both")
    model = RobustMAE(cfg["model"], cfg["decoder"], cfg["loss"], cfg["mask"])
    batch = random_batch(cfg)
    observed = []
    original = model_module.ScratchEncoder.forward_features

    def record_features(self, images, return_cls=False, ids_keep=None):
        observed.append(ids_keep is None)
        return original(self, images, return_cls=return_cls, ids_keep=ids_keep)

    monkeypatch.setattr(model_module.ScratchEncoder, "forward_features", record_features)
    loss, logs = model.objective(batch, "corruption_only")
    assert torch.isfinite(loss)
    assert observed == [False, False, True]
    assert logs["loss/variance"] > 0
    assert logs["loss/variance_patch_mean"] > 0
    assert logs["loss/variance_cls"] > 0
    assert torch.allclose(logs["loss/weighted_variance"], .1 * logs["loss/variance"])
    model.zero_grad()
    logs["loss/variance"].backward()
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0
    assert all("state" not in key for key in batch)
    no_cls = copy.deepcopy(cfg)
    no_cls["model"]["use_cls"] = False
    with pytest.raises(ValueError, match="requires model.use_cls=true"):
        RobustMAE(no_cls["model"], no_cls["decoder"], no_cls["loss"], no_cls["mask"])
    assert resolved_variance_config({"lambda_cls_global": 0}) == resolved_variance_config(
        {"lambda_cls_global": 0, "lambda_variance": 0.0,
         "variance_target_std": .1, "variance_feature": "patch_mean"})


def test_colors_reset_and_wrapped_angle(cfg):
    rng = np.random.RandomState(2)
    cc = cfg["dataset"]["color_sampling"]
    a, _ = sample_colors(rng, [CLEAN_RGB], cc)
    b, _ = sample_colors(rng, [CLEAN_RGB, a], cc)
    assert valid_colors(a, [CLEAN_RGB], cc) and valid_colors(b, [CLEAN_RGB, a], cc)
    assert not valid_colors(np.full((3,3), 255), [], cc)
    assert not valid_colors(np.tile([200, 0, 0], (3,1)), [], cc)
    source = np.array([70., 80., 256., 280., .2, 3., -2.])
    env = new_env(224)
    try:
        clean, canonical = render_state(env, source, CLEAN_RGB, 1)
        shifted, state = render_state(env, source, a, 1)
        assert np.array_equal(canonical, state)
        assert not np.array_equal(clean, shifted)
        assert check_state(state, canonical, cfg["dataset"]["reset_tolerances"]["canonical"])[0]
        wrapped = canonical.copy()
        wrapped[4] += 2 * np.pi
        assert state_errors(wrapped, canonical)["angle"] < 1e-5
    finally:
        env.close()


def test_dataset_integrity_and_labels(generated):
    root = Path(generated["dataset"]["output_path"])
    manifest, rows = validate_dataset(root)
    assert manifest["image_count"] == manifest["sample_count"] * 4
    assert {r["segment"] for r in rows} == {"early", "middle", "late"}
    assert physical_targets(rows).shape == (len(rows), 8)
    ds = FourViewDataset(root, "train", 32)
    assert "state" not in ds[0]
    assert -1 <= ds[0]["clean"].min() <= ds[0]["clean"].max() <= 1


@pytest.mark.parametrize("objective", ["clean_mae", "render_only", "corruption_only", "robust_alternating"])
def test_training_modes_and_checkpoint(generated, tmp_path, objective):
    cfg = generated
    cfg["training"].update(objective=objective, output_dir=str(tmp_path / objective))
    out = train(cfg)
    state = load_checkpoint(out / "checkpoint_latest.pt")
    assert state["global_step"] == 2 and isinstance(state["encoder"], dict)
    assert state["next_update_type"] == ("render_only" if objective == "robust_alternating" else objective)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["train/global_step"] == 1
    assert records[-1]["train/global_step"] == 2
    assert (out / "reconstruction_00000002.png").exists()


def test_resume_equivalence(generated, tmp_path):
    cfg = generated
    cfg["training"].update(objective="robust_alternating", output_dir=str(tmp_path / "resumed"))
    out = train(cfg)
    # An older checkpoint stored only mask.ratio; its effective schedule is unchanged.
    legacy = load_checkpoint(out / "checkpoint_latest.pt")
    legacy["config"]["mask"].pop("render_ratio")
    legacy["config"]["mask"].pop("corruption_ratio")
    for key in ("lambda_variance", "variance_target_std", "variance_feature"):
        legacy["config"]["loss"].pop(key)
    legacy_path = out / "legacy_ratio_only.pt"
    torch.save(legacy, legacy_path)
    cfg["training"].update(total_optimizer_updates=4, resume=str(legacy_path))
    train(cfg)
    resumed = load_checkpoint(out / "checkpoint_latest.pt")
    cfg["training"].update(output_dir=str(tmp_path / "uninterrupted"), resume=None)
    complete = load_checkpoint(train(cfg) / "checkpoint_latest.pt")
    for name in ("encoder", "decoder"):
        for key, value in complete[name].items():
            assert torch.equal(value, resumed[name][key]), key


def test_mixed_mask_training_and_resume_guard(generated, tmp_path):
    cfg = generated
    cfg["mask"].update(render_ratio=.5, corruption_ratio=.25)
    cfg["training"].update(objective="robust_alternating", output_dir=str(tmp_path / "mixed"))
    out = train(cfg)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    ratios = [record["mask/ratio"] for record in records if "mask/ratio" in record]
    assert ratios[:2] == [.5, .25]
    cfg["training"].update(total_optimizer_updates=4, resume=str(out / "checkpoint_latest.pt"))
    cfg["mask"]["corruption_ratio"] = .5
    with pytest.raises(ValueError, match="Resume configuration differs: mask"):
        train(cfg)
    cfg["mask"]["corruption_ratio"] = .25
    cfg["loss"]["lambda_variance"] = .1
    with pytest.raises(ValueError, match="Resume configuration differs: loss"):
        train(cfg)


def test_variance_training_smoke(generated, tmp_path):
    cfg = generated
    cfg["model"]["use_cls"] = True
    cfg["decoder"]["use_cls_global_decoder"] = True
    cfg["loss"].update(lambda_cls_global=.1, lambda_variance=.1,
                       variance_target_std=.1, variance_feature="both")
    cfg["training"].update(objective="robust_alternating", output_dir=str(tmp_path / "variance"))
    out = train(cfg)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in records if "loss/weighted_variance" in r]
    assert len(updates) == 2
    assert all("latent/std_full_clean_patch_mean" in r and "latent/std_full_clean_cls" in r
               for r in updates)
    state = load_checkpoint(out / "checkpoint_latest.pt")
    assert state["config"]["loss"]["lambda_variance"] == .1


def test_linear_probe_toy():
    rng = np.random.RandomState(5)
    x = rng.normal(size=(100, 6))
    y = x @ rng.normal(size=(6, 8)) + 2
    probe = LinearProbe().fit(x[:80], y[:80], 1e-5)
    assert np.mean((probe.predict(x[80:]) - y[80:])**2) < 1e-9
    rgb = x[:, :3] * .2 + .5
    p = LinearProbe().fit(x[:80], rgb[:80], 1e-5)
    assert np.mean((p.predict(x[80:]) - rgb[80:])**2) < 1e-9


def test_probe_end_to_end(generated, tmp_path):
    cfg = generated
    cfg["evaluation"].update(encoder="random", output_dir=str(tmp_path / "eval"))
    out = evaluate(cfg)
    rows = json.loads((out / "results.json").read_text())
    assert {"default", "redBlock", "redAgent", "redAnchor"} <= {r["condition"] for r in rows}
    assert any(r["segment"] == "middle" for r in rows)
    assert (out / "physical_probe.npz").exists()
    assert (out / "robustness.png").exists()


def test_logger_lifecycle_and_local_recovery(cfg, tmp_path):
    with pytest.raises(RuntimeError):
        with RunLogger(tmp_path / "failed", cfg["logging"], {"config": cfg}) as logger:
            logger.log({"train/global_step": 1, "clean/mae": .5})
            # Records are flushed while the run is still active.
            assert "clean/mae" in (tmp_path / "failed/metrics.jsonl").read_text()
            raise RuntimeError("deliberate test error")
    assert json.loads((tmp_path / "failed/run_status.json").read_text())["status"] == "failed"


def test_real_wandb_offline(cfg, tmp_path):
    cfg["logging"].update(mode="offline", fallback_to_local=False, name="tracka-smoke")
    with RunLogger(tmp_path / "offline", cfg["logging"], {"config": cfg}) as logger:
        logger.log({"train/global_step": 1, "render/cross_mae": .5})
        logger.log({"train/global_step": 2, "corruption/denoise_mae": .4})
        from PIL import Image
        fig = tmp_path / "figure.png"
        Image.new("RGB", (32, 32), "white").save(fig)
        rows = [{"condition": "default", "segment": "full", "corruption_type": None,
                 "strength": None, "standardized_mse": .3}]
        rows += [{"condition": kind, "segment": "full", "corruption_type": kind,
                  "strength": .1, "standardized_mse": .4} for kind in ("gaussian", "salt_pepper", "blur")]
        logger.robustness_plots(rows, fig)
        assert logger.error is None
        assert logger.run_id
    assert list((tmp_path / "offline/wandb").glob("offline-run-*/*.wandb"))


def test_wandb_init_failure_keeps_local_logging(cfg, tmp_path, monkeypatch):
    import wandb
    def fail(**kwargs):
        raise RuntimeError("simulated unreachable service")
    monkeypatch.setattr(wandb, "init", fail)
    cfg["logging"].update(mode="online", fallback_to_local=True)
    with pytest.warns(UserWarning, match="W&B unavailable"):
        with RunLogger(tmp_path / "fallback", cfg["logging"], {"config": cfg}) as logger:
            logger.log({"train/global_step": 1, "clean/mae": .1})
    assert "clean/mae" in (tmp_path / "fallback/metrics.jsonl").read_text()
    status = json.loads((tmp_path / "fallback/run_status.json").read_text())
    assert status["status"] == "finished" and "simulated" in status["wandb_error"]


def test_alternating_logging_cadence(generated, tmp_path):
    generated["training"].update(objective="robust_alternating", total_optimizer_updates=6,
                                  output_dir=str(tmp_path / "logging"))
    generated["logging"]["log_every"] = 2
    out = train(generated)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    render_steps = [r["train/global_step"] for r in records if "render/cross_mae" in r]
    corr_steps = [r["train/global_step"] for r in records if "corruption/denoise_mae" in r]
    assert 3 in render_steps and 4 in corr_steps


def test_python39_syntax_contract():
    import ast
    root = Path(__file__).parents[1]
    for path in list((root / "tracka").glob("*.py")) + [root / n for n in
                 ("generate_pusht_pairs.py", "train_encoder.py", "eval_encoder_probes.py",
                  "recover_tracka_wandb.py", "aggregate_tracka_results.py")]:
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
