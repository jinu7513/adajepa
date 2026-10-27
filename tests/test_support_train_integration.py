from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from models.vit import ViTPredictor
from train_transition_support import make_model, query_loss, save_checkpoint
from test_support_transition import FakeEncoder, FakeEmbedding, fake_sequence


def source_fixture():
    cfg = OmegaConf.create({
        "predictor": {"_target_": "models.vit.ViTPredictor", "depth": 3,
                      "heads": 2, "mlp_dim": 16, "dropout": 0.0,
                      "emb_dropout": 0.0, "pool": "mean"},
        "num_hist": 3, "num_pred": 1, "img_size": 4,
        "num_action_repeat": 1, "num_proprio_repeat": 1,
        "proprio_emb_dim": 2, "action_emb_dim": 2, "concat_dim": 1,
    })
    source = {
        "encoder": FakeEncoder(),
        "action_encoder": FakeEmbedding(),
        "proprio_encoder": FakeEmbedding(),
        "predictor": ViTPredictor(num_patches=2, num_frames=3, dim=8,
                                  depth=3, heads=2, mlp_dim=16, dropout=0,
                                  emb_dropout=0, pool="mean"),
    }
    return source, cfg


@pytest.mark.parametrize("feature_mode,train_base", [("delta", False), ("residual", True)])
def test_source_init_query_loss_and_resumable_checkpoint(tmp_path, feature_mode, train_base):
    source, source_cfg = source_fixture()
    experiment = OmegaConf.load(Path(__file__).resolve().parents[1] / "conf/transition_support.yaml")
    experiment.context_dim = 12
    experiment.film_layers = 2
    experiment.feature_mode = feature_mode
    experiment.train_base_predictor = train_base
    model = make_model(source, source_cfg, experiment)
    observations, actions = fake_sequence()
    loss, visual_loss, context, valid = query_loss(model, observations, actions, support_k=4)
    assert torch.isfinite(loss) and torch.isfinite(visual_loss)
    assert context.shape == (2, 12) and valid.all()
    loss.backward()
    assert model.predictor.film["2"][-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    optimizer = torch.optim.AdamW((p for p in model.predictor.parameters() if p.requires_grad), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    path = tmp_path / "checkpoints" / "model_latest.pth"
    save_checkpoint(path, model, optimizer, scaler, 0, 1.0, experiment, tmp_path,
                    resume_epoch=1, next_batch=1000, train_sum=2.0, train_steps=1000)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["resume_epoch"] == 1 and loaded["next_batch"] == 1000
    assert loaded["train_steps"] == 1000 and loaded["checkpoint_version"] == 1
