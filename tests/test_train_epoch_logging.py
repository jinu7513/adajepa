"""Test the epoch logger without importing optional training dependencies."""

import ast
import json
import logging
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class ScalarTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def item(self):
        return self.value


class VectorTensor(ScalarTensor):
    def reshape(self, *_args):
        return self

    def tolist(self):
        return self.value


def trainer_method(name):
    source = Path(__file__).resolve().parents[1] / "train.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {
        "torch": SimpleNamespace(is_tensor=lambda item: isinstance(item, ScalarTensor)),
        "np": np,
        "OrderedDict": OrderedDict,
        "json": json,
        "log": logging.getLogger(__name__),
        "open": open,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


def test_tensor_rollout_metrics_are_json_serializable(tmp_path, monkeypatch):
    logs_update = trainer_method("logs_update")
    logs_flash = trainer_method("logs_flash")
    logged = []
    trainer = SimpleNamespace(
        epoch=1,
        epoch_log=OrderedDict(),
        accelerator=SimpleNamespace(is_main_process=True),
        wandb_run=SimpleNamespace(log=logged.append),
    )

    logs_update(trainer, {
        "train_loss": [0.1, np.float32(0.2)],
        "val_loss": np.float32(0.3),
        "val_z_visual_err_rollout": [ScalarTensor(0.4), ScalarTensor(0.6)],
        "train_z_visual_err_rollout": VectorTensor([0.7, 0.9]),
    })
    assert all(isinstance(total, float) for _, total in trainer.epoch_log.values())

    monkeypatch.chdir(tmp_path)
    logs_flash(trainer, step=1)
    record = json.loads((tmp_path / "epoch_metrics.jsonl").read_text(encoding="utf-8"))
    assert record["epoch"] == 1
    assert record["val_z_visual_err_rollout"] == 0.5
    assert abs(record["train_z_visual_err_rollout"] - 0.8) < 1e-12
    assert logged == [record]
