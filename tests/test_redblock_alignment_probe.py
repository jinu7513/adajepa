"""Check the diagnostic's alignment and pose invariants on synthetic data."""

import numpy as np
import torch

from scripts.probe_redblock_alignment import (
    ResidualAdapter,
    apply_adapter,
    fit_oracle,
    paired_metrics,
    pose_metrics,
)


def test_identity_initialized_adapter_and_oracle_repair_a_color_shift():
    rng = np.random.RandomState(7)
    gray_train = rng.normal(size=(256, 16)).astype(np.float32)
    gray_val = rng.normal(size=(64, 16)).astype(np.float32)
    color_shift = np.linspace(-0.4, 0.4, 16, dtype=np.float32)
    red_train = gray_train + color_shift
    red_val = gray_val + color_shift

    identity = ResidualAdapter(16, 32)
    torch.testing.assert_close(identity(torch.from_numpy(red_val)), torch.from_numpy(red_val))

    adapted = fit_oracle(red_train, gray_train, 16, 32, 300, 3e-3,
                         torch.device("cpu"), seed=0)
    before = paired_metrics(red_val, gray_val)
    after = paired_metrics(apply_adapter(adapted, red_val, torch.device("cpu")), gray_val)
    assert after["paired_mse"] < before["paired_mse"]


def test_pose_angle_error_wraps_at_pi_boundary():
    actual = np.array([[200, 250, 300, 350, np.sin(np.pi - 0.01),
                        np.cos(np.pi - 0.01)]], dtype=np.float64)
    predicted = np.array([[200, 250, 300, 350, np.sin(-np.pi + 0.01),
                           np.cos(-np.pi + 0.01)]], dtype=np.float64)
    metrics = pose_metrics(predicted, actual)
    assert metrics["agent_position_rmse_px"] == 0
    assert metrics["block_position_rmse_px"] == 0
    assert np.isclose(metrics["block_angle_mae_deg"], 0.02 * 180 / np.pi)
