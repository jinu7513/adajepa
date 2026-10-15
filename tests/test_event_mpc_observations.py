"""CPU checks for pairing Event MPC observations with actual prior frames."""

import numpy as np
import pickle
import pytest
import torch
from torch import nn

from planning.event_observation import with_previous_visual
from preprocessor import Preprocessor


def make_preprocessor():
    return Preprocessor(
        action_mean=torch.zeros(2), action_std=torch.ones(2),
        state_mean=torch.zeros(7), state_std=torch.ones(7),
        proprio_mean=torch.zeros(4), proprio_std=torch.ones(4),
        transform=lambda image: image,
    )


def test_previous_visual_is_exact_previous_rendered_frame():
    visual = np.arange(2 * 3 * 2 * 2 * 3, dtype=np.uint8).reshape(2, 3, 2, 2, 3)
    original = {"visual": visual, "proprio": np.zeros((2, 3, 4))}
    paired = with_previous_visual(original)
    assert "visual_previous" not in original
    np.testing.assert_array_equal(paired["visual_previous"][:, 0], visual[:, 0])
    np.testing.assert_array_equal(paired["visual_previous"][:, 1:], visual[:, :-1])
    np.testing.assert_array_equal(paired["visual"], visual)
    assert paired["proprio"] is original["proprio"]


def test_preprocessor_transforms_event_pair_like_current_image():
    visual = np.full((1, 2, 2, 2, 3), 51, dtype=np.uint8)
    visual[:, 1] = 204
    paired = with_previous_visual({"visual": visual, "proprio": np.zeros((1, 2, 4))})
    processed = make_preprocessor().transform_obs(paired)
    torch.testing.assert_close(processed["visual_previous"][:, 0], processed["visual"][:, 0])
    torch.testing.assert_close(processed["visual_previous"][:, 1], processed["visual"][:, 0])
    assert processed["visual_previous"].shape == processed["visual"].shape


@pytest.mark.parametrize("corruption", [None, "blur"])
def test_evaluator_uses_previous_frame_after_visual_corruption(corruption):
    from planning.evaluator import PlanEvaluator

    class FakeWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.encoder = type("EventEncoder", (), {"requires_previous_frame": True})()

        def rollout(self, obs_0, act):
            assert "visual_previous" in obs_0
            batch, length = act.shape[:2]
            return {
                "visual": torch.zeros(batch, length + 1, 1, 1),
                "proprio": torch.zeros(batch, length + 1, 4),
            }, None

        def encode_obs(self, obs):
            assert "visual_previous" in obs
            return {"visual": torch.zeros(1, 1, 1, 1), "proprio": torch.zeros(1, 1, 4)}

    class FakeEnv:
        def rollout(self, seed, state, actions):
            visual = np.zeros((1, 3, 4, 4, 3), dtype=np.uint8)
            visual[:, 1, 1:3, 1:3] = 200
            visual[:, 2, 0:2, 0:2] = 150
            return {"visual": visual, "proprio": np.zeros((1, 3, 4))}, np.zeros((1, 3, 7))

        def eval_state(self, target, actual):
            return {"success": np.array([False])}

    initial = {"visual": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8),
               "proprio": np.zeros((1, 1, 4))}
    initial = with_previous_visual(initial)
    evaluator = PlanEvaluator(
        obs_0=initial, obs_g=initial, state_0=np.zeros((1, 7)),
        state_g=np.zeros((1, 7)), env=FakeEnv(), wm=FakeWorld(),
        frameskip=1, seed=[1], preprocessor=make_preprocessor(),
        n_plot_samples=0, decode_for_viz=False, ood_corruption=corruption,
    )
    _, _, observations, _ = evaluator.eval_actions(torch.zeros(1, 2, 2))
    np.testing.assert_array_equal(observations["visual_previous"][:, 0], observations["visual"][:, 0])
    np.testing.assert_array_equal(observations["visual_previous"][:, 1:], observations["visual"][:, :-1])


def test_event_adaptation_selects_only_last_shared_block_and_norm():
    from planning.adajepa import AdaJEPATrainer

    backbone = nn.Module()
    backbone.shared_blocks = nn.ModuleList([nn.Linear(3, 3), nn.Linear(3, 3)])
    backbone.final_norm = nn.LayerNorm(3)
    encoder = nn.Module()
    encoder.backbone = backbone
    encoder.requires_previous_frame = True
    trainer = AdaJEPATrainer.__new__(AdaJEPATrainer)
    trainer.wm = type("World", (), {"encoder": encoder})()
    trainer.encoder_last_layer_only = True
    selected = trainer._select_encoder_params()
    expected = list(backbone.shared_blocks[-1].parameters()) + list(backbone.final_norm.parameters())
    assert [id(param) for param in selected] == [id(param) for param in expected]


@pytest.mark.parametrize("corruption", [None, "blur"])
def test_segment_goal_keeps_actual_previous_frame(tmp_path, corruption):
    from plan import PlanWorkspace

    targets = tmp_path / "plan_targets.pkl"
    segment = {"states": [[0.0] * 5], "actions": [[0.0, 0.0]] * 25, "shape": "T"}
    with targets.open("wb") as stream:
        pickle.dump({"segments": [segment], "traj_len": 25}, stream)

    class FakeEnv:
        def update_env(self, env_info):
            self.env_info = env_info

        def rollout(self, seed, states, actions):
            visual = np.zeros((1, 26, 4, 4, 3), dtype=np.uint8)
            for time in range(26):
                visual[:, time] = time
            return {"visual": visual, "proprio": np.zeros((1, 26, 4))}, np.zeros((1, 26, 7))

    workspace = PlanWorkspace.__new__(PlanWorkspace)
    workspace.cfg_dict = {"seed": 0, "ood_corruption": corruption,
                          "env_kwargs_override": {"color": "Red"}}
    workspace.wm = type("World", (), {"encoder": type("Encoder", (),
                        {"requires_previous_frame": True})()})()
    workspace.env = FakeEnv()
    workspace.env_name = "pusht"
    workspace.eval_seed = [1]
    workspace.n_evals = 1
    workspace.frameskip = 5
    workspace.data_preprocessor = make_preprocessor()
    workspace.prepare_targets_from_segments(str(targets))
    np.testing.assert_array_equal(workspace.obs_0["visual_previous"], workspace.obs_0["visual"])
    assert np.all(workspace.obs_g["visual"] == 25)
    assert np.all(workspace.obs_g["visual_previous"] == 24)
    assert workspace.env.env_info[0]["color"] == "Red"
