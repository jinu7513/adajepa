from types import SimpleNamespace

import torch
from torch import nn

from planning.mpc import MPCPlanner


class FakeWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("marker", nn.Parameter(torch.zeros(1)))
        self.predictor = SimpleNamespace(uses_support_film=True)
        self.seen = None

    def observe_rollout(self, observations, actions):
        self.seen = (observations, actions)


def test_mpc_feedback_uses_only_newly_executed_real_frames():
    planner = object.__new__(MPCPlanner)
    planner.wm = FakeWorldModel()
    planner.evaluator = SimpleNamespace(frameskip=5)
    planner.preprocessor = SimpleNamespace(transform_obs=lambda value: value)
    planner.iter = 1
    taken_actions = torch.randn(1, 2, 10)
    frames = torch.arange(21).float().reshape(1, 21, 1)
    observations = {"visual": frames, "proprio": frames.clone()}
    planner._post_env_feedback(taken_actions, observations)
    selected, actual_actions = planner.wm.seen
    assert selected["visual"].flatten().tolist() == [10.0, 15.0, 20.0]
    torch.testing.assert_close(actual_actions, taken_actions)
