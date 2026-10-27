"""Observed-transition support conditioning for the existing PushT predictor.

This module deliberately leaves TransitionFiLMPredictor (the earlier C arm)
unchanged. Support is episode-local and never updated from imagined rollouts.
"""

import copy

import torch
from torch import nn
from einops import rearrange

from .vit import ViTPredictor
from .visual_world_model import VWorldModel


class TransitionFeatureEncoder(nn.Module):
    def __init__(self, visual_dim, action_dim, context_dim=128, mode="delta"):
        super().__init__()
        if mode not in ("delta", "residual"):
            raise ValueError("mode must be 'delta' or 'residual'")
        self.mode = mode
        input_dim = 2 * visual_dim + action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, context_dim),
            nn.GELU(), nn.Linear(context_dim, context_dim),
        )

    def forward(self, previous, action, following, reference_prediction=None):
        if previous.shape != following.shape or previous.shape[:-1] != action.shape[:-1]:
            raise ValueError("Support visual/action dimensions do not align")
        if self.mode == "residual":
            if reference_prediction is None or reference_prediction.shape != following.shape:
                raise ValueError("Residual support requires matching frozen-reference predictions")
            displacement = following - reference_prediction.detach()
        else:
            displacement = following - previous
        return self.net(torch.cat([previous, action, displacement], dim=-1))


class SupportAggregator(nn.Module):
    def __init__(self, context_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, context_dim),
            nn.GELU(), nn.Linear(context_dim, context_dim),
        )

    def forward(self, encoded, mask):
        if encoded.ndim != 3 or mask.shape != encoded.shape[:2]:
            raise ValueError("Support must be [batch, K, channels] with a [batch, K] mask")
        if encoded.shape[1] == 0:
            return encoded.new_zeros(encoded.shape[0], encoded.shape[-1]), mask.new_zeros(mask.shape[0])
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        count = weights.sum(dim=1)
        pooled = (encoded * weights).sum(dim=1) / count.clamp_min(1)
        valid = mask.any(dim=1)
        return self.net(pooled), valid


class SupportFiLMPredictor(ViTPredictor):
    """Predictor with global context inferred from up to K real transitions."""

    uses_support_film = True

    def __init__(self, *, visual_dim, action_dim, context_dim=128,
                 feature_mode="delta", film_layers=2, affine_type="mlp", **kwargs):
        super().__init__(**kwargs)
        if affine_type not in ("linear", "mlp"):
            raise ValueError("affine_type must be 'linear' or 'mlp'")
        if not 1 <= film_layers <= len(self.transformer.layers):
            raise ValueError("film_layers must select at least one existing layer")
        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.feature_mode = feature_mode
        self.feature_encoder = TransitionFeatureEncoder(
            self.visual_dim, self.action_dim, context_dim, feature_mode
        )
        self.support_aggregator = SupportAggregator(context_dim)
        self.film = nn.ModuleDict()
        dim = int(kwargs["dim"])
        for layer_idx in range(len(self.transformer.layers) - film_layers,
                               len(self.transformer.layers)):
            if affine_type == "linear":
                projection = nn.Linear(context_dim, 2 * dim)
                last = projection
            else:
                projection = nn.Sequential(
                    nn.Linear(context_dim, context_dim), nn.GELU(),
                    nn.Linear(context_dim, 2 * dim),
                )
                last = projection[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            self.film[str(layer_idx)] = projection
        self.reference_predictor = None

    def attach_reference(self, predictor):
        if self.feature_mode != "residual":
            raise ValueError("A frozen reference is only used in residual mode")
        self.reference_predictor = copy.deepcopy(predictor)
        for parameter in self.reference_predictor.parameters():
            parameter.requires_grad_(False)
        self.reference_predictor.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.reference_predictor is not None:
            self.reference_predictor.eval()
        return self

    def context_from_support(self, previous, action, following, mask,
                             reference_prediction=None):
        if previous.ndim != 3 or action.ndim != 3 or following.ndim != 3:
            raise ValueError("Support tensors must have [batch, K, channels] shape")
        if mask.shape != previous.shape[:2]:
            raise ValueError("Support mask shape differs from support tensors")
        if previous.shape[-1] != self.visual_dim or action.shape[-1] != self.action_dim:
            raise ValueError("Support feature width does not match the predictor")
        if previous.shape[1] == 0:
            return previous.new_zeros(previous.shape[0], self.context_dim), mask.new_zeros(mask.shape[0])
        encoded = self.feature_encoder(previous, action, following, reference_prediction)
        return self.support_aggregator(encoded, mask)

    def forward(self, x, context=None, context_valid=None):
        batch, tokens, dim = x.shape
        if context is None:
            context = x.new_zeros(batch, self.context_dim)
        if context_valid is None:
            context_valid = torch.zeros(batch, dtype=torch.bool, device=x.device)
        if context.shape != (batch, self.context_dim) or context_valid.shape != (batch,):
            raise ValueError("Expected one context and validity bit per batch item")
        x = self.dropout(x + self.pos_embedding[:, :tokens])
        valid = context_valid.to(x.dtype).view(batch, 1, 1)
        for layer_idx, (attention, feedforward) in enumerate(self.transformer.layers):
            x = x + attention(x)
            projection = self.film[str(layer_idx)] if str(layer_idx) in self.film else None
            if projection is None:
                x = x + feedforward(x)
            else:
                gamma, beta = projection(context).chunk(2, dim=-1)
                normalized = feedforward.net[0](x)
                normalized = normalized * (1 + gamma.unsqueeze(1) * valid) + beta.unsqueeze(1) * valid
                for layer in list(feedforward.net)[1:]:
                    normalized = layer(normalized)
                x = x + normalized
        return self.transformer.norm(x)


class SupportVWorldModel(VWorldModel):
    """Drop-in planning model with an episode-local completed-transition buffer."""

    def __init__(self, *args, support_k=8, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.predictor, "uses_support_film", False):
            raise TypeError("SupportVWorldModel requires SupportFiLMPredictor")
        self.max_support_k = 8
        self.set_active_support_k(support_k)
        self.clear_observed_transition()

    def set_active_support_k(self, k):
        if int(k) not in (0, 2, 4, 8):
            raise ValueError("Supported K values are 0, 2, 4 and 8")
        self.support_k = int(k)

    def get_observed_support_state(self):
        return list(self._support_entries), list(self._observed_z_history)

    def restore_observed_support_state(self, state):
        self._support_entries, self._observed_z_history = state

    def clear_observed_transition(self):
        self._support_entries = []
        self._observed_z_history = []
        self._active_context = None

    def _reference_prediction(self, history):
        reference = self.predictor.reference_predictor
        if reference is None:
            raise ValueError("Residual mode requires the frozen reference predictor")
        frames = list(history[-self.num_hist:])
        while len(frames) < self.num_hist:
            frames.insert(0, frames[0])
        source = rearrange(torch.stack(frames, dim=1), "b t p d -> b (t p) d")
        with torch.no_grad():
            prediction = reference(source)
        patches = history[-1].shape[1]
        return prediction[:, -patches:, :self.predictor.visual_dim].mean(dim=1).detach()

    def _context_from_entries(self, entries, batch, device, dtype):
        selected = entries[-self.support_k:] if self.support_k else []
        if not selected:
            empty_vis = torch.empty(batch, 0, self.predictor.visual_dim, device=device, dtype=dtype)
            empty_act = torch.empty(batch, 0, self.predictor.action_dim, device=device, dtype=dtype)
            return self.predictor.context_from_support(
                empty_vis, empty_act, empty_vis, torch.zeros(batch, 0, dtype=torch.bool, device=device)
            )
        previous = torch.stack([entry[0] for entry in selected], dim=1)
        actions = torch.stack([entry[1] for entry in selected], dim=1)
        following = torch.stack([entry[2] for entry in selected], dim=1)
        references = (torch.stack([entry[3] for entry in selected], dim=1)
                      if self.predictor.feature_mode == "residual" else None)
        if previous.shape[0] != batch:
            raise ValueError("Observed support batch differs from MPC candidate batch")
        mask = torch.ones(batch, len(selected), dtype=torch.bool, device=device)
        return self.predictor.context_from_support(previous, actions, following, mask, references)

    def support_context_from_sequence(self, z, support_k=None):
        """Training: query uses final input frame, never final target frame."""
        if z.ndim != 4 or z.shape[1] < 4:
            raise ValueError("Expected [batch, frames>=4, patches, channels]")
        query_index = z.shape[1] - 2
        k = self.support_k if support_k is None else int(support_k)
        if k not in (0, 2, 4, 8) or k > query_index:
            raise ValueError("K exceeds completed support before the query")
        batch = z.shape[0]
        if k == 0:
            return self._context_from_entries([], batch, z.device, z.dtype)
        visual = z[..., :self.predictor.visual_dim].mean(dim=2)
        action = z[:, :, 0, -self.action_dim:]
        start = query_index - k
        previous = visual[:, start:query_index]
        following = visual[:, start + 1:query_index + 1]
        reference = None
        if self.predictor.feature_mode == "residual":
            reference = torch.stack([
                self._reference_prediction([z[:, j] for j in range(i + 1)])
                for i in range(start, query_index)
            ], dim=1)
        mask = torch.ones(batch, k, dtype=torch.bool, device=z.device)
        return self.predictor.context_from_support(previous, action[:, start:query_index],
                                                   following, mask, reference)

    @torch.no_grad()
    def observe_rollout(self, observations, actions):
        """Append only executed transitions with T+1 real observations and T actions."""
        if actions.ndim != 3 or observations["visual"].shape[1] != actions.shape[1] + 1:
            raise ValueError("Expected T+1 actual observations aligned to T executed chunks")
        if self.concat_dim != 1:
            raise ValueError("Support conditioning currently requires concat_dim=1")
        # Bound VRAM: encode each newly observed frame once, not an entire
        # 50-episode MPC batch of all frames twice.
        encoded_parts = [self.encode_obs({key: value[:, index:index + 1]
                                          for key, value in observations.items()})
                         for index in range(actions.shape[1] + 1)]
        visual_tokens = torch.cat([part["visual"] for part in encoded_parts], dim=1)
        proprio = torch.cat([part["proprio"] for part in encoded_parts[:-1]], dim=1)
        visual = visual_tokens.mean(dim=2)
        action = self.encode_act(actions)
        if self.predictor.feature_mode == "residual":
            patches = visual_tokens.shape[2]
            proprio_tiled = proprio.unsqueeze(2).expand(-1, -1, patches, -1)
            action_tiled = action.unsqueeze(2).expand(-1, -1, patches, -1)
            source_z = torch.cat([
                visual_tokens[:, :-1],
                proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat),
                action_tiled.repeat(1, 1, 1, self.num_action_repeat),
            ], dim=-1)
        for index in range(actions.shape[1]):
            reference = None
            if self.predictor.feature_mode == "residual":
                self._observed_z_history.append(source_z[:, index].detach())
                reference = self._reference_prediction(self._observed_z_history)
            self._support_entries.append((visual[:, index].detach(), action[:, index].detach(),
                                          visual[:, index + 1].detach(), reference))
            self._support_entries = self._support_entries[-self.max_support_k:]
            self._observed_z_history = self._observed_z_history[-self.num_hist:]

    def predict(self, z, *, support_context=None, context_valid=None):
        if support_context is None:
            if self._active_context is not None:
                support_context, context_valid = self._active_context
            else:
                support_context, context_valid = self._context_from_entries(
                    self._support_entries, z.shape[0], z.device, z.dtype
                )
        if context_valid is None:
            raise ValueError("Explicit support context requires a validity mask")
        flat = rearrange(z, "b t p d -> b (t p) d")
        return rearrange(self.predictor(flat, support_context, context_valid),
                         "b (t p) d -> b t p d", t=z.shape[1])

    def rollout(self, obs_0, act, *, known_observed_actions=False):
        # One context per MPC decision; imagined predictions never become support.
        if self._active_context is not None:
            raise RuntimeError("Nested support-conditioned rollouts are unsupported")
        self._active_context = self._context_from_entries(
            self._support_entries, obs_0["visual"].shape[0], act.device, act.dtype
        )
        try:
            return super().rollout(obs_0, act, known_observed_actions=known_observed_actions)
        finally:
            self._active_context = None
