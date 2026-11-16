"""Training-only inverse action diagnostic from real endpoint visual latents."""

import torch
from torch import nn


class InverseActionHead(nn.Module):
    def __init__(self, visual_dim, action_dim, horizons=(1, 2, 3), hidden_dim=256):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("Inverse horizons must be positive")
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(nn.Linear(2 * visual_dim, hidden_dim), nn.ReLU(),
                                  nn.Linear(hidden_dim, h * action_dim))
            for h in self.horizons
        })

    def forward(self, visual_latents, action_chunks):
        # visual_latents: [batch, observed_frames, patches, dim]
        # action_chunks: [batch, observed_frames, actions_per_chunk * action_dim]
        if visual_latents.ndim != 4 or action_chunks.ndim != 3:
            raise ValueError("Expected observed visual latents and action chunks")
        if visual_latents.shape[1] < max(self.horizons) + 1:
            raise ValueError("Actual endpoint frame missing for an inverse horizon")
        start = visual_latents[:, 0].mean(dim=1)
        losses = {}
        for horizon in self.horizons:
            endpoint = visual_latents[:, horizon].mean(dim=1)
            predicted = self.heads[str(horizon)](torch.cat([start, endpoint], dim=-1))
            actual = action_chunks[:, :horizon].reshape(action_chunks.shape[0], -1).detach()
            if predicted.shape != actual.shape:
                raise ValueError("Action endpoint indexing or dimension mismatch")
            # Fixed unit-variance Gaussian NLL, excluding the constant term.
            losses[horizon] = 0.5 * torch.mean((predicted - actual) ** 2)
        return sum(losses.values()) / len(losses), losses
