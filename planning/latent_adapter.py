"""Online visual-latent adapter with a frozen encoder and predictor."""

import torch
from torch import nn
from torch.nn import functional as F

from .adajepa import AdaJEPATrainer


class VisualLatentAdapter(nn.Module):
    def __init__(self, dim, kind="mlp", width=32):
        super().__init__()
        if kind == "bias":
            self.bias = nn.Parameter(torch.zeros(dim))
            self.net = None
        elif kind == "mlp":
            self.register_parameter("bias", None)
            self.net = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, dim)
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            raise ValueError(f"Unknown visual adapter kind: {kind}")

    def forward(self, visual):
        return visual + (self.bias if self.net is None else self.net(visual))


class LatentAdapterTrainer(AdaJEPATrainer):
    """Use observed transitions to update only the post-encoder adapter."""

    def __init__(self, wm, kind, width, lr, steps, optimizer_name="adam", identity_weight=0.01):
        if wm.concat_dim not in (0, 1):
            raise ValueError("Unsupported world-model concat_dim")
        self.wm = wm
        self.device = next(wm.parameters()).device
        self.criterion = nn.MSELoss()
        self.lr = float(lr)
        self.steps = int(steps)
        self.identity_weight = float(identity_weight)
        if self.lr < 0 or self.steps < 0 or self.identity_weight < 0:
            raise ValueError("Adapter lr, steps and identity_weight must be nonnegative")
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unknown adapter optimizer: {optimizer_name}")
        self.optimizer_type = optimizers[optimizer_name.lower()]
        self.adapter = VisualLatentAdapter(wm.encoder.emb_dim, kind, int(width)).to(self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad_(False)
        wm.visual_adapter = self.adapter
        wm.visual_adapter_enabled = True
        self._initial = [param.detach().clone() for param in self.adapter.parameters()]

    @torch.no_grad()
    def reset(self):
        for param, initial in zip(self.adapter.parameters(), self._initial):
            param.copy_(initial)

    def _adapt_encoded(self, z):
        if self.wm.concat_dim == 0:
            return torch.cat([self.adapter(z[:, :, :-2]), z[:, :, -2:]], dim=2)
        dim = self.wm.encoder.emb_dim
        return torch.cat([self.adapter(z[..., :dim]), z[..., dim:]], dim=-1)

    def _base_segments(self, obs_seqs, act_seqs, merge):
        if merge and len(obs_seqs) > 1:
            obs_seqs, act_seqs = self._merge_segments(obs_seqs, act_seqs)
        segments = [self._prepare_segment(o, a) for o, a in zip(obs_seqs, act_seqs)]
        self.wm.visual_adapter_enabled = False
        try:
            with torch.no_grad():
                return [self.wm.encode(o, a).detach() for o, a in segments]
        finally:
            self.wm.visual_adapter_enabled = True

    def finetune(self, obs_seqs, act_seqs, merge=True):
        if not obs_seqs or self.steps == 0 or self.lr == 0:
            return []
        base = self._base_segments(obs_seqs, act_seqs, merge)
        for param in self.adapter.parameters():
            param.requires_grad_(True)
        self.adapter.train()
        optimizer = self.optimizer_type(self.adapter.parameters(), lr=self.lr)
        losses = []
        try:
            for _ in range(self.steps):
                optimizer.zero_grad()
                adapted = [self._adapt_encoded(z) for z in base]
                pred_loss = torch.stack([
                    self._prediction_loss(z, detach_src=False, detach_tgt=True)
                    for z in adapted
                ]).mean()
                identity_loss = torch.stack([
                    F.mse_loss(self.adapter(z[..., :self.wm.encoder.emb_dim]),
                               z[..., :self.wm.encoder.emb_dim])
                    if self.wm.concat_dim == 1 else
                    F.mse_loss(self.adapter(z[:, :, :-2]), z[:, :, :-2])
                    for z in base
                ]).mean()
                (pred_loss + self.identity_weight * identity_loss).backward()
                optimizer.step()
                losses.append(float(pred_loss.detach()))
        finally:
            self.adapter.eval()
            for param in self.adapter.parameters():
                param.requires_grad_(False)
        return losses

    @torch.no_grad()
    def score_segments(self, obs_seqs, act_seqs):
        base = self._base_segments(obs_seqs, act_seqs, merge=False)
        return [float(self._prediction_loss(self._adapt_encoded(z))) for z in base]
