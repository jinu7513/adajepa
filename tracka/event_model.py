"""Reconstruction-free dual-stream RGB/Event EMA representation learner."""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from .event import EVENT_CHANNELS, event_from_rgb
from .model import SIGRegProjector, patchify, sigreg_loss


def representation_distance(prediction, target, mode):
    if prediction.shape != target.shape:
        raise ValueError("Representation target/prediction shape mismatch")
    if mode == "cosine":
        return (1 - F.cosine_similarity(prediction, target, dim=-1)).mean()
    if mode == "normalized_mse":
        return (F.normalize(prediction, dim=-1) - F.normalize(target, dim=-1)).square().sum(-1).mean()
    if mode == "mse":
        return (prediction - target).square().mean()
    raise ValueError(f"Unknown representation distance: {mode}")


def vicreg_guard(features):
    """Optional one-branch variance/covariance guard, not a default objective."""
    if features.shape[0] < 2:
        return features.sum() * 0
    centered = features.float() - features.float().mean(0, keepdim=True)
    std = (centered.square().mean(0) + 1e-4).sqrt()
    variance = F.relu(1 - std).mean()
    covariance = centered.T @ centered / (features.shape[0] - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    return variance + off_diagonal.square().sum() / features.shape[-1]


def mlp(dim):
    return nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))


class RGBEventEncoder(nn.Module):
    """Return patch [B,P,D] and global [B,D] from a previous/current RGB pair."""

    def __init__(self, *, image_size=224, patch_size=16, embed_dim=384, heads=6,
                 rgb_depth=1, event_depth=1, shared_depth=5,
                 event_mode="hard_two_channel", event_threshold=0.2, event_eps=1e-3,
                 shift_previous=False, fusion="fixed_sum", global_representation="cls",
                 rgb_input_mode="current"):
        super().__init__()
        if (patch_size < 1 or embed_dim < 1 or heads < 1 or image_size < patch_size
                or image_size % patch_size or embed_dim % heads
                or min(rgb_depth, event_depth, shared_depth) < 0):
            raise ValueError("Invalid RGB/Event dimensions or block depths")
        if fusion not in ("rgb_only", "event_only", "fixed_sum", "gated_sum"):
            raise ValueError("Unknown fusion mode")
        if global_representation not in ("cls", "mean_pool"):
            raise ValueError("Unknown global representation")
        if rgb_input_mode not in ("current", "pair") or (rgb_input_mode == "pair" and fusion != "rgb_only"):
            raise ValueError("RGB pair input is supported only for the rgb_only control")
        if (event_mode not in EVENT_CHANNELS or not math.isfinite(event_threshold)
                or not math.isfinite(event_eps) or event_threshold <= 0 or event_eps <= 0):
            raise ValueError("Invalid event transform configuration")
        self.image_size, self.patch_size, self.emb_dim = image_size, patch_size, embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.event_mode, self.event_threshold, self.event_eps = event_mode, event_threshold, event_eps
        self.shift_previous = shift_previous
        self.fusion, self.global_representation = fusion, global_representation
        self.rgb_input_mode = rgb_input_mode
        self.rgb_embed = nn.Linear((6 if rgb_input_mode == "pair" else 3) * patch_size ** 2, embed_dim)
        self.event_embed = nn.Linear(EVENT_CHANNELS[event_mode] * patch_size ** 2, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.num_patches, embed_dim))

        def blocks(n):
            return nn.ModuleList([nn.TransformerEncoderLayer(embed_dim, heads, 4 * embed_dim,
                                  dropout=0., activation="gelu", batch_first=True,
                                  norm_first=True) for _ in range(n)])

        self.rgb_blocks, self.event_blocks = blocks(rgb_depth), blocks(event_depth)
        self.shared_blocks = blocks(shared_depth)
        self.rgb_norm, self.event_norm = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if global_representation == "cls" else None
        self.gate = None
        if fusion == "gated_sum":
            self.gate = nn.Sequential(nn.Linear(2 * embed_dim + 1, embed_dim // 2),
                                      nn.GELU(), nn.Linear(embed_dim // 2, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, -2.0)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, previous, current):
        if previous.shape != current.shape or current.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("RGB/Event encoder expects aligned, equal-resolution consecutive frames")
        event, event_info = event_from_rgb(previous, current, mode=self.event_mode,
                                          threshold=self.event_threshold, eps=self.event_eps,
                                          patch_size=self.patch_size,
                                          shift_previous=self.shift_previous)
        rgb = None
        if self.fusion != "event_only":
            rgb_input = torch.cat((previous, current), 1) if self.rgb_input_mode == "pair" else current
            rgb = self.rgb_embed(patchify(rgb_input, self.patch_size)) + self.pos_embed
            for block in self.rgb_blocks:
                rgb = block(rgb)
            rgb = self.rgb_norm(rgb)
        event_tokens = None
        if self.fusion != "rgb_only":
            event_tokens = self.event_embed(patchify(event, self.patch_size)) + self.pos_embed
            for block in self.event_blocks:
                event_tokens = block(event_tokens)
            event_tokens = self.event_norm(event_tokens)
        if ((rgb is not None and rgb.shape[1] != self.num_patches)
                or (event_tokens is not None and event_tokens.shape[1] != self.num_patches)
                or event_info["patch_density"].shape[1] != self.num_patches
                or (rgb is not None and event_tokens is not None and rgb.shape != event_tokens.shape)):
            raise ValueError("RGB/Event patch grids or positional indexing differ")
        gate_values = None
        if self.fusion == "rgb_only":
            fused = rgb
        elif self.fusion == "event_only":
            fused = event_tokens
        elif self.fusion == "fixed_sum":
            fused = rgb + event_tokens
        else:
            gate_values = torch.sigmoid(self.gate(torch.cat(
                (rgb, event_tokens, event_info["patch_density"]), dim=-1)))
            fused = rgb + gate_values * event_tokens
        if self.cls_token is not None:
            fused = torch.cat((self.cls_token.expand(len(fused), -1, -1), fused), 1)
        for block in self.shared_blocks:
            fused = block(fused)
        fused = self.final_norm(fused)
        patches = fused[:, 1:] if self.cls_token is not None else fused
        global_token = fused[:, 0] if self.cls_token is not None else patches.mean(1)
        return {"patch_tokens": patches, "global_token": global_token,
                "event_density": event_info["event_density"],
                "patch_density": event_info["patch_density"],
                "mean_abs_delta_log": event_info["mean_abs_delta_log"],
                "gate": gate_values}


class EventTeacherStudent(nn.Module):
    def __init__(self, model_cfg, loss_cfg):
        super().__init__()
        self.student = RGBEventEncoder(**model_cfg)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.q = mlp(self.student.emb_dim)
        self.F = mlp(self.student.emb_dim)
        self.loss_cfg = dict(loss_cfg)
        if loss_cfg["distance"] not in ("cosine", "normalized_mse", "mse"):
            raise ValueError("Unsupported representation distance")
        if loss_cfg["anti_collapse"] not in ("none", "sigreg", "vicreg"):
            raise ValueError("Unsupported anti-collapse mode")
        if any(not math.isfinite(loss_cfg[key]) or loss_cfg[key] < 0
               for key in ("lambda_rob", "lambda_temp", "lambda_ac")):
            raise ValueError("Loss weights must be nonnegative")
        if loss_cfg["anti_collapse"] == "none" and loss_cfg["lambda_ac"]:
            raise ValueError("lambda_ac requires an anti-collapse mode")
        self.sigreg_projector = None
        if loss_cfg["anti_collapse"] == "sigreg" and loss_cfg["lambda_ac"]:
            self.sigreg_projector = SIGRegProjector(self.student.emb_dim,
                                                     loss_cfg["sigreg_hidden_dim"],
                                                     loss_cfg["sigreg_dim"])

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum):
        if not 0 <= momentum < 1:
            raise ValueError("EMA momentum must be in [0,1)")
        for target, source in zip(self.teacher.parameters(), self.student.parameters()):
            target.lerp_(source, 1 - momentum)
        for target, source in zip(self.teacher.buffers(), self.student.buffers()):
            target.copy_(source)

    def objective(self, batch):
        clean_prev, clean_now, clean_next = (batch["clean_" + key]
                                              for key in ("previous", "current", "next"))
        shift_prev, shift_now = batch["shift_previous"], batch["shift_current"]
        student = self.student(shift_prev, shift_now)
        with torch.no_grad():
            teacher_now = self.teacher(clean_prev, clean_now)
            teacher_next = self.teacher(clean_now, clean_next)
        patch_prediction = self.q(student["patch_tokens"])
        next_prediction = self.F(student["global_token"])
        mode = self.loss_cfg["distance"]
        robust = representation_distance(patch_prediction, teacher_now["patch_tokens"], mode)
        temporal = representation_distance(next_prediction, teacher_next["global_token"], mode)
        total = self.loss_cfg["lambda_rob"] * robust + self.loss_cfg["lambda_temp"] * temporal
        anti = total.new_zeros(())
        if self.loss_cfg["lambda_ac"]:
            if self.loss_cfg["anti_collapse"] == "sigreg":
                projected = self.sigreg_projector(student["global_token"])
                anti = sigreg_loss(projected, 1., self.loss_cfg["sigreg_directions"],
                                   self.loss_cfg["sigreg_knots"])
            else:
                anti = vicreg_guard(student["global_token"])
            total = total + self.loss_cfg["lambda_ac"] * anti
        with torch.no_grad():
            teacher_prediction = self.F(teacher_now["global_token"])
            teacher_temporal = representation_distance(teacher_prediction,
                                                        teacher_next["global_token"], mode)
            persistence = representation_distance(teacher_now["global_token"],
                                                   teacher_next["global_token"], mode)
            patch_std = student["patch_tokens"].float().std(dim=0, unbiased=False).mean()
            global_std = student["global_token"].float().std(dim=0, unbiased=False).mean()
            teacher_student = (student["global_token"] - teacher_now["global_token"]).square().mean().sqrt()
            diagnostics = {"loss/robust": robust, "loss/temporal": temporal,
                           "loss/anti_collapse": anti, "loss/total": total,
                           "temporal/teacher_input_error": teacher_temporal,
                           "temporal/persistence_error": persistence,
                           "latent/global_std_raw": global_std,
                           "latent/patch_std_raw": patch_std,
                           "latent/global_norm": student["global_token"].norm(dim=-1).mean(),
                           "latent/patch_norm": student["patch_tokens"].norm(dim=-1).mean(),
                           "latent/student_shift_teacher_clean_distance": teacher_student,
                           "event/density_clean": teacher_now["event_density"],
                           "event/density_shift": student["event_density"],
                           "event/mean_abs_delta_log_clean": teacher_now["mean_abs_delta_log"],
                           "event/mean_abs_delta_log_shift": student["mean_abs_delta_log"],
                           "event/patch_density_mean": student["patch_density"].mean()}
            diagnostics["event/patch_density_std"] = student["patch_density"].std(unbiased=False)
            diagnostics["event/patch_density_max"] = student["patch_density"].max()
            if student["gate"] is not None:
                diagnostics["gate/mean"] = student["gate"].mean()
                diagnostics["gate/std"] = student["gate"].std(unbiased=False)
        return total, diagnostics
