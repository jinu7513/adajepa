"""Frame-derived event *proxies*, not asynchronous event-camera measurements."""

import math

import torch
import torch.nn.functional as F


EVENT_CHANNELS = {"hard_signed": 1, "hard_two_channel": 2,
                  "delta_log": 1, "soft_clipped": 1}


def event_from_rgb(previous, current, *, mode="hard_two_channel", threshold=0.2,
                   eps=1e-3, patch_size=16, shift_previous=False):
    """Return event [B,C,H,W] and diagnostics from RGB tensors in [-1,1].

    A fixed one-pixel shift is an optional ablation and never wraps across the
    image boundary. Density is threshold-crossing density for every mode.
    """
    if mode not in EVENT_CHANNELS:
        raise ValueError(f"Unknown event mode: {mode}")
    if previous.shape != current.shape or previous.ndim != 4 or previous.shape[1] != 3:
        raise ValueError("Event inputs must be equal-sized [B,3,H,W] frames")
    if patch_size < 1 or previous.shape[-2] % patch_size or previous.shape[-1] % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    if not math.isfinite(threshold) or not math.isfinite(eps) or threshold <= 0 or eps <= 0:
        raise ValueError("Event threshold and eps must be positive")
    if shift_previous:
        previous = F.pad(previous, (1, 0, 0, 0), mode="replicate")[..., :-1]
    weights = previous.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)

    def log_luminance(rgb):
        rgb01 = ((rgb.float() + 1) * .5).clamp(0, 1)
        return ((rgb01 * weights).sum(1, keepdim=True) + eps).log()

    delta = log_luminance(current) - log_luminance(previous)
    positive = delta >= threshold
    negative = delta <= -threshold
    if mode == "hard_signed":
        event = positive.to(delta.dtype) - negative.to(delta.dtype)
    elif mode == "hard_two_channel":
        event = torch.cat((positive.to(delta.dtype), negative.to(delta.dtype)), dim=1)
    elif mode == "delta_log":
        event = delta
    else:
        event = (delta / threshold).clamp(-1, 1)
    active = (positive | negative).to(delta.dtype)
    density = F.avg_pool2d(active, patch_size, stride=patch_size).flatten(2).transpose(1, 2)
    diagnostics = {"event_density": active.mean(),
                   "patch_density": density, "mean_abs_delta_log": delta.abs().mean()}
    return event, diagnostics
