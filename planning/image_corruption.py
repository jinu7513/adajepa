"""Online image corruption for OOD evaluation.

Supports: gaussian, salt_pepper, blur; legacy snp1/snp5 and dark are preserved.
Applied to uint8 HWC numpy arrays, compatible with observation dicts
from env.rollout().
"""

import numpy as np
import cv2


def apply_gaussian(frame, sigma, rng):
    """Sigma is in uint8 intensity units; noise is independent per channel."""
    return np.clip(frame.astype(np.float32) + rng.normal(0, sigma, frame.shape),
                   0, 255).astype(np.uint8)


def apply_canonical_salt_pepper(frame, probability, rng):
    """One exclusive draw per pixel: total selected fraction has expectation p."""
    draw = rng.random_sample(frame.shape[:2])
    out = frame.copy()
    out[draw < probability / 2] = 255
    out[(draw >= probability / 2) & (draw < probability)] = 0
    return out


def apply_salt_pepper(frame, density, rng):
    """Salt-and-pepper noise on a single HxWxC uint8 frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    n = int(density * h * w)
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 255
    ys = rng.randint(0, h, n); xs = rng.randint(0, w, n)
    out[ys, xs] = 0
    return out


def apply_blur(frame, sigma):
    """Gaussian blur with given sigma."""
    if sigma == 0:
        return frame.copy()
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_dark(frame, factor):
    """Darken by multiplying pixel values by factor < 1."""
    out = frame.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


# Registry: name -> (function, default_level)
CORRUPTIONS = {
    "gaussian": (apply_gaussian, 5.0),
    "salt_pepper": (apply_canonical_salt_pepper, 0.01),
    "blur":  (apply_blur, 2.0),
    "snp1":  (apply_salt_pepper, 0.01),
    "snp5":  (apply_salt_pepper, 0.05),
    "dark":  (apply_dark, 0.5),
}


def corrupt_frames(frames, corruption_name, level=None, seed=0):
    """Apply corruption to a batch of frames.

    Args:
        frames: np.ndarray, shape (..., H, W, C) uint8.
            Can be (H,W,C), (T,H,W,C), (B,T,H,W,C), etc.
        corruption_name: registry key (canonical or legacy), or None/"none".
        level: override default level. If None, use default.
        seed: RNG seed for stochastic corruptions (snp).

    Returns:
        Corrupted frames, same shape and dtype.
    """
    if corruption_name is None or corruption_name == "none":
        return frames

    if corruption_name not in CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{corruption_name}'. "
            f"Choose from: {list(CORRUPTIONS.keys())}"
        )

    func, default_level = CORRUPTIONS[corruption_name]
    if level is None:
        level = default_level
    if not np.isfinite(level) or level < 0:
        raise ValueError("Corruption strength must be finite and nonnegative")
    if corruption_name == "salt_pepper" and level > 1:
        raise ValueError("salt_pepper strength must lie in [0,1]")

    # seed=None -> truly random (system entropy), fresh noise each call.
    # seed=int -> reproducible noise for that seed.
    rng = np.random.RandomState(seed)
    orig_shape = frames.shape
    flat = frames.reshape(-1, *frames.shape[-3:])  # (N, H, W, C)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        if corruption_name.startswith("snp") or corruption_name in ("gaussian", "salt_pepper"):
            out[i] = func(flat[i], level, rng)
        else:
            out[i] = func(flat[i], level)
    return out.reshape(orig_shape)


def corrupt_obs_dict(obs, corruption_name, level=None, seed=0):
    """Apply corruption to the 'visual' key of an observation dict.

    Only modifies 'visual'; other keys (proprio, etc.) are unchanged.

    Args:
        obs: dict with "visual" key containing uint8 numpy array.
        corruption_name: corruption type or None/"none".
        level: optional level override.
        seed: RNG seed.

    Returns:
        New dict with corrupted visual, other keys shared.
    """
    if corruption_name is None or corruption_name == "none":
        return obs
    out = dict(obs)
    out["visual"] = corrupt_frames(obs["visual"], corruption_name, level, seed)
    return out
