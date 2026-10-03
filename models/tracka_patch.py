"""Use a frozen Track A scratch encoder as a patch-token world-model encoder."""

from pathlib import Path

import torch.nn as nn

from tracka.common import file_hash, load_checkpoint
from tracka.model import ScratchEncoder


class TrackAPatchEncoder(nn.Module):
    """Load one Track A checkpoint and expose its patch grid, not its CLS token.

    Inputs must be BCHW images at the checkpoint resolution normalized to [-1, 1].
    The world-model trainer owns freezing; this wrapper does not silently change
    ``requires_grad`` so that trainability is visible in its configuration.
    """

    def __init__(self, checkpoint_path, expected_step=None, expected_sha256=None):
        super().__init__()
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            raise ValueError("Track A checkpoint_path must be absolute (Hydra changes the working directory)")
        if not path.is_file():
            raise FileNotFoundError(f"Track A checkpoint not found: {path}")
        path = path.resolve()
        digest = file_hash(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Track A checkpoint SHA-256 mismatch: {path}")
        state = load_checkpoint(path)  # Trusted local training artifact; contains RNG state.
        if not isinstance(state, dict) or "encoder" not in state or "config" not in state:
            raise ValueError("Track A checkpoint must contain encoder weights and config")
        step = state.get("global_step")
        if expected_step is not None and step != expected_step:
            raise ValueError(f"Expected Track A step {expected_step}, got {step} from {path}")
        model_cfg = state["config"]["model"]
        self.backbone = ScratchEncoder(**model_cfg)
        self.backbone.load_state_dict(state["encoder"], strict=True)
        self.name = "tracka_patch"
        self.latent_ndim = 2
        self.emb_dim = self.backbone.emb_dim
        self.image_size = self.backbone.image_size
        self.patch_size = self.backbone.patch_size
        self.num_patches = self.backbone.num_patches
        self.source_checkpoint = str(path)
        self.source_sha256 = digest
        self.source_step = step
        self.source_manifest_sha256 = state.get("manifest_sha256")

    def forward(self, images, return_agg=False):
        tokens = self.backbone(images)  # CLS participates in attention but is not returned.
        return self.agg(tokens) if return_agg else tokens

    def agg(self, tokens):
        return self.backbone.agg(tokens)
