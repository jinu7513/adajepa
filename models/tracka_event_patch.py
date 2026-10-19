"""RGB/Event encoder exposing patch tokens to the PushT predictor."""

from pathlib import Path

import torch
from torch import nn
from einops import rearrange

from tracka.common import file_hash, load_checkpoint
from tracka.event_model import RGBEventEncoder


class TrackAEventPatchEncoder(nn.Module):
    """Encode each frame together with its actual preceding video frame."""

    requires_previous_frame = True

    def __init__(self, checkpoint_path=None, expected_step=None, expected_sha256=None,
                 model_config=None):
        super().__init__()
        if checkpoint_path is None:
            if model_config is None:
                raise ValueError("Scratch Event encoder requires model_config")
            if expected_step is not None or expected_sha256 is not None:
                raise ValueError("Scratch Event encoder cannot specify a checkpoint step or hash")
            self.backbone = RGBEventEncoder(**model_config)
            self.source_checkpoint = None
            self.source_sha256 = None
            self.source_step = None
            self.source_manifest_sha256 = None
            self.source_initialization = "random"
        else:
            if model_config is not None:
                raise ValueError("Pretrained Event encoder architecture comes from its checkpoint")
            self._load_pretrained(checkpoint_path, expected_step, expected_sha256)
            self.source_initialization = "pretrained"
        self.name = "tracka_event_patch"
        self.latent_ndim = 2
        self.emb_dim = self.backbone.emb_dim
        self.image_size = self.backbone.image_size
        self.patch_size = self.backbone.patch_size
        self.num_patches = self.backbone.num_patches

    def _load_pretrained(self, checkpoint_path, expected_step, expected_sha256):
        path = Path(checkpoint_path).expanduser()
        if not path.is_absolute():
            raise ValueError("Event checkpoint_path must be absolute")
        if not path.is_file():
            raise FileNotFoundError(f"Event checkpoint not found: {path}")
        path = path.resolve()
        digest = file_hash(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"Event checkpoint SHA-256 mismatch: {path}")
        state = load_checkpoint(path)  # Trusted local training artifact.
        if state.get("schema_version") != "tracka-event-v1":
            raise ValueError("Expected a tracka-event-v1 checkpoint")
        step = state.get("global_step")
        if expected_step is not None and step != expected_step:
            raise ValueError(f"Expected Event step {expected_step}, got {step} from {path}")
        self.backbone = RGBEventEncoder(**state["config"]["model"])
        prefix = "student."
        student = {key[len(prefix):]: value for key, value in state["model"].items()
                   if key.startswith(prefix)}
        self.backbone.load_state_dict(student, strict=True)
        self.source_checkpoint = str(path)
        self.source_sha256 = digest
        self.source_step = step
        self.source_manifest_sha256 = state.get("manifest_sha256")

    def forward(self, previous, current):
        return self.backbone(previous, current)["patch_tokens"]

    def forward_sequence(self, previous, current):
        if previous.shape != current.shape or current.ndim != 5 or current.shape[2] != 3:
            raise ValueError("Event predictor expects aligned [B,T,3,H,W] frame pairs")
        batch, time = current.shape[:2]
        patches = self.forward(rearrange(previous, "b t c h w -> (b t) c h w"),
                               rearrange(current, "b t c h w -> (b t) c h w"))
        return rearrange(patches, "(b t) p d -> b t p d", b=batch, t=time)

    def agg(self, tokens):
        return tokens.mean(1)
