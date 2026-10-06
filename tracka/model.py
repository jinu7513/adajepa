"""Spatial bidirectional ViT and shared nuisance-conditioned MAE decoder."""
import math

import torch
from torch import nn


def patchify(images, patch_size):
    b, c, h, w = images.shape
    if h != w or h % patch_size:
        raise ValueError("Expected square images divisible by patch_size")
    p = patch_size
    return images.reshape(b, c, h // p, p, w // p, p).permute(
        0, 2, 4, 3, 5, 1).reshape(b, (h // p) ** 2, p * p * c)


def unpatchify(patches, patch_size):
    b, n, d = patches.shape
    side = int(n ** 0.5)
    p = patch_size
    if side * side != n or d != p * p * 3:
        raise ValueError("Invalid RGB patch grid")
    return patches.reshape(b, side, side, p, p, 3).permute(
        0, 5, 1, 3, 2, 4).reshape(b, 3, side * p, side * p)


def sample_mask(batch, patches, ratio, device, generator=None):
    if not 0 < ratio < 1:
        raise ValueError("mask ratio must be strictly between 0 and 1")
    keep = int(patches * (1 - ratio))
    if not 0 < keep < patches:
        raise ValueError("Mask must leave both visible and hidden patches")
    order = torch.rand(batch, patches, device=device, generator=generator).argsort(1)
    restore = order.argsort(1)
    mask = torch.ones(batch, patches, device=device)
    mask[:, :keep] = 0
    return order[:, :keep], restore, mask.gather(1, restore)


def resolved_mask_ratios(mask):
    """Resolve optional per-objective ratios, including legacy ratio-only configs."""
    base = mask["ratio"]
    render = mask.get("render_ratio")
    corruption = mask.get("corruption_ratio")
    ratios = {"clean_mae": base,
              "render_only": base if render is None else render,
              "corruption_only": base if corruption is None else corruption}
    for mode, ratio in ratios.items():
        if ratio is None or not 0 < ratio < 1:
            raise ValueError(f"mask ratio for {mode} must be strictly between 0 and 1")
    return ratios


def resolved_variance_config(loss):
    """Fill regularizer defaults for older checkpoints and validate ablations."""
    config = dict(loss)
    legacy_sigreg = "sigreg_space" not in config and config.get("lambda_sigreg", 0) > 0
    config.setdefault("lambda_variance", 0.0)
    config.setdefault("variance_target_std", 0.1)
    config.setdefault("variance_feature", "patch_mean")
    config.setdefault("lambda_sigreg", 0.0)
    config.setdefault("sigreg_space", "raw" if legacy_sigreg else "projector")
    config.setdefault("sigreg_target_std", 0.1 if legacy_sigreg else 1.0)
    config.setdefault("sigreg_feature", "both" if legacy_sigreg else "cls")
    config.setdefault("sigreg_directions", 256)
    config.setdefault("sigreg_knots", 17)
    config.setdefault("sigreg_projector_hidden_dim", 512)
    config.setdefault("sigreg_projector_dim", 128)
    config.setdefault("sigreg_patch_samples", 8)
    if not math.isfinite(config["lambda_variance"]) or config["lambda_variance"] < 0:
        raise ValueError("loss.lambda_variance must be finite and nonnegative")
    if not math.isfinite(config["variance_target_std"]) or config["variance_target_std"] <= 0:
        raise ValueError("loss.variance_target_std must be finite and positive")
    if config["variance_feature"] not in ("patch_mean", "cls", "both"):
        raise ValueError("loss.variance_feature must be patch_mean, cls, or both")
    if not math.isfinite(config["lambda_sigreg"]) or config["lambda_sigreg"] < 0:
        raise ValueError("loss.lambda_sigreg must be finite and nonnegative")
    if not math.isfinite(config["sigreg_target_std"]) or config["sigreg_target_std"] <= 0:
        raise ValueError("loss.sigreg_target_std must be finite and positive")
    if config["sigreg_space"] not in ("raw", "projector"):
        raise ValueError("loss.sigreg_space must be raw or projector")
    allowed_features = (("patch_mean", "cls", "both") if config["sigreg_space"] == "raw"
                        else ("cls", "patch_mean", "patch_tokens", "both", "both_spatial"))
    if config["sigreg_feature"] not in allowed_features:
        raise ValueError("loss.sigreg_feature is invalid for the selected sigreg_space")
    if config["lambda_sigreg"] and config["sigreg_space"] == "projector" and config["sigreg_target_std"] != 1.0:
        raise ValueError("Projector SIGReg requires loss.sigreg_target_std=1.0 for N(0,I)")
    if type(config["sigreg_directions"]) is not int or config["sigreg_directions"] < 1:
        raise ValueError("loss.sigreg_directions must be a positive integer")
    if type(config["sigreg_knots"]) is not int or config["sigreg_knots"] < 2:
        raise ValueError("loss.sigreg_knots must be an integer >= 2")
    for key in ("sigreg_projector_hidden_dim", "sigreg_projector_dim", "sigreg_patch_samples"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("loss." + key + " must be a positive integer")
    if config["lambda_variance"] and config["lambda_sigreg"]:
        raise ValueError("Choose either loss.lambda_variance or loss.lambda_sigreg, not both")
    return config


def comparable_loss_config(loss):
    """Only active loss settings constrain exact checkpoint continuation."""
    config = resolved_variance_config(loss)
    if not config["lambda_variance"]:
        for key in ("variance_target_std", "variance_feature"):
            config.pop(key)
    if not config["lambda_sigreg"]:
        for key in tuple(config):
            if key.startswith("sigreg_"):
                config.pop(key)
    elif config["sigreg_space"] == "raw":
        for key in ("sigreg_projector_hidden_dim", "sigreg_projector_dim", "sigreg_patch_samples"):
            config.pop(key)
    return config


def variance_hinge(features, target_std):
    """VICReg-style per-dimension standard-deviation floor across distinct images."""
    if features.ndim != 2:
        raise ValueError("Variance regularization expects one vector per image")
    if features.size(0) < 2:
        # A short final validation batch cannot estimate between-image variance.
        zero = features.sum() * 0
        return zero, zero
    std = (features.float().var(dim=0, unbiased=False) + 1e-4).sqrt()
    return torch.relu(target_std - std).mean(), std.mean()


def sigreg_loss(features, target_std, directions=256, knots=17):
    """Batch-scaled SIGReg ECF statistic against N(0, target_std^2 I).

    Uses the positive-half quadrature and Gaussian window from the official
    LeJEPA minimal implementation. An optional leading axis holds independent
    spatial locations, with the batch axis next to last in either case.
    """
    if features.ndim not in (2, 3):
        raise ValueError("SIGReg expects [batch, dim] or [locations, batch, dim]")
    if features.size(-2) < 2:
        return features.sum() * 0
    x = features.float() / target_std
    a = torch.randn(x.size(-1), directions, device=x.device, dtype=x.dtype)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
    t = torch.linspace(0, 3, knots, device=x.device, dtype=x.dtype)
    dt = 3 / (knots - 1)
    window = torch.exp(-t.square() / 2)
    weights = torch.full_like(t, 2 * dt) * window
    weights[[0, -1]] *= 0.5
    phase = (x @ a).unsqueeze(-1) * t
    error = (phase.cos().mean(-3) - window).square() + phase.sin().mean(-3).square()
    return (error @ weights).mean() * x.size(-2)


class SIGRegProjector(nn.Module):
    """Training-time projection head; SIGReg sees its unnormalized output."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, output_dim))

    def forward(self, vectors):
        return self.net(vectors)


def gather_tokens(tokens, ids):
    return tokens.gather(1, ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class ScratchEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=384, depth=6,
                 heads=6, use_cls=False):
        super().__init__()
        if image_size % patch_size or embed_dim % heads:
            raise ValueError("Invalid image/patch size or embedding/head dimensions")
        self.name = "tracka_scratch_vit"
        self.image_size, self.patch_size = image_size, patch_size
        self.emb_dim, self.latent_ndim = embed_dim, 2
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(3 * patch_size ** 2, embed_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls else None
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, heads, 4 * embed_dim,
                                       dropout=0., activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward_features(self, images, return_cls=False, ids_keep=None):
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("Unexpected encoder input resolution")
        tokens = self.patch_embed(patchify(images, self.patch_size)) + self.pos_embed
        if ids_keep is not None:
            tokens = gather_tokens(tokens, ids_keep)
        if self.cls_token is not None:
            tokens = torch.cat([self.cls_token.expand(len(images), -1, -1), tokens], 1)
        for block in self.blocks:
            tokens = block(tokens)  # No attention mask: bidirectional spatial attention.
        tokens = self.norm(tokens)
        cls = tokens[:, 0] if self.cls_token is not None else None
        patches = tokens[:, 1:] if cls is not None else tokens
        if return_cls:
            if cls is None:
                raise ValueError("return_cls requires model.use_cls=true")
            return {"patch_tokens": patches, "cls_token": cls}
        return {"patch_tokens": patches}

    def forward_visible(self, images, ids_keep):
        return self.forward_features(images, ids_keep=ids_keep)["patch_tokens"]

    def forward(self, images):
        return self.forward_features(images)["patch_tokens"]

    def agg(self, tokens):
        return tokens.mean(1)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads, cross_attention):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True) if cross_attention else None
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, nuisance):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        if self.cross is not None:
            x = x + self.cross(self.norm2(x), nuisance, nuisance, need_weights=False)[0]
        return x + self.mlp(self.norm3(x))


class SharedDecoder(nn.Module):
    def __init__(self, encoder, dim=192, depth=2, heads=6, conditioning="additive",
                 rgb_embed_dim=64, use_type_embedding=True, use_cls_global_decoder=False):
        super().__init__()
        if conditioning not in ("additive", "cross_attention") or dim % heads:
            raise ValueError("Invalid decoder configuration")
        if use_cls_global_decoder and encoder.cls_token is None:
            raise ValueError("Global decoder requires model.use_cls=true")
        self.conditioning = conditioning
        self.proj = nn.Linear(encoder.emb_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.empty(1, encoder.num_patches, dim))
        self.rgb_mlp = nn.Sequential(nn.Linear(3, rgb_embed_dim), nn.GELU(),
                                     nn.Linear(rgb_embed_dim, rgb_embed_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 3, rgb_embed_dim)) if use_type_embedding else None
        self.nuisance_proj = nn.Linear(3 * rgb_embed_dim if conditioning == "additive" else rgb_embed_dim, dim)
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads, conditioning == "cross_attention")
                                     for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3 * encoder.patch_size ** 2))
        self.global_proj = nn.Linear(encoder.emb_dim, dim) if use_cls_global_decoder else None
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        if self.type_embed is not None:
            nn.init.trunc_normal_(self.type_embed, std=.02)

    def decode_tokens(self, tokens, eta):
        # eta is actual renderer RGB / 255, shape [B,3 objects,3 channels].
        nuisance = self.rgb_mlp(eta)
        if self.type_embed is not None:
            nuisance = nuisance + self.type_embed
        if self.conditioning == "additive":
            nuisance = self.nuisance_proj(nuisance.flatten(1)).unsqueeze(1)
            tokens = tokens + nuisance
        else:
            nuisance = self.nuisance_proj(nuisance)
        for block in self.blocks:
            tokens = block(tokens, nuisance)
        return self.head(tokens)

    def forward(self, visible, ids_restore, eta):
        x = self.proj(visible)
        missing = ids_restore.shape[1] - x.shape[1]
        x = torch.cat([x, self.mask_token.expand(len(x), missing, -1)], 1)
        return self.decode_tokens(gather_tokens(x, ids_restore) + self.pos_embed, eta)

    def forward_global(self, cls, eta):
        if self.global_proj is None:
            raise ValueError("Global decoder disabled")
        return self.decode_tokens(self.global_proj(cls).unsqueeze(1) + self.pos_embed, eta)


class RobustMAE(nn.Module):
    def __init__(self, model, decoder, loss, mask):
        super().__init__()
        self.encoder = ScratchEncoder(**model)
        self.decoder = SharedDecoder(self.encoder, **decoder)
        self.loss_cfg = resolved_variance_config(loss)
        self.mask_ratios = resolved_mask_ratios(mask)
        selected_feature = (self.loss_cfg["variance_feature"] if self.loss_cfg["lambda_variance"]
                            else self.loss_cfg["sigreg_feature"])
        needs_cls = selected_feature in ("cls", "both", "both_spatial")
        if (self.loss_cfg["lambda_variance"] or self.loss_cfg["lambda_sigreg"]) and needs_cls:
            if self.encoder.cls_token is None:
                raise ValueError("CLS regularization requires model.use_cls=true")
        self.sigreg_cls_projector = None
        self.sigreg_patch_projector = None
        if self.loss_cfg["lambda_sigreg"] and self.loss_cfg["sigreg_space"] == "projector":
            if selected_feature in ("cls", "both", "both_spatial"):
                self.sigreg_cls_projector = SIGRegProjector(
                    self.encoder.emb_dim, self.loss_cfg["sigreg_projector_hidden_dim"],
                    self.loss_cfg["sigreg_projector_dim"])
            if selected_feature in ("patch_mean", "patch_tokens", "both", "both_spatial"):
                self.sigreg_patch_projector = SIGRegProjector(
                    self.encoder.emb_dim, self.loss_cfg["sigreg_projector_hidden_dim"],
                    self.loss_cfg["sigreg_projector_dim"])
            if selected_feature in ("patch_tokens", "both_spatial"):
                if self.loss_cfg["sigreg_patch_samples"] > self.encoder.num_patches:
                    raise ValueError("loss.sigreg_patch_samples exceeds the encoder patch count")

    def target(self, images):
        target = patchify(images, self.encoder.patch_size)
        if self.loss_cfg["norm_pix_loss"]:
            target = (target - target.mean(-1, keepdim=True)) / (
                target.var(-1, unbiased=False, keepdim=True) + 1e-6).sqrt()
        return target

    def objective(self, batch, mode, generator=None):
        if mode not in self.mask_ratios:
            raise ValueError("Select one update objective")
        ratio = self.mask_ratios[mode]
        keep, restore, mask = sample_mask(len(batch["clean"]), self.encoder.num_patches,
                                         ratio, batch["clean"].device, generator)
        features = {}
        views = ["clean"] + (["A", "B"] if mode == "render_only" else ["corrupt"] if mode == "corruption_only" else [])
        for view in views:
            features[view] = self.encoder.forward_features(batch[view], ids_keep=keep,
                                                           return_cls=self.encoder.cls_token is not None)

        def reconstruct(source, target, eta):
            feat = features[source]
            prediction = self.decoder(feat["patch_tokens"], restore, batch[eta])
            truth = self.target(batch[target])
            mse = ((prediction - truth) ** 2).mean(-1)
            rec = (mse * mask).sum() / mask.sum()
            glob = rec.new_zeros(())
            if self.decoder.global_proj is not None:
                glob = ((self.decoder.forward_global(feat["cls_token"], batch[eta]) - truth) ** 2).mean()
            return rec, glob

        z = features["clean"]["patch_tokens"]
        logs = {"latent/norm_clean": z.norm(dim=-1).mean(),
                "latent/variance_clean": z.mean(1).var(0, unbiased=False).mean(),
                "mask/ratio": z.new_tensor(ratio)}
        if mode == "clean_mae":
            rec, glob = reconstruct("clean", "clean", "eta_clean")
            total = rec
            logs["clean/mae"] = rec
        else:
            if mode == "render_only":
                pairs = [reconstruct("clean", q, "eta_" + q) for q in ("A", "B")]
                pairs += [reconstruct(q, "clean", "eta_clean") for q in ("A", "B")]
                shifted = [features[q]["patch_tokens"] for q in ("A", "B")]
                prefix, rec_key, weight = "render", "cross_mae", self.loss_cfg["lambda_render_inv"]
            else:
                pairs = [reconstruct("corrupt", "clean", "eta_clean")]
                shifted = [features["corrupt"]["patch_tokens"]]
                prefix, rec_key, weight = "corruption", "denoise_mae", self.loss_cfg["lambda_corr_inv"]
            rec = torch.stack([r for r, _ in pairs]).mean()
            glob = torch.stack([g for _, g in pairs]).mean()
            inv = torch.stack([(z - other).square().mean() for other in shifted]).mean()
            total = rec + weight * inv
            logs.update({prefix + "/" + rec_key: rec, prefix + "/invariance": inv,
                         prefix + "/weighted_invariance": weight * inv,
                         "latent/" + prefix + "_distance": inv.sqrt(),
                         "latent/norm_" + prefix: torch.stack([v.norm(dim=-1).mean() for v in shifted]).mean()})
        total = total + self.loss_cfg["lambda_cls_global"] * glob
        if self.loss_cfg["lambda_variance"] or self.loss_cfg["lambda_sigreg"]:
            # The training mask is sampled independently for each image. Regularize
            # full-image features so mask-pattern variation cannot satisfy this term.
            full = self.encoder.forward_features(
                batch["clean"], return_cls=self.encoder.cls_token is not None)
            terms = []
            kind = "variance" if self.loss_cfg["lambda_variance"] else "sigreg"
            feature = self.loss_cfg[kind + "_feature"]
            projector_mode = kind == "sigreg" and self.loss_cfg["sigreg_space"] == "projector"

            def regularize(v, name, projector=None):
                logs["latent/std_raw_full_clean_" + name] = v.float().std(dim=0, unbiased=False).mean()
                if kind == "variance":
                    value, std = variance_hinge(v, self.loss_cfg["variance_target_std"])
                    logs["latent/std_full_clean_" + name] = std
                    return value
                logs["latent/std_full_clean_" + name] = (
                    v.float().var(dim=0, unbiased=False) + 1e-4).sqrt().mean()
                projected = projector(v) if projector is not None else v
                value = sigreg_loss(projected, self.loss_cfg["sigreg_target_std"],
                                    self.loss_cfg["sigreg_directions"], self.loss_cfg["sigreg_knots"])
                if projector is not None:
                    logs["latent/std_projected_" + name] = projected.float().std(dim=0, unbiased=False).mean()
                return value

            if feature in ("patch_mean", "both"):
                value = regularize(full["patch_tokens"].mean(1), "patch_mean",
                                   self.sigreg_patch_projector if projector_mode else None)
                terms.append(value)
                logs["loss/" + kind + "_patch_mean"] = value
            if feature in ("cls", "both", "both_spatial"):
                value = regularize(full["cls_token"], "cls",
                                   self.sigreg_cls_projector if projector_mode else None)
                terms.append(value)
                logs["loss/" + kind + "_cls"] = value
            if feature in ("patch_tokens", "both_spatial"):
                # The same spatial positions are sampled for every image; a
                # fixed position embedding alone cannot satisfy each batch test.
                count = self.loss_cfg["sigreg_patch_samples"]
                indices = torch.randperm(self.encoder.num_patches,
                                         device=full["patch_tokens"].device,
                                         generator=generator)[:count]
                selected = full["patch_tokens"][:, indices]
                batch_size, _, dim = selected.shape
                projected = self.sigreg_patch_projector(selected.reshape(batch_size * count, dim))
                projected = projected.reshape(batch_size, count, -1).transpose(0, 1)
                value = sigreg_loss(projected, self.loss_cfg["sigreg_target_std"],
                                    self.loss_cfg["sigreg_directions"], self.loss_cfg["sigreg_knots"])
                terms.append(value)
                logs["loss/sigreg_patch_tokens"] = value
                logs["sigreg/patch_samples"] = value.new_tensor(count)
                logs["latent/std_raw_full_clean_patch_tokens"] = selected.float().std(dim=0, unbiased=False).mean()
                logs["latent/std_projected_patch_tokens"] = projected.float().std(dim=1, unbiased=False).mean()
            regularizer_loss = torch.stack(terms).mean()
            weighted = self.loss_cfg["lambda_" + kind] * regularizer_loss
            total = total + weighted
            logs["loss/" + kind] = regularizer_loss
            logs["loss/weighted_" + kind] = weighted
        logs.update({"loss/total": total, "loss/cls_global": glob})
        return total, logs

    @torch.no_grad()
    def preview(self, batch, mode="render_only"):
        if mode not in self.mask_ratios:
            raise ValueError("Select one update objective")
        keep, restore, mask = sample_mask(len(batch["clean"]), self.encoder.num_patches,
                                         self.mask_ratios[mode], batch["clean"].device)
        rows = []
        for view, eta in (("clean", "eta_clean"), ("A", "eta_A"), ("B", "eta_B"), ("corrupt", "eta_clean")):
            image = batch[view]
            z = self.encoder.forward_visible(image, keep)
            pred = self.decoder(z, restore, batch[eta])
            # norm_pix_loss predictions are normalized patches, not recoverable RGB.
            if self.loss_cfg["norm_pix_loss"]:
                pred = torch.zeros_like(pred)
            visible = patchify(image, self.encoder.patch_size) * (1 - mask.unsqueeze(-1))
            rows.append(torch.cat([image, unpatchify(visible, self.encoder.patch_size),
                                   unpatchify(pred, self.encoder.patch_size)], dim=-1))
        return torch.cat(rows, dim=-2)
