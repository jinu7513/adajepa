"""Spatial bidirectional ViT and shared nuisance-conditioned MAE decoder."""
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
        self.loss_cfg = loss
        self.mask_ratios = resolved_mask_ratios(mask)

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
