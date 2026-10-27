# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.register_buffer("bias", generate_mask_matrix(NUM_PATCHES, NUM_FRAMES), persistent=False)

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T].to(dots.device) == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x


class TransitionFiLMPredictor(ViTPredictor):
    """Identity-initialized, per-frame FiLM on the final two predictor blocks.

    Context is supplied by VWorldModel, which knows which transitions were
    observed. In particular this module never manufactures context from its
    own imagined output during MPC optimization.
    """

    uses_transition_film = True

    def __init__(self, *, visual_dim, action_dim, context_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = int(kwargs["num_patches"])
        dim = int(kwargs["dim"])
        self.context_input_dim = 2 * int(visual_dim) + int(action_dim)
        self.context_net = nn.Sequential(
            nn.LayerNorm(self.context_input_dim),
            nn.Linear(self.context_input_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )
        self.film = nn.ModuleDict()
        depth = len(self.transformer.layers)
        for layer_idx in range(max(0, depth - 2), depth):
            projection = nn.Linear(context_dim, 2 * dim)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.film[str(layer_idx)] = projection

    def forward(self, x, transition_context=None, context_valid=None):
        batch, token_count, dim = x.shape
        if token_count % self.num_patches:
            raise ValueError("Predictor token count is not divisible by patches/frame")
        frames = token_count // self.num_patches
        if transition_context is None:
            transition_context = x.new_zeros(batch, frames, self.context_input_dim)
        if context_valid is None:
            context_valid = x.new_zeros(batch, frames, dtype=torch.bool)
        if transition_context.shape != (batch, frames, self.context_input_dim):
            raise ValueError("Transition context has the wrong batch/time/features shape")
        if context_valid.shape != (batch, frames):
            raise ValueError("Transition validity has the wrong batch/time shape")
        context = self.context_net(transition_context)
        x = self.dropout(x + self.pos_embedding[:, :token_count])
        for layer_idx, (attn, ff) in enumerate(self.transformer.layers):
            x = attn(x) + x
            x = ff(x) + x
            projection = self.film[str(layer_idx)] if str(layer_idx) in self.film else None
            if projection is not None:
                gamma, beta = projection(context).chunk(2, dim=-1)
                mask = context_valid.to(dtype=x.dtype).unsqueeze(-1)
                x = x.reshape(batch, frames, self.num_patches, dim)
                x = (1 + (gamma * mask).unsqueeze(2)) * x + (beta * mask).unsqueeze(2)
                x = x.reshape(batch, token_count, dim)
        return self.transformer.norm(x)
