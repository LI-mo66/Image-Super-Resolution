"""Prior-conditioned ordinary mixer control for the N9 experiment."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.lfmn import ConvFFN, PreNorm
from model.lfmnpcstr import Net as PCSTRNet


class PriorConditionedConvMixer(nn.Module):
    """Matched local/global convolutional mixer without token aggregation."""

    def __init__(self, dim=48, prior_dim=32, hidden_dim=52, mlp_dim=96):
        super().__init__()
        self.dim = int(dim)
        self.prior_dim = int(prior_dim)
        self.norm = nn.LayerNorm(dim)
        self.prior_norm = nn.LayerNorm(prior_dim)
        merged_dim = dim + prior_dim
        self.reduce = nn.Conv2d(merged_dim, hidden_dim, 1)
        self.depthwise = nn.Conv2d(
            hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim
        )
        self.expand = nn.Conv2d(hidden_dim, dim, 1)
        # A pooled projection provides cheap global context and makes both
        # parameter count and analytic MACs close to the PCSTR route.
        self.global_projection = nn.Linear(merged_dim, dim, bias=False)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))

    def forward(self, x, prior):
        if x.shape[0] != prior.shape[0] or x.shape[-2:] != prior.shape[-2:]:
            raise ValueError('feature and prior batch/spatial shapes must match')
        batch, _, height, width = x.shape
        residual = rearrange(x, 'b c h w -> b (h w) c')
        feature = self.norm(residual)
        prior_tokens = self.prior_norm(
            rearrange(prior, 'b c h w -> b (h w) c')
        )
        merged = torch.cat((feature, prior_tokens), dim=-1)
        merged_map = rearrange(
            merged, 'b (h w) c -> b c h w', h=height, w=width
        )
        local = F.gelu(self.reduce(merged_map))
        local = F.gelu(self.depthwise(local))
        local = self.expand(local)
        global_context = self.global_projection(merged.mean(dim=1))
        global_context = global_context.view(batch, self.dim, 1, 1)
        mixed_map = local + global_context
        mixed = residual + rearrange(
            mixed_map, 'b c h w -> b (h w) c'
        )
        mixed = mixed + self.mlp(mixed, x_size=(height, width))
        return rearrange(
            mixed, 'b (h w) c -> b c h w', h=height, w=width
        ).contiguous()


class Net(PCSTRNet):
    """C0 control: prior-conditioned mixer with no token mechanism."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale,
            n_feats=n_feats,
            side_c=side_c,
            n_stage=n_stage,
            mlp_dim=96,
        )
        for stage in range(n_stage):
            self.blocks[stage][0] = PriorConditionedConvMixer(
                dim=n_feats,
                prior_dim=side_c,
                hidden_dim=52,
                mlp_dim=96,
            )


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
