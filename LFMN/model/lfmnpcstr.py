"""LFMN with prior-conditioned image-adaptive soft token routing.

This is the frozen N9/A0 candidate.  It changes only the global mixer in
each TAB: the original residual scaffold and ConvFFN are retained, while
EMA prototypes, hard grouping, IASA, and IRCA are replaced by a soft,
per-image token round trip conditioned on both Fm and Fs.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.lfmn import ConvFFN, Net as BaselineNet, PreNorm


class PriorConditionedSoftTokenRouter(nn.Module):
    """Differentiable pixel-to-token aggregation and token-to-pixel restore."""

    def __init__(
            self, dim=48, prior_dim=32, route_dim=16, num_tokens=32,
            qk_dim=24, heads=4, mlp_dim=96, temperature=1.0):
        super().__init__()
        if qk_dim % heads != 0 or dim % heads != 0:
            raise ValueError('qk_dim and dim must be divisible by heads')
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.dim = int(dim)
        self.prior_dim = int(prior_dim)
        self.route_dim = int(route_dim)
        self.num_tokens = int(num_tokens)
        self.qk_dim = int(qk_dim)
        self.heads = int(heads)
        self.temperature = float(temperature)
        self.eps = 1e-6

        # Keep the original TAB normalization/residual/ConvFFN scaffold.
        self.norm = nn.LayerNorm(dim)
        self.prior_norm = nn.LayerNorm(prior_dim)
        self.route_feature = nn.Linear(dim, route_dim, bias=False)
        self.route_prior = nn.Linear(prior_dim, route_dim, bias=False)
        self.to_assignment = nn.Linear(route_dim, num_tokens, bias=True)
        self.to_pixel_value = nn.Linear(dim, dim, bias=False)

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))

        # Diagnostics are opt-in so normal training retains no graph tensors.
        self.collect_routing_stats = False
        self.last_routing_stats = None

    def _normalized_inputs(self, x, prior):
        if x.ndim != 4 or prior.ndim != 4:
            raise ValueError('feature and prior must be BCHW tensors')
        if x.shape[0] != prior.shape[0] or x.shape[-2:] != prior.shape[-2:]:
            raise ValueError('feature and prior batch/spatial shapes must match')
        if x.shape[1] != self.dim or prior.shape[1] != self.prior_dim:
            raise ValueError('unexpected feature or prior channel count')
        feature = rearrange(x, 'b c h w -> b (h w) c')
        prior_tokens = rearrange(prior, 'b c h w -> b (h w) c')
        return feature, self.norm(feature), self.prior_norm(prior_tokens)

    def assignment_map(self, x, prior):
        """Return A in BxNxK form for diagnostics and causal checks."""
        _, feature, prior_tokens = self._normalized_inputs(x, prior)
        route = F.gelu(
            self.route_feature(feature) + self.route_prior(prior_tokens)
        )
        logits = self.to_assignment(route) / self.temperature
        # FP32 softmax is more robust under AMP; return the model dtype.
        return torch.softmax(logits.float(), dim=-1).to(logits.dtype)

    def _record_stats(self, assignment, mass):
        if not self.collect_routing_stats:
            self.last_routing_stats = None
            return
        with torch.no_grad():
            probability = assignment.float().clamp_min(self.eps)
            pixel_entropy = -(probability * probability.log()).sum(dim=-1)
            pixel_entropy = pixel_entropy / math.log(self.num_tokens)
            token_probability = mass.float() / mass.float().sum(
                dim=-1, keepdim=True
            ).clamp_min(self.eps)
            token_entropy = -(
                token_probability.clamp_min(self.eps)
                * token_probability.clamp_min(self.eps).log()
            ).sum(dim=-1)
            effective_tokens = token_entropy.exp()
            self.last_routing_stats = {
                'pixel_entropy_mean': pixel_entropy.mean().cpu(),
                'effective_tokens_mean': effective_tokens.mean().cpu(),
                'mass_min': mass.float().min().cpu(),
                'mass_max': mass.float().max().cpu(),
                'mass_ratio': (
                    mass.float().max()
                    / mass.float().min().clamp_min(self.eps)
                ).cpu(),
            }

    def forward(self, x, prior):
        _, _, height, width = x.shape
        residual, feature, prior_tokens = self._normalized_inputs(x, prior)
        route = F.gelu(
            self.route_feature(feature) + self.route_prior(prior_tokens)
        )
        logits = self.to_assignment(route) / self.temperature
        assignment = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        pixel_value = self.to_pixel_value(feature)

        # A^T V / sum(A), with accumulation and division in FP32.
        assignment_float = assignment.float()
        mass = assignment_float.sum(dim=1)
        tokens = torch.bmm(
            assignment_float.transpose(1, 2), pixel_value.float()
        )
        tokens = tokens / mass.unsqueeze(-1).clamp_min(self.eps)
        tokens = tokens.to(pixel_value.dtype)
        self._record_stats(assignment, mass)

        query = rearrange(
            self.to_q(tokens), 'b k (h d) -> b h k d', h=self.heads
        )
        key = rearrange(
            self.to_k(tokens), 'b k (h d) -> b h k d', h=self.heads
        )
        value = rearrange(
            self.to_v(tokens), 'b k (h d) -> b h k d', h=self.heads
        )
        token_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0
        )
        token_output = rearrange(
            token_output, 'b h k d -> b k (h d)'
        )
        token_output = self.to_out(token_output)

        # A T restores the K token values to all N spatial positions.
        spatial = torch.bmm(assignment, token_output)
        mixed = residual + spatial
        mixed = mixed + self.mlp(mixed, x_size=(height, width))
        return rearrange(
            mixed, 'b (h w) c -> b c h w', h=height, w=width
        ).contiguous()


class Net(BaselineNet):
    """A0 PCSTR candidate with K=32, Q/K=24, and full-width V=48."""

    def __init__(
            self, scale=2, n_feats=48, side_c=32, n_stage=8,
            mlp_dim=96):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        for stage in range(n_stage):
            self.blocks[stage][0] = PriorConditionedSoftTokenRouter(
                dim=n_feats,
                prior_dim=side_c,
                route_dim=16,
                num_tokens=32,
                qk_dim=24,
                heads=4,
                mlp_dim=mlp_dim,
                temperature=1.0,
            )

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for stage, block in enumerate(self.blocks):
            prev = feat
            beta, gamma = self.sfmls[stage](fs)
            fm = beta * prev + gamma
            global_mixer, local_mixer = block
            transformed = global_mixer(fm, fs)
            transformed = local_mixer(transformed, self.patch_size[stage])
            feat = self.esas[stage](
                prev + self.mid_convs[stage](transformed)
            )

        if self.scale == 4:
            upsampled = self.lrelu(
                self.pixel_shuffle(self.upconv1(x0 + feat))
            )
            upsampled = self.lrelu(
                self.pixel_shuffle(self.upconv2(upsampled))
            )
        else:
            upsampled = self.lrelu(
                self.pixel_shuffle(self.upconv(x0 + feat))
            )
        base = F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        return self.last_conv(upsampled) + base


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
