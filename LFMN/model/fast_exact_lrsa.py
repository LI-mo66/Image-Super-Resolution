"""Shared evaluation-only patch fast path for the explicit corrected B0."""
import torch
from torch import nn
from einops import rearrange

from .lfmn import LRSA
from .overlap_fast import OrderedOverlap


class FastExactLRSA(LRSA):
    def __init__(self, source):
        nn.Module.__init__(self)
        self.layer = source.layer
        self.normalize_overlap = True
        self.fast_patches = OrderedOverlap()
        self.fast_eval_enabled = True

    def forward(self, x, ps):
        if (self.training or torch.is_grad_enabled() or not self.fast_eval_enabled
                or not self.normalize_overlap):
            return super().forward(x, ps)
        crops = self.fast_patches.extract(x, ps)
        b, n, c, ph, pw = crops.shape
        tokens = rearrange(crops, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        tokens = attn(tokens) + tokens
        crops = rearrange(tokens, '(b n) (h w) c -> b n c h w', b=b, n=n, h=ph, w=pw)
        x = self.fast_patches.reverse(crops, x, ps)
        height, width = x.shape[-2:]
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        tokens = tokens + ff(tokens, x_size=(height, width))
        return rearrange(tokens, 'b (h w) c -> b c h w', h=height, w=width)
