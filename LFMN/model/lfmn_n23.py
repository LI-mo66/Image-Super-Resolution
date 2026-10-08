"""N23: replace LRSA spatial attention, not TAB or the reconstruction backbone.

HiT-inspired SCC, with LFMN's PreNorm/FFN and explicit exact coverage profile.
This is a structural candidate; no accuracy improvement is implied.
"""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .lfmn import Net as BaselineNet, patch_divide, patch_reverse
from .n23_scc import SCC


class SpatialLayer(nn.Module):
    def __init__(self, norm, window):
        super().__init__()
        self.norm = norm
        self.fn = SCC(window_size=window, dim=48, num_heads=4, base_size=8)


class HierarchicalLRSA(nn.Module):
    def __init__(self, source, window):
        super().__init__()
        self.window = int(window)
        self.normalize_overlap = source.normalize_overlap
        self.layer = nn.ModuleList([
            SpatialLayer(source.layer[0].norm, self.window), source.layer[1]
        ])

    def forward(self, x, ps):
        if ps != self.window:
            raise ValueError('N23 window/Net schedule mismatch')
        height, width = x.shape[-2:]
        x = F.pad(x, (0, max(0, ps-width), 0, max(0, ps-height)), mode='replicate')
        layer, ff = self.layer
        normed = layer.norm(rearrange(x, 'b c h w -> b (h w) c'))
        normed = rearrange(normed, 'b (h w) c -> b c h w', h=x.shape[-2])
        # DFE is computed on original spatial neighbors, not separately on
        # sorted content groups or arbitrarily isolated patch boundaries.
        projected = layer.fn.project_image(normed)
        step = ps - 2
        qv, _, _ = patch_divide(projected, step, ps)
        raw, _, _ = patch_divide(x, step, ps)
        batch, count = qv.shape[:2]
        tokens = rearrange(qv, 'b n c h w -> (b n) (h w) c')
        response = layer.fn.correlate(tokens)
        response = rearrange(response, '(b n) (h w) c -> b n c h w',
                             b=batch, n=count, h=ps, w=ps)
        result = patch_reverse(raw + response, x, step, ps,
                               normalize_overlap=self.normalize_overlap)
        result = result[:, :, :height, :width].contiguous()
        tokens = rearrange(result, 'b c h w -> b (h w) c')
        tokens = tokens + ff(tokens, x_size=(height, width))
        return rearrange(tokens, 'b (h w) c -> b c h w', h=height, w=width)


class Net(BaselineNet):
    def __init__(self, scale=4, normalize_overlap=True, **kwargs):
        super().__init__(scale=scale, normalize_overlap=normalize_overlap, **kwargs)
        if len(self.blocks) != 8 or self.first_conv.out_channels != 48:
            raise ValueError('N23 fixes eight 48-channel stages')
        self.patch_size = [8, 16, 32, 64] * 2
        # Do not consume the RNG stream used for data/augmentation after model
        # creation. Common source modules retain their original initialization.
        with torch.random.fork_rng(devices=[]):
            for block, window in zip(self.blocks, self.patch_size):
                block[1] = HierarchicalLRSA(block[1], window)


def make_model(args):
    if args.scale[0] != 4 or getattr(args, 'rgb_range', 255) != 255:
        raise ValueError('Registered N23 screening protocol fixes x4/rgb255')
    return Net(scale=4)
