"""N23 spatial/channel correlation primitives inspired by HiT-SR (ECCV 2024).

This is an independent adaptation, not the official HiT-SR network. Spatial
projection happens on the original image before overlapping patch extraction.
The production relative-bias path never builds a (p*p, p*p) tensor.
"""

import torch
from torch import nn
import torch.nn.functional as F


class DFE(nn.Module):
    """Spatial and channel projections with the approved 48 -> 9 -> 48 width."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(48, 9, 1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(9, 9, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(9, 48, 1),
        )
        self.linear = nn.Conv2d(48, 48, 1)

    def forward(self, image):
        return self.conv(image) * self.linear(image)


class DynamicPositionBias(nn.Module):
    """67 parameters for C=48, four spatial heads (official hidden width 3)."""

    def __init__(self):
        super().__init__()
        self.pos_proj = nn.Linear(2, 3)
        self.pos1 = nn.Sequential(nn.LayerNorm(3), nn.ReLU(), nn.Linear(3, 3))
        self.pos2 = nn.Sequential(nn.LayerNorm(3), nn.ReLU(), nn.Linear(3, 3))
        self.pos3 = nn.Sequential(nn.LayerNorm(3), nn.ReLU(), nn.Linear(3, 4))

    def forward(self, offsets):
        return self.pos3(self.pos2(self.pos1(self.pos_proj(offsets))))


class SCC(nn.Module):
    """Fixed-window spatial/channel correlation; no normalization or gates here.

    API:
      project_image: B,48,H,W -> B,48,H,W (before patch extraction).
      correlate:     Bw,p*p,48 -> Bw,p*p,48 (already projected patches).
      forward:       B,48,p,p -> B,48,p,p (isolated-patch test convenience).
    """

    def __init__(self, window_size, dim=48, base_size=8, num_heads=4):
        super().__init__()
        if dim != 48 or base_size != 8 or num_heads != 4:
            raise ValueError("N23 is fixed to dim=48, base_size=8, num_heads=4")
        if window_size not in (8, 16, 32, 64):
            raise ValueError("N23 windows must be one of 8, 16, 32, 64")
        self.dim = dim
        self.window_size = int(window_size)
        self.base_size = base_size
        self.num_heads = num_heads
        self.head_dim = 6
        self.qv = DFE()
        self.proj = nn.Linear(48, 48)
        self.spatial_linear = nn.Linear((self.window_size // 8) ** 2, 1)
        self.pos = DynamicPositionBias()

    def project_image(self, image):
        if image.ndim != 4 or image.shape[1] != 48:
            raise ValueError("project_image expects B,48,H,W")
        return self.qv(image)

    def _offset_table(self):
        p = self.window_size
        weight = self.pos.pos_proj.weight
        axis = torch.arange(1 - p, p, device=weight.device, dtype=weight.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        offsets = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
        return self.pos(offsets).reshape(2 * p - 1, 2 * p - 1, 4)

    def position_bias(self, reference=False):
        """Return [4,p*p,64]; exact arithmetic matches the official aggregation.

        Each anchor averages positional biases over its g*g source positions.
        A two-dimensional prefix sum computes all box means on the relative
        offset table in O(p*p) work/storage, followed by O(64*p*p) indexing.
        Floating-point summation order differs from the small reference path.
        """
        if reference:
            return self.reference_bias()
        p = self.window_size
        g = p // 8
        table = self._offset_table().permute(2, 0, 1)
        integral = F.pad(table.cumsum(-2).cumsum(-1), (1, 0, 1, 0))
        boxes = (integral[:, g:, g:] - integral[:, :-g, g:]
                 - integral[:, g:, :-g] + integral[:, :-g, :-g]) / (g * g)
        axis = torch.arange(p, device=table.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        anchor = torch.arange(8, device=table.device) * g
        ay, ax = torch.meshgrid(anchor, anchor, indexing="ij")
        # Offsets run from query-(anchor+g-1) through query-anchor.
        row = yy.flatten()[:, None] - ay.flatten()[None, :] - (g - 1) + p - 1
        col = xx.flatten()[:, None] - ax.flatten()[None, :] - (g - 1) + p - 1
        return boxes[:, row, col]

    def reference_bias(self):
        """Testing only: direct official pairwise average, restricted to p<=16."""
        p = self.window_size
        if p > 16:
            raise ValueError("quadratic reference_bias is restricted to p<=16")
        table = self._offset_table()
        axis = torch.arange(p, device=table.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        delta = coords[:, None, :] - coords[None, :, :] + p - 1
        pairwise = table[delta[..., 0], delta[..., 1]]
        g = p // 8
        return (pairwise.reshape(p * p, 8, g, 8, g, 4)
                .permute(5, 0, 1, 3, 2, 4).reshape(4, p * p, 64, g * g)
                .mean(-1))

    def _summarize_values(self, values):
        batch, heads, _, channels = values.shape
        g = self.window_size // 8
        cells = (values.reshape(batch, heads, 8, g, 8, g, channels)
                 .permute(0, 1, 2, 4, 6, 3, 5)
                 .reshape(batch, heads, 64, channels, g * g))
        return self.spatial_linear(cells).squeeze(-1)

    def correlate(self, tokens, reference_bias=False):
        p = self.window_size
        if tokens.ndim != 3 or tokens.shape[1:] != (p * p, 48):
            raise ValueError("correlate expects Bw,p*p,48 projected tokens")
        batch, count, _ = tokens.shape
        qv = tokens.reshape(batch, count, 2, 4, 6).permute(2, 0, 3, 1, 4)
        query, value = qv.unbind(0)
        summary = self._summarize_values(value)
        scores = query @ summary.transpose(-2, -1) / self.head_dim
        scores = scores + self.position_bias(reference=reference_bias).unsqueeze(0)
        spatial = (scores @ summary).transpose(1, 2).reshape(batch, count, 24)
        query = query.transpose(1, 2).reshape(batch, count, 24)
        value = value.transpose(1, 2).reshape(batch, count, 24)
        channel_map = query.transpose(-2, -1) @ value / count
        channel = (channel_map @ value.transpose(-2, -1)).transpose(-2, -1)
        return self.proj(torch.cat((spatial, channel), dim=-1))

    def forward(self, image):
        p = self.window_size
        if image.ndim != 4 or image.shape[1:] != (48, p, p):
            raise ValueError("forward expects B,48,p,p; use split API for full images")
        projected = self.project_image(image)
        tokens = projected.flatten(2).transpose(1, 2)
        return self.correlate(tokens).transpose(1, 2).reshape_as(image)
