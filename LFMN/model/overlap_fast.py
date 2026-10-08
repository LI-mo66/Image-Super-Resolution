"""Order-preserving overlap gather/reconstruction for evaluation only.

Geometry matches LFMN's legacy patch order. Reverse sums each pixel's
contributions in ascending patch index, rather than using atomic scatter-add.
These helpers introduce no parameters or persistent checkpoint state.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class OverlapGeometry:
    height: int
    width: int
    patch_size: int
    coordinates: tuple
    extract_indices: torch.Tensor
    contribution_indices: torch.Tensor
    coverage: torch.Tensor
    maxcover: int

    @property
    def patches(self):
        return len(self.coordinates)


class OrderedOverlap(nn.Module):
    """Stateless tensor operations with a non-persistent geometry cache.

    Wrapper must pad inputs to H,W >= ps and enable this only in eval/no_grad.
    geometry(...) exposes coordinates and maxcover for equivalence tests.
    """

    def __init__(self):
        super().__init__()
        self._geometry_cache = {}

    def clear_cache(self):
        self._geometry_cache.clear()

    @property
    def cache_size(self):
        return len(self._geometry_cache)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def geometry(self, height, width, ps, device):
        height, width, ps = int(height), int(width), int(ps)
        if ps <= 2 or min(height, width) < ps:
            raise ValueError("OrderedOverlap requires ps>2 and H,W>=ps")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (height, width, ps, str(device))
        if key in self._geometry_cache:
            return self._geometry_cache[key]
        # Full validation images have many distinct shapes. Retaining every
        # geometry would cause unbounded GPU residency; keep only the current
        # shape (projected/raw extraction and reverse still share this entry).
        self.clear_cache()
        step = ps if height == ps and width == ps else ps - 2
        tops = [min(i, height - ps) for i in range(0, height + step - ps, step)]
        lefts = [min(j, width - ps) for j in range(0, width + step - ps, step)]
        coordinates = tuple((top, left) for top in tops for left in lefts)
        coverage = np.zeros((height, width), dtype=np.int64)
        for top, left in coordinates:
            coverage[top:top + ps, left:left + ps] += 1
        if not np.all(coverage > 0):
            raise RuntimeError("legacy patch geometry left uncovered pixels")
        maxcover = int(coverage.max())
        count = len(coordinates)
        indices = np.empty((count, ps, ps), dtype=np.int64)
        contribution = np.full((maxcover, height * width), -1, dtype=np.int64)
        cursor = np.zeros((height, width), dtype=np.int64)
        yy, xx = np.meshgrid(np.arange(ps), np.arange(ps), indexing="ij")
        patch_offsets = np.arange(ps * ps).reshape(ps, ps)
        for patch, (top, left) in enumerate(coordinates):
            pixels = (yy + top) * width + xx + left
            indices[patch] = pixels
            ranks = cursor[top:top + ps, left:left + ps]
            contribution[ranks, pixels] = patch * ps * ps + patch_offsets
            ranks += 1
        result = OverlapGeometry(
            height, width, ps, coordinates,
            torch.from_numpy(indices.reshape(-1)).to(device),
            torch.from_numpy(contribution).to(device),
            torch.from_numpy(coverage.reshape(-1)).to(device), maxcover,
        )
        self._geometry_cache[key] = result
        return result

    def extract(self, image, ps):
        if image.ndim != 4:
            raise ValueError("extract expects B,C,H,W")
        batch, channels, height, width = image.shape
        geom = self.geometry(height, width, ps, image.device)
        patches = image.flatten(2).index_select(2, geom.extract_indices)
        return (patches.reshape(batch, channels, geom.patches, ps, ps)
                .permute(0, 2, 1, 3, 4).contiguous())

    def reverse(self, crops, image, ps):
        if image.ndim != 4:
            raise ValueError("reverse expects reference B,C,H,W")
        batch, channels, height, width = image.shape
        geom = self.geometry(height, width, ps, image.device)
        if crops.shape != (batch, geom.patches, channels, ps, ps):
            raise ValueError("crop shape does not match reference patch geometry")
        if crops.dtype != image.dtype or crops.device != image.device:
            raise ValueError("crop/reference dtype and device must match")
        values = (crops.permute(0, 2, 1, 3, 4).contiguous()
                  .reshape(batch, channels, -1))
        output = image.new_zeros(batch, channels, height * width)
        for rank in range(geom.maxcover):
            indices = geom.contribution_indices[rank]
            selected = values.index_select(2, indices.clamp_min(0))
            # Only present contributions change the accumulator: even adding
            # an absent +0 can change an existing -0 bit pattern. where also
            # excludes unrelated NaN crop values without using 0*NaN.
            output = torch.where(indices[None, None, :] >= 0,
                                 output + selected, output)
        output = output / geom.coverage.to(dtype=image.dtype)[None, None, :]
        return output.reshape(batch, channels, height, width)
