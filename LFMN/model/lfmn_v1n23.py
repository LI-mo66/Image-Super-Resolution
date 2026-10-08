"""V1/N23 late SCC residual. Retains all original V1 LRSA and SRPR paths.

This is NOT the previous eight-stage N23 replacement; pure SCC responses are
added at stages 4/8 only. Old V1 overlap normalization remains unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .lfmn import LRSA, patch_divide, patch_reverse
from .lfmnsrprv2 import Net as V1Net
from .v1n23_scc import SCC


class ResidualLRSA(LRSA):
    def __init__(self, source, window):
        nn.Module.__init__(self)
        self.layer = source.layer
        self.normalize_overlap = source.normalize_overlap
        self.correlation = SCC(window)
        self.correlation_gain = nn.Parameter(torch.zeros(()))
        self.window, self.enabled = window, True

    def forward(self, image, ps):
        local = super().forward(image, ps)
        if not self.enabled:
            return local
        h, w = image.shape[-2:]
        padded = F.pad(image, (0, max(0, self.window-w), 0, max(0, self.window-h)), mode='replicate')
        ph, pw = padded.shape[-2:]
        tokens = rearrange(padded, 'b c h w -> b (h w) c')
        projected = self.correlation.project_image(rearrange(
            self.layer[0].norm(tokens), 'b (h w) c -> b c h w', h=ph, w=pw))
        patches, _, _ = patch_divide(projected, self.window-2, self.window)
        b, n, _, _, _ = patches.shape
        response = self.correlation.correlate(rearrange(patches, 'b n c h w -> (b n) (h w) c'))
        patches = rearrange(response, '(b n) (h w) c -> b n c h w', b=b, n=n, h=self.window, w=self.window)
        # Exact coverage belongs ONLY to the new correction operator, not the
        # preserved V1 path. SCC has no input residual or additional FFN here.
        correction = patch_reverse(patches, padded, self.window-2, self.window,
                                   normalize_overlap=True)[..., :h, :w]
        assert correction.shape == local.shape
        return local + .1*torch.tanh(self.correlation_gain)*correction


class Net(V1Net):
    def __init__(self, scale=4, **kwargs):
        super().__init__(scale=scale, **kwargs)
        with torch.random.fork_rng(devices=[]):
            for stage, window in ((3, 32), (7, 64)):
                self.blocks[stage][1] = ResidualLRSA(self.blocks[stage][1], window)

    def set_enabled(self, enabled):
        for stage in (3, 7):
            self.blocks[stage][1].enabled = bool(enabled)


def make_model(args):
    return Net(scale=args.scale[0])
