"""V1N23C: calibrated SCC writeback BEFORE the preserved local FFN.

This is an independent adaptation, not the complete HiT-SR architecture.
The V1 forward, original local attention/FFN and legacy overlap are unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .lfmn import LRSA, patch_divide, patch_reverse
from .lfmnsrprv2 import Net as V1Net
from .v1n23c_scc import SCC


class CalibratedLRSA(LRSA):
    def __init__(self, source, window):
        nn.Module.__init__(self)
        self.layer = source.layer
        self.normalize_overlap = source.normalize_overlap
        self.correlation = SCC(window)
        self.context_norm_in = nn.LayerNorm(48, eps=1e-5)
        self.context_norm_out = nn.LayerNorm(48, eps=1e-5)
        self.correlation_gain = nn.Parameter(torch.zeros(()))
        self.window = window
        self.enabled = True
        self.collect_statistics = False
        self.last_statistics = {}
        # Apply official small Linear initialization only to NEW SCC modules.
        for module in self.correlation.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def correction(self, image):
        h, w = image.shape[-2:]
        padded = F.pad(image, (0, max(0, self.window-w),
                             0, max(0, self.window-h)), mode='replicate')
        ph, pw = padded.shape[-2:]
        tokens = rearrange(padded, 'b c h w -> b (h w) c')
        projected = self.correlation.project_image(rearrange(
            self.context_norm_in(tokens), 'b (h w) c -> b c h w', h=ph, w=pw))
        patches, _, _ = patch_divide(projected, self.window-2, self.window)
        b, n, _, _, _ = patches.shape
        response = self.correlation.correlate(
            rearrange(patches, 'b n c h w -> (b n) (h w) c'))
        patches = rearrange(response, '(b n) (h w) c -> b n c h w',
                            b=b, n=n, h=self.window, w=self.window)
        raw = patch_reverse(patches, padded, self.window-2, self.window,
                            normalize_overlap=True)[..., :h, :w]
        normalized = self.context_norm_out(rearrange(raw, 'b c h w -> b h w c'))
        return raw, rearrange(normalized, 'b h w c -> b c h w')

    def forward(self, image, ps):
        self.last_statistics = {}
        if not self.enabled:
            return super().forward(image, ps)
        # Exact original attention/residual/legacy-stitch sequence. Do not
        # normalize the original path to make historical comparisons look better.
        patches, _, _ = patch_divide(image, ps-2, ps)
        b, n, c, ph, pw = patches.shape
        tokens = rearrange(patches, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        tokens = attn(tokens) + tokens
        patches = rearrange(tokens, '(b n) (h w) c -> b n c h w', n=n, w=pw)
        local = patch_reverse(patches, image, ps-2, ps,
                              normalize_overlap=self.normalize_overlap)
        raw, calibrated = self.correction(image)
        if local.shape != calibrated.shape:
            raise RuntimeError('SCC/local layout mismatch')
        delta = .1*torch.tanh(self.correlation_gain)*calibrated
        fused = local + delta
        h, w = fused.shape[-2:]
        tokens = rearrange(fused, 'b c h w -> b (h w) c')
        output = ff(tokens, x_size=(h, w)) + tokens
        if self.collect_statistics:
            with torch.no_grad():
                rms = lambda x: x.detach().float().square().mean().sqrt()
                energy = delta.detach().float().square().mean()
                mean_energy = delta.detach().float().mean((-2, -1)).square().mean()
                self.last_statistics = {
                    'window': self.window,
                    'gain': float(.1*torch.tanh(self.correlation_gain.detach())),
                    'raw_rms': float(rms(raw)),
                    'calibrated_rms': float(rms(calibrated)),
                    'local_rms': float(rms(local)),
                    'delta_rms': float(rms(delta)),
                    'delta_over_local_rms': float(rms(delta)/rms(local).clamp_min(1e-12)),
                    'spatial_mean_energy_fraction': float(mean_energy/energy.clamp_min(1e-20)),
                    'delta_abs_max': float(delta.detach().abs().max()),
                }
        return rearrange(output, 'b (h w) c -> b c h w', h=h, w=w)


class Net(V1Net):
    def __init__(self, scale=4, **kwargs):
        super().__init__(scale=scale, **kwargs)
        # Preserve ALL common V1 initial weights and global data/init RNG.
        with torch.random.fork_rng(devices=[]):
            for stage, window in ((3, 32), (7, 64)):
                self.blocks[stage][1] = CalibratedLRSA(self.blocks[stage][1], window)

    def set_enabled(self, enabled):
        for stage in (3, 7):
            self.blocks[stage][1].enabled = bool(enabled)

    def set_collect_statistics(self, enabled):
        for stage in (3, 7):
            self.blocks[stage][1].collect_statistics = bool(enabled)

    def mechanism_statistics(self):
        return {str(stage+1): dict(self.blocks[stage][1].last_statistics) for stage in (3, 7)}


def make_model(args):
    if args.scale != [4] or args.rgb_range != 255:
        raise ValueError('V1N23C protocol is x4, rgb_range=255')
    return Net(scale=4)
