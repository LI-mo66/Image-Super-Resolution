"""F4: current-state spatial gates on stage residual updates of original LFMN."""
import torch
from torch import nn
from torch.nn import functional as F
from .lfmn import Net as BaselineNet


class SpatialResidualGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(48, 8, 1), nn.LeakyReLU(.1),
            nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.Conv2d(8, 48, 1))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)
        self.enabled = True

    def forward(self, current, residual):
        if not self.enabled:
            return residual
        z = F.layer_norm(current.permute(0, 2, 3, 1), (48,), eps=1e-5)
        logits = self.projection(z.permute(0, 3, 1, 2).contiguous())
        return (1 + torch.tanh(logits)) * residual


class Net(BaselineNet):
    def __init__(self, scale=4, **kwargs):
        if scale != 4:
            raise ValueError('F4 supports x4 only')
        super().__init__(scale=scale, **kwargs)
        if self.first_conv.out_channels != 48 or len(self.blocks) != 8:
            raise ValueError('F4 fixes eight 48-channel stages')
        # Add modules after shared baseline initialization without consuming its RNG.
        with torch.random.fork_rng(devices=[]):
            self.residual_gates = nn.ModuleList([SpatialResidualGate() for _ in self.blocks])

    def forward(self, x):
        x0, fs = self.first_conv(x), self.fea(x)
        feat = x0
        for i, (tab, lrsa) in enumerate(self.blocks):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            s = lrsa(tab(beta * prev + gamma), self.patch_size[i])
            residual = self.mid_convs[i](s)
            feat = self.esas[i](prev + self.residual_gates[i](prev, residual))
        u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
        u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        return self.last_conv(u) + F.interpolate(
            x, scale_factor=4, mode='bilinear', align_corners=False)


def make_model(args):
    if args.scale != [4] or args.rgb_range != 255:
        raise ValueError('F4 protocol fixes x4 and rgb_range=255')
    return Net(scale=4)
