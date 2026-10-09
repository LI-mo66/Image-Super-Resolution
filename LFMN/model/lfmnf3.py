"""F3: zero-initialized current-feature correction of the shared SFML prior."""
import torch
from torch import nn
from torch.nn import functional as F
from .lfmn import Net as BaselineNet, SFML


class CurrentSFML(SFML):
    def __init__(self, source):
        nn.Module.__init__(self)
        for name in ('reduce', 'dw', 'expand_a', 'expand_b'):
            setattr(self, name, getattr(source, name))
        self.current_correction = nn.Sequential(
            nn.Conv2d(48, 8, 1), nn.LeakyReLU(.1),
            nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.Conv2d(8, 12, 1))
        nn.init.zeros_(self.current_correction[-1].weight)
        nn.init.zeros_(self.current_correction[-1].bias)
        self.enabled = True

    def forward(self, fs, current):
        u = F.leaky_relu(self.dw(F.leaky_relu(self.reduce(fs), .1)), .1)
        if self.enabled:
            z = F.layer_norm(current.permute(0, 2, 3, 1), (48,), eps=1e-5)
            u = u + self.current_correction(z.permute(0, 3, 1, 2).contiguous())
        return torch.sigmoid(self.expand_a(u)), self.expand_b(u)


class Net(BaselineNet):
    def __init__(self, scale=4, **kwargs):
        if scale != 4:
            raise ValueError('F3 supports x4 only')
        super().__init__(scale=scale, **kwargs)
        if self.first_conv.out_channels != 48 or len(self.blocks) != 8:
            raise ValueError('F3 fixes eight 48-channel stages')
        with torch.random.fork_rng(devices=[]):
            self.sfmls = nn.ModuleList([CurrentSFML(source) for source in self.sfmls])

    def forward(self, x):
        x0, fs = self.first_conv(x), self.fea(x)
        feat = x0
        for i, (tab, lrsa) in enumerate(self.blocks):
            prev = feat
            beta, gamma = self.sfmls[i](fs, prev)
            s = lrsa(tab(beta * prev + gamma), self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
        u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
        u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        return self.last_conv(u) + F.interpolate(
            x, scale_factor=4, mode='bilinear', align_corners=False)


def make_model(args):
    if args.scale != [4] or args.rgb_range != 255:
        raise ValueError('F3 protocol fixes x4 and rgb_range=255')
    return Net(scale=4)
