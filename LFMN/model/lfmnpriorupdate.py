"""Refresh the shared SFML prior once, after reconstruction stage four."""
import torch
from torch import nn
from torch.nn import functional as F

from model.lfmn import Net as BaselineNet


class PriorUpdate(nn.Module):
    def __init__(self, n_feats=48, side_c=32, mid=8):
        super().__init__()
        self.reduce = nn.Conv2d(n_feats + side_c, mid, 1)
        self.spatial = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.project = nn.Conv2d(mid, side_c, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, fs, state):
        hidden = F.leaky_relu(self.reduce(torch.cat((fs, state), dim=1)), 0.1)
        return self.project(F.leaky_relu(self.spatial(hidden), 0.1))


class Net(BaselineNet):
    def __init__(self, scale=4, mode='state'):
        super().__init__(scale=scale)
        if mode not in ('shallow', 'state', 'change'):
            raise ValueError('unknown prior update mode: ' + mode)
        self.mode = mode
        self.prior_update = PriorUpdate()
        self.update_enabled = True  # Diagnostic ablation; not a training option.

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        previous_state = None
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            fm = beta * prev + gamma
            global_attn, local_attn = self.blocks[i]
            s = local_attn(global_attn(fm), self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
            if i == 3 and self.update_enabled:
                # Repeat shallow channels to match the 48-channel state input.
                source = torch.cat((fs, fs[:, :16]), dim=1) if self.mode == 'shallow' else feat
                if self.mode == 'change':
                    source = feat - previous_state
                fs = fs + self.prior_update(fs, source)
            previous_state = feat
        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        return self.last_conv(u) + F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )


def make_model(args):
    return Net(
        scale=args.scale[0],
        mode=getattr(args, 'prior_update_mode', 'state'),
    )
