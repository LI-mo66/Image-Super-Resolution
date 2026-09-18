"""Scratch-training A1: anchored single mid-network prior evolution."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class AnchoredPriorUpdate(nn.Module):
    """Predict a zero-start residual update for the shared shallow prior."""

    def __init__(self, n_feats=48, side_c=32, mid=8):
        super().__init__()
        self.reduce = nn.Conv2d(n_feats + side_c, mid, 1)
        self.spatial = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.project = nn.Conv2d(mid, side_c, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, fs0, state):
        hidden = F.leaky_relu(
            self.reduce(torch.cat((fs0, state), dim=1)), 0.1
        )
        hidden = F.leaky_relu(self.spatial(hidden), 0.1)
        return self.project(hidden)


class Net(BaselineNet):
    """Baseline LFMN with one anchored prior update after stage four."""

    def __init__(self, scale=4, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale,
            n_feats=n_feats,
            side_c=side_c,
            n_stage=n_stage,
        )
        self.prior_update = AnchoredPriorUpdate(
            n_feats=n_feats, side_c=side_c
        )
        self.update_enabled = True

    def forward(self, x):
        x0 = self.first_conv(x)
        fs0 = self.fea(x)
        fs = fs0
        feat = x0
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            fm = beta * prev + gamma
            global_attn, local_attn = self.blocks[i]
            s = local_attn(global_attn(fm), self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
            if i == 3 and self.update_enabled:
                fs = fs0 + self.prior_update(fs0, feat)

        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        return self.last_conv(u) + F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
