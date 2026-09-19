"""LFMN with zero-start cross-stage difference memory aggregation.

The baseline decoder consumes only the last of eight reconstruction states.
This variant retains the seven adjacent state differences and lets a tiny
per-channel gate recover useful information that later stages may suppress.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class StageDifferenceMemory(nn.Module):
    """Fuse adjacent stage differences with bounded per-channel gates."""

    def __init__(self, n_stage=8, n_feats=48):
        super().__init__()
        if n_stage < 2:
            raise ValueError('stage-difference memory requires at least two stages')
        self.gates = nn.Parameter(torch.zeros(n_stage - 1, n_feats, 1, 1))
        self.enabled = True

    def forward(self, states):
        if len(states) != self.gates.shape[0] + 1:
            raise ValueError(
                'expected {} stage states, got {}'.format(
                    self.gates.shape[0] + 1, len(states)
                )
            )
        if not self.enabled:
            return torch.zeros_like(states[-1])
        weights = torch.tanh(self.gates)
        correction = torch.zeros_like(states[-1])
        for index, (earlier, later) in enumerate(zip(states[:-1], states[1:])):
            correction = correction + weights[index] * (earlier - later)
        return correction


class Net(BaselineNet):
    """Baseline LFMN plus a lightweight intermediate-state memory."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        self.stage_diff = StageDifferenceMemory(
            n_stage=n_stage, n_feats=n_feats
        )

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        states = []
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            fm = beta * prev + gamma
            global_attn, local_attn = self.blocks[i]
            transformed = local_attn(
                global_attn(fm), self.patch_size[i]
            )
            feat = self.esas[i](prev + self.mid_convs[i](transformed))
            states.append(feat)

        feat = feat + self.stage_diff(states)
        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        base = F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        return self.last_conv(u) + base


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
