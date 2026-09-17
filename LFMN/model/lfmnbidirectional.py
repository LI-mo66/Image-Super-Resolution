"""LFMN with a lightweight reverse feature-fusion path.

The baseline stages run unchanged first.  A cached late-stage feature then
sends zero-initialized 1x1 residual messages to the later half of the stage
states.  This keeps public checkpoints exactly compatible and makes the first
forward pass identical to LFMN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class ReverseFeatureFusion(nn.Module):
    """Project the final state back into selected cached stage states."""

    def __init__(self, n_feats=48, n_stage=8, start_stage=4):
        super().__init__()
        self.start_stage = int(start_stage)
        self.projections = nn.ModuleList([
            nn.Conv2d(n_feats, n_feats, 1) if i >= self.start_stage
            else nn.Identity()
            for i in range(n_stage)
        ])
        for projection in self.projections:
            if isinstance(projection, nn.Conv2d):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)

    def corrections(self, final_feature, states):
        return [
            state + projection(final_feature) if i >= self.start_stage else state
            for i, (projection, state) in enumerate(zip(self.projections, states))
        ]


class Net(BaselineNet):
    """Baseline LFMN plus cached late-to-early residual fusion."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8,
                 bidirectional_start=4):
        super().__init__(scale=scale, n_feats=n_feats, side_c=side_c,
                         n_stage=n_stage)
        self.bidirectional = ReverseFeatureFusion(
            n_feats=n_feats, n_stage=n_stage, start_stage=bidirectional_start
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
            g_attn, l_attn = self.blocks[i]
            t = g_attn(fm)
            s = l_attn(t, self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
            states.append(feat)

        # The reverse messages use the last forward state but do not change
        # the zero-initialized baseline function. Aggregate corrected states
        # with a zero-start projection into the decoder input.
        corrected = self.bidirectional.corrections(feat, states)
        reverse_delta = sum(
            (updated - original)
            for updated, original in zip(corrected, states)
        )
        feat = feat + reverse_delta

        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                             align_corners=False)
        return self.last_conv(u) + base


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(
        scale=scale,
        bidirectional_start=getattr(args, 'bidirectional_start', 4),
    )
