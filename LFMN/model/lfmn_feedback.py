"""LFMN with lightweight stage-state feedback modulation.

The baseline model remains in lfmn.py. This variant adds zero-initialized
corrections to the SFML modulation at selected reconstruction stages, so a
baseline checkpoint produces the same output before fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class StageFeedbackCorrection(nn.Module):
    """Predict bounded residual corrections for beta and gamma from Xi-1."""

    def __init__(self, n_feats=48, mid=8, residual_scale=0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.reduce = nn.Conv2d(n_feats, mid, 1)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.expand = nn.Conv2d(mid, n_feats * 2, 1)

        # Preserve the exact baseline function when loading an LFMN checkpoint.
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, stage_feature):
        state = F.leaky_relu(self.reduce(stage_feature), 0.1)
        state = F.leaky_relu(self.dw(state), 0.1)
        correction = self.residual_scale * torch.tanh(self.expand(state))
        return correction.chunk(2, dim=1)


class Net(BaselineNet):
    """Baseline LFMN plus feedback corrections at selected 1-based stages."""

    def __init__(
            self, scale=2, n_feats=48, side_c=32, n_stage=8,
            feedback_stages=(3, 5, 7), feedback_mid=8, feedback_scale=0.1):
        super().__init__(scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage)

        normalized = sorted(set(int(stage) for stage in feedback_stages))
        invalid = [stage for stage in normalized if stage < 1 or stage > n_stage]
        if invalid:
            raise ValueError(
                'feedback stages must be between 1 and {}; got {}'.format(
                    n_stage, invalid
                )
            )
        if not normalized:
            raise ValueError('at least one feedback stage is required')

        self.feedback_stages = tuple(normalized)
        self.feedback = nn.ModuleDict({
            str(stage): StageFeedbackCorrection(
                n_feats=n_feats,
                mid=feedback_mid,
                residual_scale=feedback_scale,
            )
            for stage in self.feedback_stages
        })

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            stage = i + 1
            if stage in self.feedback_stages:
                delta_beta, delta_gamma = self.feedback[str(stage)](prev)
                beta = beta + delta_beta
                gamma = gamma + delta_gamma
            fm = beta * prev + gamma
            g_attn, l_attn = self.blocks[i]
            t = g_attn(fm)
            s = l_attn(t, self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))

        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        base = F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        return self.last_conv(u) + base


def _parse_stages(value):
    if isinstance(value, str):
        return tuple(int(item) for item in value.replace(',', '-').split('-') if item)
    return tuple(value)


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(
        scale=scale,
        feedback_stages=_parse_stages(args.feedback_stages),
        feedback_mid=args.feedback_mid,
        feedback_scale=args.feedback_scale,
    )
