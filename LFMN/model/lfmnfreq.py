"""LFMN with a shared high-frequency prior and stage-selective gates.

The frequency projection is computed once and reused by all reconstruction
stages. Zero-initialized channel gates preserve the exact baseline function
when loading an LFMN checkpoint while still receiving gradients immediately.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class SharedFrequencyPrior(nn.Module):
    """Project a fixed local high-pass residual once for all LFMN stages."""

    def __init__(self, side_c=32, n_feats=48, n_stage=8, residual_scale=0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.project = nn.Conv2d(side_c, 2 * n_feats, 1, bias=False)
        self.stage_gates = nn.Parameter(torch.zeros(n_stage, 2 * n_feats))

    def encode(self, shallow_prior):
        low = F.avg_pool2d(
            shallow_prior,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        return torch.tanh(self.project(shallow_prior - low))

    def correction(self, encoded_prior, stage_index):
        gate = torch.tanh(self.stage_gates[stage_index]).view(1, -1, 1, 1)
        correction = self.residual_scale * encoded_prior * gate
        return correction.chunk(2, dim=1)


class Net(BaselineNet):
    """Baseline LFMN plus an amortized frequency-selective modulation prior."""

    def __init__(
            self, scale=2, n_feats=48, side_c=32, n_stage=8,
            freq_scale=0.1):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        self.freq_prior = SharedFrequencyPrior(
            side_c=side_c,
            n_feats=n_feats,
            n_stage=n_stage,
            residual_scale=freq_scale,
        )

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        encoded_frequency = self.freq_prior.encode(fs)
        feat = x0
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            delta_beta, delta_gamma = self.freq_prior.correction(
                encoded_frequency, i
            )
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


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale, freq_scale=args.freq_scale)
