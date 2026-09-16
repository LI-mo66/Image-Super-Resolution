"""Residual-demand supervised modulation for LFMN.

During training, a shared probe estimates the reconstruction residual available
at each stage.  Its remaining error provides direct supervision for a shared
demand router.  At inference, the probe is skipped and the predicted demand
map controls only the strength of the original SFML transformation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class ResidualDemandModulator(nn.Module):
    """Shared spatial demand router plus a training-only residual probe."""

    def __init__(
            self, side_c=32, n_feats=48, n_stage=8, scale=4,
            mid=4, kernel_size=3, modulation_scale=0.5):
        super().__init__()
        if kernel_size % 2 != 1 or kernel_size < 1:
            raise ValueError('rdsm kernel size must be a positive odd number')
        self.scale = int(scale)
        self.modulation_scale = float(modulation_scale)
        self.reduce = nn.Conv2d(side_c + n_feats, mid, 1)
        self.spatial = nn.Conv2d(
            mid, mid, kernel_size, padding=kernel_size // 2, groups=mid
        )
        self.predict = nn.Conv2d(mid, 1, 1)
        self.stage_embedding = nn.Parameter(torch.zeros(n_stage, mid))
        self.stage_gain = nn.Parameter(torch.zeros(n_stage))

        # A shared training-time probe. Pixel-unshuffled RGB residuals have
        # 3 * scale^2 channels at the LR spatial resolution.
        self.residual_probe = nn.Conv2d(n_feats, 3 * self.scale ** 2, 1)
        # Start from "nothing recovered" so the first oracle demand map is
        # the real HR residual, not random probe error.
        nn.init.zeros_(self.residual_probe.weight)
        nn.init.zeros_(self.residual_probe.bias)

    def demand(self, shallow_prior, stage_feature, stage_index):
        hidden = self.reduce(torch.cat((shallow_prior, stage_feature), dim=1))
        embedding = self.stage_embedding[stage_index].view(1, -1, 1, 1)
        hidden = F.leaky_relu(hidden + embedding, negative_slope=0.1)
        hidden = F.leaky_relu(self.spatial(hidden), negative_slope=0.1)
        return torch.sigmoid(self.predict(hidden))

    def strength(self, demand, stage_index):
        # stage_gain=0 gives strength=1 exactly, preserving the baseline.
        gain = self.modulation_scale * torch.tanh(self.stage_gain[stage_index])
        return 1.0 + gain * (2.0 * demand - 1.0)


class Net(BaselineNet):
    """LFMN whose original SFML residual is routed by predicted demand."""

    def __init__(
            self, scale=2, n_feats=48, side_c=32, n_stage=8,
            rdsm_mid=4, rdsm_kernel=3, rdsm_scale=0.5,
            use_auxiliary=True):
        super().__init__(scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage)
        self.use_auxiliary = bool(use_auxiliary)
        self.rdsm = ResidualDemandModulator(
            side_c=side_c,
            n_feats=n_feats,
            n_stage=n_stage,
            scale=scale,
            mid=rdsm_mid,
            kernel_size=rdsm_kernel,
            modulation_scale=rdsm_scale,
        )

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        demand_predictions = []
        residual_predictions = []

        for i in range(len(self.blocks)):
            prev = feat
            demand = self.rdsm.demand(fs, prev, i)
            demand_predictions.append(demand)
            if self.training and self.use_auxiliary:
                residual_predictions.append(self.rdsm.residual_probe(prev))

            beta, gamma = self.sfmls[i](fs)
            original_modulation = beta * prev + gamma
            modulation_residual = original_modulation - prev
            strength_delta = self.rdsm.strength(demand, i) - 1.0
            # The zero-initialized stage gain makes this addition exactly zero,
            # so a baseline checkpoint starts from the original SFML output.
            fm = original_modulation + strength_delta * modulation_residual

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
        sr = self.last_conv(u) + base

        if self.training and self.use_auxiliary:
            auxiliary = {
                'base': base,
                'scale': self.scale,
                'demand_predictions': tuple(demand_predictions),
                'residual_predictions': tuple(residual_predictions),
            }
            return sr, auxiliary
        return sr


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    loss_spec = getattr(args, 'loss', '')
    use_auxiliary = 'RRES' in loss_spec or 'RDEM' in loss_spec
    return Net(
        scale=scale,
        rdsm_mid=args.rdsm_mid,
        rdsm_kernel=args.rdsm_kernel,
        rdsm_scale=args.rdsm_scale,
        use_auxiliary=use_auxiliary,
    )
