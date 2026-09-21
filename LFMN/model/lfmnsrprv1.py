"""N11 SRPRv1: residual-driven persistent proximal reconstruction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfmn import Net as BaselineNet


class PairedBicubic:
    """Fixed analysis/synthesis pair used for the x4 observation residual.

    The synthesis path is the exact transpose of the separable bilinear
    analysis kernel used here. It is intentionally named backprojection, not
    bicubic interpolation, because the adjoint identity is checked separately.
    """

    def __init__(self, scale=4):
        self.scale = int(scale)
        if self.scale != 4:
            raise ValueError('SRPRv1 currently requires scale=4')

    def down(self, x):
        return F.avg_pool2d(x, kernel_size=4, stride=4)

    def adjoint(self, r, output_size):
        h, w = output_size
        return F.interpolate(r, size=(h, w), mode='nearest') / 16.0


class ProximalState(nn.Module):
    def __init__(self, in_channels, state_channels=48):
        super().__init__()
        hidden = 16
        self.candidate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, state_channels, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, state_channels, 1),
        )
        self.feature_delta = nn.Sequential(
            nn.Conv2d(state_channels * 3, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 48, 1),
        )
        self.rgb_delta = nn.Conv2d(state_channels * 3, 3, 1)
        nn.init.zeros_(self.rgb_delta.weight)
        nn.init.zeros_(self.rgb_delta.bias)

    def forward(self, feature, state, backprojection):
        joined = torch.cat((feature, state, backprojection), dim=1)
        candidate = self.candidate(joined)
        gate = torch.sigmoid(self.gate(joined))
        next_state = state + gate * (candidate - state)
        feature_delta = self.feature_delta(torch.cat(
            (feature, next_state, backprojection), dim=1
        ))
        rgb_delta = self.rgb_delta(torch.cat((feature, next_state, backprojection), dim=1))
        return next_state, feature_delta, rgb_delta, gate


class Net(BaselineNet):
    """Baseline LFMN plus an independent HR residual-state recursion."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8,
                 normalize_overlap=False):
        super().__init__(scale, n_feats, side_c, n_stage, normalize_overlap)
        if int(scale) != 4:
            raise ValueError('SRPRv1 requires scale=4')
        self.observation = PairedBicubic(scale)
        self.state_channels = 48
        self.hr_lift = nn.Conv2d(48, self.state_channels, 3, padding=1)
        self.state_proj = nn.Conv2d(3, self.state_channels, 3, padding=1)
        self.proximal = nn.ModuleList([
            ProximalState(48 * 3, self.state_channels)
            for _ in range(n_stage)
        ])
        self.last_diagnostics = {}

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
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
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x_hat = self.last_conv(u) + base

        state = torch.zeros(
            x_hat.shape[0], self.state_channels, x_hat.shape[2], x_hat.shape[3],
            device=x_hat.device, dtype=x_hat.dtype
        )
        residual_norms, back_norms, state_norms = [], [], []
        update_norms, gate_means, gate_saturation = [], [], []
        feature_hr = F.interpolate(self.hr_lift(feat), size=x_hat.shape[-2:], mode='bilinear', align_corners=False)
        for i, prox in enumerate(self.proximal):
            residual = x - self.observation.down(x_hat)
            back = self.observation.adjoint(residual, x_hat.shape[-2:])
            back_state = self.state_proj(back)
            state, feature_delta, rgb_delta, gate = prox(feature_hr, state, back_state)
            feature_hr = feature_hr + F.interpolate(feature_delta, size=feature_hr.shape[-2:], mode='bilinear', align_corners=False)
            x_hat = x_hat + rgb_delta
            residual_norms.append(residual.detach().float().mean())
            back_norms.append(back.detach().float().mean())
            state_norms.append(state.detach().float().mean())
            update_norms.append(rgb_delta.detach().float().abs().mean())
            gate_means.append(gate.detach().float().mean())
            gate_saturation.append(((gate < 0.01) | (gate > 0.99)).float().mean())

        self.last_diagnostics = {
            'residual_mean': torch.stack(residual_norms),
            'backprojection_mean': torch.stack(back_norms),
            'state_mean': torch.stack(state_norms),
            'rgb_update_abs_mean': torch.stack(update_norms),
            'gate_mean': torch.stack(gate_means),
            'gate_saturation': torch.stack(gate_saturation),
        }
        return x_hat


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
