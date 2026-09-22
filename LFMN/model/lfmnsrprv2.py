"""N12 SRPRv2: low-resolution residual-state proximal reconstruction."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfmn import (
    Net as BaselineNet, SFML, ESA, TAB, LRSA, default_conv, fea_Net,
)


def _cubic(x):
    ax = x.abs()
    return torch.where(
        ax <= 1,
        1.5 * ax**3 - 2.5 * ax**2 + 1,
        torch.where(ax <= 2, -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2,
                    torch.zeros_like(ax)),
    )


def _reflect_indices(index, length):
    if length <= 1:
        return torch.zeros_like(index)
    period = 2 * length - 2
    index = index.remainder(period)
    return torch.where(index >= length, period - index, index)


def _bicubic_matrix(in_length, out_length, device, dtype):
    scale = float(out_length) / float(in_length)
    kernel_width = 4.0 / scale if scale < 1 else 4.0
    positions = torch.arange(1, out_length + 1, device=device, dtype=dtype)
    u = positions / scale + 0.5 * (1.0 - 1.0 / scale)
    left = torch.floor(u - kernel_width / 2)
    count = math.ceil(kernel_width) + 2
    offsets = torch.arange(count, device=device, dtype=dtype)
    indices = left[:, None] + offsets[None, :]
    distance = u[:, None] - indices
    weights = _cubic(distance * scale) * scale if scale < 1 else _cubic(distance)
    weights = weights / weights.sum(dim=1, keepdim=True)
    indices = _reflect_indices(indices.long() - 1, in_length)
    matrix = torch.zeros(out_length, in_length, device=device, dtype=dtype)
    matrix.scatter_add_(1, indices, weights)
    return matrix


class BicubicAdjoint:
    """Explicit separable bicubic analysis and its exact matrix transpose."""

    def __init__(self, scale=4):
        if int(scale) != 4:
            raise ValueError('SRPRv2 currently requires scale=4')
        self.scale = 4
        self._cache = {}

    def _matrices(self, height, width, device, dtype):
        key = (height, width, str(device), dtype)
        if key not in self._cache:
            self._cache[key] = (
                _bicubic_matrix(height, height // 4, device, dtype),
                _bicubic_matrix(width, width // 4, device, dtype),
            )
        return self._cache[key]

    def down(self, x):
        b, c, height, width = x.shape
        ah, aw = self._matrices(height, width, x.device, x.dtype)
        return torch.einsum('oh,bchw,pw->bcop', ah, x, aw)

    def adjoint(self, residual, output_size):
        height, width = output_size
        ah, aw = self._matrices(height, width, residual.device, residual.dtype)
        return torch.einsum('ho,bcop,wp->bchw', ah.t(), residual, aw.t())


class LRProximalWriteback(nn.Module):
    def __init__(self, channels=48, hidden=16):
        super().__init__()
        inputs = channels * 2 + 3
        self.candidate = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(hidden, channels, 1),
        )
        self.gate = nn.Conv2d(inputs, channels, 1)
        self.writeback = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.q_update = nn.Conv2d(channels + 3, 3, 1)
        nn.init.zeros_(self.writeback[-1].weight)
        nn.init.zeros_(self.writeback[-1].bias)
        nn.init.zeros_(self.q_update.weight)
        nn.init.zeros_(self.q_update.bias)

    def forward(self, feat, state, observation):
        joined = torch.cat((feat, state, observation), dim=1)
        candidate = self.candidate(joined)
        gate = torch.sigmoid(self.gate(joined))
        next_state = state + gate * (candidate - state)
        correction_input = torch.cat((feat, next_state, observation), dim=1)
        delta_feat = self.writeback(correction_input)
        delta_q = self.q_update(torch.cat((next_state, observation), dim=1))
        return next_state, delta_feat, delta_q, gate


class Net(BaselineNet):
    """LFMN backbone with LR persistent residual state and stage writeback."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8,
                 normalize_overlap=False):
        super().__init__(scale, n_feats, side_c, n_stage, normalize_overlap)
        if int(scale) != 4:
            raise ValueError('SRPRv2 requires scale=4')
        self.observation = BicubicAdjoint(scale)
        self.state_channels = n_feats
        self.proximal = nn.ModuleList([
            LRProximalWriteback(n_feats, hidden=16) for _ in range(n_stage)
        ])
        # A shared, cheap HR probe makes each observation residual depend on
        # the current backbone feature before the next stage is run.
        self.stage_probe = nn.Conv2d(n_feats, 3 * scale * scale, 1)
        self.last_diagnostics = {}

    def diagnostics_snapshot(self):
        """Return the last per-stage diagnostics detached on CPU."""
        return {
            key: value.detach().float().cpu().clone()
            for key, value in self.last_diagnostics.items()
        }

    def _final_decode(self, x, x0, feat, q):
        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return self.last_conv(u) + base + F.interpolate(q, scale_factor=self.scale, mode='bilinear', align_corners=False)

    def _observation_estimate(self, x, feat, q):
        base = F.interpolate(x + q, scale_factor=4, mode='bilinear', align_corners=False)
        return base + F.pixel_shuffle(self.stage_probe(feat), 4)

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        state = torch.zeros_like(feat)
        q = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        diagnostics = {key: [] for key in (
            'residual_l2', 'backprojection_l2', 'observation_l2',
            'state_l2', 'state_delta_l2', 'feature_delta_l2',
            'q_delta_l2', 'gate_mean', 'gate_std', 'gate_saturation',
            'data_consistency_ratio',
        )}
        for i, prox in enumerate(self.proximal):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            fm = beta * prev + gamma
            g_attn, l_attn = self.blocks[i]
            t = g_attn(fm)
            s = l_attn(t, self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
            x_hat = self._observation_estimate(x, feat, q)
            residual = x - self.observation.down(x_hat)
            back = self.observation.adjoint(residual, x_hat.shape[-2:])
            observation = F.interpolate(back, size=x.shape[-2:], mode='bilinear', align_corners=False)
            state_before = state
            state, delta_feat, delta_q, gate = prox(feat, state, observation)
            state_delta = state - state_before
            feat = feat + delta_feat
            q = q + delta_q
            next_x = self._observation_estimate(x, feat, q)
            next_residual = x - self.observation.down(next_x)
            diagnostics['residual_l2'].append(residual.detach().float().square().mean().sqrt())
            diagnostics['backprojection_l2'].append(back.detach().float().square().mean().sqrt())
            diagnostics['observation_l2'].append(observation.detach().float().square().mean().sqrt())
            diagnostics['state_l2'].append(state.detach().float().square().mean().sqrt())
            diagnostics['state_delta_l2'].append(state_delta.detach().float().square().mean().sqrt())
            diagnostics['feature_delta_l2'].append(delta_feat.detach().float().square().mean().sqrt())
            diagnostics['q_delta_l2'].append(delta_q.detach().float().square().mean().sqrt())
            diagnostics['gate_mean'].append(gate.detach().float().mean())
            diagnostics['gate_std'].append(gate.detach().float().std())
            diagnostics['gate_saturation'].append(((gate < .01) | (gate > .99)).float().mean())
            ratio = next_residual.detach().float().square().mean().sqrt() / residual.detach().float().square().mean().sqrt().clamp_min(1e-8)
            diagnostics['data_consistency_ratio'].append(ratio)
        self.last_diagnostics = {key: torch.stack(value) for key, value in diagnostics.items()}
        return self._final_decode(x, x0, feat, q)


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 4
    return Net(scale=scale)
