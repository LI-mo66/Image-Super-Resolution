"""N15 SRPR-SS: semi-shared state-routed proximal reconstruction.

The LFMN backbone and decoder are unchanged.  A single proximal dynamics core
is reused at all eight stages; zero-initialized channel-wise modulation keeps
stage capacity without duplicating the complete core.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfmn import Net as BaselineNet


def _cubic(x):
    ax = x.abs()
    return torch.where(
        ax <= 1,
        1.5 * ax**3 - 2.5 * ax**2 + 1,
        torch.where(
            ax <= 2,
            -0.5 * ax**3 + 2.5 * ax**2 - 4 * ax + 2,
            torch.zeros_like(ax),
        ),
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
            raise ValueError('SRPR-SS currently requires scale=4')
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
        _, _, height, width = x.shape
        ah, aw = self._matrices(height, width, x.device, x.dtype)
        return torch.einsum('oh,bchw,pw->bcop', ah, x, aw)

    def adjoint(self, residual, output_size):
        height, width = output_size
        ah, aw = self._matrices(height, width, residual.device, residual.dtype)
        return torch.einsum('ho,bcop,wp->bchw', ah.t(), residual, aw.t())


class SharedProximalCore(nn.Module):
    """One reconstruction dynamics core reused by every backbone stage."""

    def __init__(self, channels=48, hidden=16):
        super().__init__()
        inputs = channels * 2 + 3
        self.candidate = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.gate = nn.Conv2d(inputs, channels, 1)
        self.writeback = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.q_update = nn.Conv2d(channels + 3, 3, 1)
        # RGB-range inputs make a random 1x1 gate heavily saturated before
        # learning starts.  A neutral 0.5 update is both non-saturated and
        # identical across stages; stage specialization is learned thereafter.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        # Loading a B0 state_dict must initially recover the exact B0 mapping.
        nn.init.zeros_(self.writeback[-1].weight)
        nn.init.zeros_(self.writeback[-1].bias)
        nn.init.zeros_(self.q_update.weight)
        nn.init.zeros_(self.q_update.bias)


class Net(BaselineNet):
    """LFMN with persistent reconstruction state and semi-shared updates."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8,
                 normalize_overlap=False):
        super().__init__(scale, n_feats, side_c, n_stage, normalize_overlap)
        if int(scale) != 4:
            raise ValueError('SRPR-SS requires scale=4')
        if int(n_stage) != 8:
            raise ValueError('SRPR-SS is pre-registered for the eight-stage LFMN')

        # Candidate initialization must not advance the global RNG beyond B0;
        # this preserves the subsequent data-loader random stream in paired runs.
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.observation = BicubicAdjoint(scale)
            self.state_channels = n_feats
            self.proximal_core = SharedProximalCore(n_feats, hidden=16)
            self.stage_probe = nn.Conv2d(n_feats, 3 * scale * scale, 1)
            shape = (n_stage, n_feats, 1, 1)
            self.stage_candidate_scale = nn.Parameter(torch.zeros(shape))
            self.stage_candidate_bias = nn.Parameter(torch.zeros(shape))
            self.stage_gate_bias = nn.Parameter(torch.zeros(shape))
            self.stage_writeback_scale = nn.Parameter(torch.zeros(shape))
            self.stage_q_scale = nn.Parameter(torch.zeros(n_stage, 3, 1, 1))
        finally:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        self.last_diagnostics = {}
        self.diagnostics_enabled = False

    def set_diagnostics_enabled(self, enabled=True):
        """Enable sampled telemetry; disabled inference has no reduction cost."""
        self.diagnostics_enabled = bool(enabled)

    def diagnostics_snapshot(self):
        return {
            key: value.detach().float().cpu().clone()
            for key, value in self.last_diagnostics.items()
        }

    def _final_decode(self, x, x0, feat, q):
        u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
        u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        base = F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        q_hr = F.interpolate(
            q, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        return self.last_conv(u) + base + q_hr

    def _observation_estimate(self, x, feat, q):
        base = F.interpolate(
            x + q, scale_factor=4, mode='bilinear', align_corners=False
        )
        return base + F.pixel_shuffle(self.stage_probe(feat), 4)

    def _proximal_step(self, stage, feat, state, observation):
        joined = torch.cat((feat, state, observation), dim=1)
        candidate = self.proximal_core.candidate(joined)
        candidate = candidate * (
            1 + torch.tanh(self.stage_candidate_scale[stage])
        ) + self.stage_candidate_bias[stage]
        gate = torch.sigmoid(
            self.proximal_core.gate(joined) + self.stage_gate_bias[stage]
        )
        next_state = state + gate * (candidate - state)
        correction_input = torch.cat((feat, next_state, observation), dim=1)
        delta_feat = self.proximal_core.writeback(correction_input)
        delta_feat = delta_feat * (
            1 + torch.tanh(self.stage_writeback_scale[stage])
        )
        delta_q = self.proximal_core.q_update(
            torch.cat((next_state, observation), dim=1)
        )
        delta_q = delta_q * (1 + torch.tanh(self.stage_q_scale[stage]))
        return next_state, delta_feat, delta_q, gate

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        state = torch.zeros_like(feat)
        q = torch.zeros(
            x.shape[0], 3, x.shape[2], x.shape[3],
            device=x.device, dtype=x.dtype,
        )
        diagnostics = None
        if self.diagnostics_enabled:
            diagnostics = {key: [] for key in (
                'residual_l2', 'backprojection_l2', 'observation_l2',
                'state_l2', 'state_delta_l2', 'feature_delta_l2',
                'q_delta_l2', 'gate_mean', 'gate_std', 'gate_saturation',
                'data_consistency_ratio', 'candidate_modulation_l2',
                'writeback_modulation_l2', 'q_modulation_l2',
            )}
        for stage in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[stage](fs)
            fm = beta * prev + gamma
            g_attn, l_attn = self.blocks[stage]
            t = g_attn(fm)
            s = l_attn(t, self.patch_size[stage])
            feat = self.esas[stage](prev + self.mid_convs[stage](s))

            x_hat = self._observation_estimate(x, feat, q)
            residual = x - self.observation.down(x_hat)
            back = self.observation.adjoint(residual, x_hat.shape[-2:])
            observation = F.interpolate(
                back, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
            state_before = state if diagnostics is not None else None
            state, delta_feat, delta_q, gate = self._proximal_step(
                stage, feat, state, observation
            )
            feat = feat + delta_feat
            q = q + delta_q

            if diagnostics is not None:
                state_delta = state - state_before
                next_x = self._observation_estimate(x, feat, q)
                next_residual = x - self.observation.down(next_x)
                def rms(value):
                    return value.detach().float().square().mean().sqrt()

                diagnostics['residual_l2'].append(rms(residual))
                diagnostics['backprojection_l2'].append(rms(back))
                diagnostics['observation_l2'].append(rms(observation))
                diagnostics['state_l2'].append(rms(state))
                diagnostics['state_delta_l2'].append(rms(state_delta))
                diagnostics['feature_delta_l2'].append(rms(delta_feat))
                diagnostics['q_delta_l2'].append(rms(delta_q))
                diagnostics['gate_mean'].append(gate.detach().float().mean())
                diagnostics['gate_std'].append(gate.detach().float().std())
                diagnostics['gate_saturation'].append(
                    ((gate < .01) | (gate > .99)).float().mean()
                )
                diagnostics['data_consistency_ratio'].append(
                    rms(next_residual) / rms(residual).clamp_min(1e-8)
                )
                diagnostics['candidate_modulation_l2'].append(
                    rms(self.stage_candidate_scale[stage])
                )
                diagnostics['writeback_modulation_l2'].append(
                    rms(self.stage_writeback_scale[stage])
                )
                diagnostics['q_modulation_l2'].append(
                    rms(self.stage_q_scale[stage])
                )
        self.last_diagnostics = (
            {key: torch.stack(value) for key, value in diagnostics.items()}
            if diagnostics is not None else {}
        )
        return self._final_decode(x, x0, feat, q)


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 4
    return Net(scale=scale)
