#!/usr/bin/env python3
"""Inspect whether a trained RDSM router learns and changes LFMN features."""
import argparse
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = PROJECT_ROOT / 'LFMN'
sys.path.insert(0, str(LFMN_ROOT))

from loss.rdsm import target_residual  # noqa: E402
from model.lfmnrdsm import Net  # noqa: E402


class Moments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.square = 0.0

    def add(self, value):
        value = value.detach().double()
        self.count += value.numel()
        self.total += value.sum().item()
        self.square += value.square().sum().item()

    @property
    def mean(self):
        return self.total / self.count

    @property
    def std(self):
        return math.sqrt(max(self.square / self.count - self.mean ** 2, 0.0))


class PairMoments:
    def __init__(self):
        self.count = 0
        self.x = self.y = self.xx = self.yy = self.xy = 0.0
        self.absolute_error = 0.0

    def add(self, x, y):
        x = x.detach().double().flatten()
        y = y.detach().double().flatten()
        self.count += x.numel()
        self.x += x.sum().item()
        self.y += y.sum().item()
        self.xx += x.square().sum().item()
        self.yy += y.square().sum().item()
        self.xy += (x * y).sum().item()
        self.absolute_error += (x - y).abs().sum().item()

    @property
    def mae(self):
        return self.absolute_error / self.count

    @property
    def correlation(self):
        numerator = self.xy - self.x * self.y / self.count
        x_energy = self.xx - self.x ** 2 / self.count
        y_energy = self.yy - self.y ** 2 / self.count
        denominator = math.sqrt(max(x_energy * y_energy, 0.0))
        return numerator / denominator if denominator > 0 else float('nan')


def image_tensor(path, device):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = image[..., None].repeat(3, axis=2)
    image = image[..., :3]
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)


def resolve_checkpoint(run_dir, epoch):
    model_dir = run_dir / 'model'
    if epoch is not None:
        checkpoint = model_dir / 'model_{}.pt'.format(epoch)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint
    candidates = []
    for checkpoint in model_dir.glob('model_*.pt'):
        try:
            candidates.append((int(checkpoint.stem.rsplit('_', 1)[1]), checkpoint))
        except ValueError:
            pass
    if not candidates:
        raise FileNotFoundError('no model_*.pt under {}'.format(model_dir))
    return max(candidates)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=PROJECT_ROOT / 'datasets')
    parser.add_argument('--start', type=int, default=801)
    parser.add_argument('--end', type=int, default=810)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    if args.end < args.start:
        raise ValueError('--end must be at least --start')
    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(args.run_dir.resolve(), args.epoch)
    model = Net(scale=4, use_auxiliary=True).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    stochastic = [
        module for module in model.modules()
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout))
    ]
    if stochastic:
        raise RuntimeError('diagnostic train-mode forward found BN/dropout modules')
    model.train()  # RDSM returns its training-only probe outputs in this mode.

    n_stage = len(model.blocks)
    demands = [Moments() for _ in range(n_stage)]
    oracles = [Moments() for _ in range(n_stage)]
    pairs = [PairMoments() for _ in range(n_stage)]
    strength = [Moments() for _ in range(n_stage)]
    applied = [Moments() for _ in range(n_stage)]
    original = [Moments() for _ in range(n_stage)]
    previous_features = []
    modulations = [[] for _ in range(n_stage)]

    def capture_previous(_module, inputs, _output):
        previous_features.append(inputs[0][:, -48:])

    handles = [model.rdsm.reduce.register_forward_hook(capture_previous)]
    for stage, sfml in enumerate(model.sfmls):
        def capture_modulation(_module, _inputs, output, stage_index=stage):
            modulations[stage_index].append(output)
        handles.append(sfml.register_forward_hook(capture_modulation))

    valid_hr = args.data_root / 'DIV2K' / 'DIV2K_valid_HR'
    valid_lr = args.data_root / 'DIV2K' / 'DIV2K_valid_LR_bicubic' / 'X4'
    try:
        with torch.inference_mode():
            for image_id in range(args.start, args.end + 1):
                previous_features.clear()
                for values in modulations:
                    values.clear()
                stem = '{:04d}'.format(image_id)
                lr = image_tensor(valid_lr / '{}x4.png'.format(stem), device)
                hr = image_tensor(valid_hr / '{}.png'.format(stem), device)
                _, auxiliary = model(lr)
                residual_target = target_residual(auxiliary, hr)
                if len(previous_features) != n_stage:
                    raise RuntimeError('failed to capture all stage inputs')
                for stage in range(n_stage):
                    demand = auxiliary['demand_predictions'][stage]
                    residual = auxiliary['residual_predictions'][stage]
                    remaining = (residual_target - residual).abs().mean(1, keepdim=True)
                    mean = remaining.flatten(2).mean(2).view(-1, 1, 1, 1)
                    oracle = (remaining / (2.0 * mean + 1e-6)).clamp(0.0, 1.0)
                    gain = model.rdsm.modulation_scale * torch.tanh(
                        model.rdsm.stage_gain[stage]
                    )
                    delta = gain * (2.0 * demand - 1.0)
                    beta, gamma = modulations[stage][0]
                    prev = previous_features[stage]
                    modulation_residual = beta * prev + gamma - prev
                    correction = delta * modulation_residual

                    demands[stage].add(demand)
                    oracles[stage].add(oracle)
                    pairs[stage].add(demand, oracle)
                    strength[stage].add(delta.abs())
                    applied[stage].add(correction.abs())
                    original[stage].add(modulation_residual.abs())
    finally:
        for handle in handles:
            handle.remove()

    gains = model.rdsm.stage_gain.detach().cpu()
    bounded = model.rdsm.modulation_scale * torch.tanh(gains)
    print('checkpoint: {}'.format(checkpoint))
    print('images: {:04d}-{:04d}'.format(args.start, args.end))
    print('| stage | raw gain | bounded gain | demand mean/std | oracle mean/std | demand MAE | corr | mean |strength-1| | applied/original |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for stage in range(n_stage):
        ratio = applied[stage].mean / max(original[stage].mean, 1e-12)
        print('| {} | {:+.6f} | {:+.6f} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:.4f} | {:+.4f} | {:.6f} | {:.6f} |'.format(
            stage + 1, gains[stage].item(), bounded[stage].item(),
            demands[stage].mean, demands[stage].std,
            oracles[stage].mean, oracles[stage].std,
            pairs[stage].mae, pairs[stage].correlation,
            strength[stage].mean, ratio,
        ))


if __name__ == '__main__':
    main()
