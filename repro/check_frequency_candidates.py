#!/usr/bin/env python3
"""Validate frequency candidate, overlap control, loss, and optimizer groups."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = PROJECT_ROOT / 'LFMN'
sys.path.insert(0, str(LFMN_ROOT))

from loss.high_frequency import HighFrequencyL1Loss  # noqa: E402
from model.lfmn import Net as BaselineNet, patch_divide, patch_reverse  # noqa: E402
from model.lfmnfreq import Net as FrequencyNet  # noqa: E402
from model.lfmnoverlap import Net as OverlapNet  # noqa: E402
import utility  # noqa: E402


def optimizer_args(**overrides):
    values = dict(
        lr=1e-5,
        feedback_lr_mult=1.0,
        freq_lr_mult=10.0,
        weight_decay=0,
        optimizer='ADAM',
        betas=(0.9, 0.999),
        epsilon=1e-8,
        decay='200-400-600-800',
        gamma=0.5,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseline = BaselineNet(scale=4)
    frequency = FrequencyNet(scale=4)
    overlap = OverlapNet(scale=4)
    checkpoint = LFMN_ROOT / 'model' / 'scale4_model_939.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    baseline.load_state_dict(state, strict=True)
    overlap.load_state_dict(state, strict=True)
    incompatible = frequency.load_state_dict(state, strict=False)
    expected_missing = {
        key for key in frequency.state_dict() if key.startswith('freq_prior.')
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError('unexpected frequency missing keys')
    if incompatible.unexpected_keys:
        raise RuntimeError('unexpected frequency checkpoint keys')

    baseline = baseline.to(device).eval()
    frequency = frequency.to(device).eval()
    sample = torch.rand(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        baseline_output = baseline(sample)
        frequency_output = frequency(sample)
    max_abs_difference = (baseline_output - frequency_output).abs().max().item()
    if max_abs_difference != 0.0:
        raise RuntimeError('frequency candidate changed its baseline initialization')

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    frequency_params = sum(parameter.numel() for parameter in frequency.parameters())
    added_params = frequency_params - baseline_params
    added_percent = 100.0 * added_params / baseline_params
    if added_percent > 1.0:
        raise RuntimeError('frequency candidate exceeds 1% parameter budget')

    frequency.train()
    frequency.zero_grad(set_to_none=True)
    frequency(sample).mean().backward()
    gate_gradient = frequency.freq_prior.stage_gates.grad.abs().sum().item()
    if gate_gradient == 0.0:
        raise RuntimeError('zero-initialized frequency gates receive no gradient')

    optimizer = utility.make_optimizer(optimizer_args(), frequency)
    group_lrs = [group['lr'] for group in optimizer.param_groups]
    group_params = [
        sum(parameter.numel() for parameter in group['params'])
        for group in optimizer.param_groups
    ]
    if group_lrs != [1e-5, 1e-4]:
        raise RuntimeError('unexpected frequency optimizer learning rates')
    if group_params != [baseline_params, added_params]:
        raise RuntimeError('unexpected frequency optimizer parameter groups')

    identity = torch.ones(1, 2, 57, 73, device=device)
    patches, _, _ = patch_divide(identity, step=14, ps=16)
    legacy = patch_reverse(patches, identity, step=14, ps=16)
    normalized = patch_reverse(
        patches, identity, step=14, ps=16, normalize_overlap=True
    )
    legacy_error = (legacy - identity).abs().max().item()
    normalized_error = (normalized - identity).abs().max().item()
    if legacy_error == 0.0 or normalized_error != 0.0:
        raise RuntimeError('overlap normalization identity check failed')

    loss_fn = HighFrequencyL1Loss().to(device)
    target = torch.rand(2, 3, 24, 24, device=device)
    if loss_fn(target, target).item() != 0.0:
        raise RuntimeError('HFL1 is nonzero for identical inputs')
    prediction = target.detach().clone().requires_grad_(True)
    prediction.data[:, :, 8:16, 8:16] += 0.1
    hf_loss = loss_fn(prediction, target)
    hf_loss.backward()
    if hf_loss.item() <= 0.0 or prediction.grad.abs().sum().item() == 0.0:
        raise RuntimeError('HFL1 positive/gradient check failed')

    print('device: {}'.format(device))
    print('baseline parameters: {:,}'.format(baseline_params))
    print('frequency parameters: {:,}'.format(frequency_params))
    print('added parameters: {:,} ({:.3f}%)'.format(added_params, added_percent))
    print('initial max output difference: {:.1f}'.format(max_abs_difference))
    print('frequency gate gradient sum: {:.6f}'.format(gate_gradient))
    print('optimizer group learning rates: {}'.format(group_lrs))
    print('legacy/normalized identity error: {:.3f}/{:.1f}'.format(
        legacy_error, normalized_error
    ))
    print('HFL1 positive value: {:.6f}'.format(hf_loss.item()))
    print('frequency candidate validation passed')


if __name__ == '__main__':
    main()
