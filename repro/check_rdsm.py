#!/usr/bin/env python3
"""Validate RDSM initialization, auxiliary losses, gradients, and budget."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = PROJECT_ROOT / 'LFMN'
sys.path.insert(0, str(LFMN_ROOT))

from loss.rdsm import DemandDistillationLoss, ResidualProbeLoss  # noqa: E402
from model.lfmn import Net as BaselineNet  # noqa: E402
from model.lfmnrdsm import Net as RDSMNet  # noqa: E402
import utility  # noqa: E402


def optimizer_args():
    return SimpleNamespace(
        lr=1e-5,
        feedback_lr_mult=1.0,
        freq_lr_mult=1.0,
        rdsm_lr_mult=10.0,
        weight_decay=0,
        optimizer='ADAM',
        betas=(0.9, 0.999),
        epsilon=1e-8,
        decay='200-400-600-800',
        gamma=0.5,
    )


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = BaselineNet(scale=4)
    rdsm = RDSMNet(scale=4, use_auxiliary=True)
    checkpoint = LFMN_ROOT / 'model' / 'scale4_model_939.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    baseline.load_state_dict(state, strict=True)
    incompatible = rdsm.load_state_dict(state, strict=False)
    expected_missing = {key for key in rdsm.state_dict() if key.startswith('rdsm.')}
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError('unexpected RDSM checkpoint missing keys')
    if incompatible.unexpected_keys:
        raise RuntimeError('unexpected RDSM checkpoint keys')

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    rdsm_params = sum(parameter.numel() for parameter in rdsm.parameters())
    probe_params = sum(parameter.numel() for parameter in rdsm.rdsm.residual_probe.parameters())
    added_params = rdsm_params - baseline_params
    inference_added = added_params - probe_params
    if 100.0 * added_params / baseline_params > 1.0:
        raise RuntimeError('RDSM exceeds 1% training-parameter budget')

    baseline = baseline.to(device).eval()
    rdsm = rdsm.to(device).eval()
    lr = torch.rand(1, 3, 32, 32, device=device) * 255.0
    with torch.inference_mode():
        baseline_output = baseline(lr)
        rdsm_output = rdsm(lr)
    if isinstance(rdsm_output, (tuple, list)):
        raise RuntimeError('RDSM evaluation must return only the SR tensor')
    max_abs_difference = (baseline_output - rdsm_output).abs().max().item()
    if max_abs_difference != 0.0:
        raise RuntimeError(
            'zero-initialized RDSM changed baseline output: {}'.format(
                max_abs_difference
            )
        )

    rdsm.train()
    rdsm.zero_grad(set_to_none=True)
    hr = torch.rand(1, 3, 128, 128, device=device) * 255.0
    output = rdsm(lr)
    if not isinstance(output, tuple) or len(output) != 2:
        raise RuntimeError('RDSM training must return SR and auxiliary data')
    sr, auxiliary = output
    if len(auxiliary['demand_predictions']) != 8:
        raise RuntimeError('RDSM must predict demand at all eight stages')
    residual_loss = ResidualProbeLoss()(output, hr)
    demand_loss = DemandDistillationLoss()(output, hr)
    total = F.l1_loss(sr, hr) + 0.05 * residual_loss + 0.05 * demand_loss
    total.backward()

    gradients = {
        'stage_gain': rdsm.rdsm.stage_gain.grad,
        'router': rdsm.rdsm.predict.weight.grad,
        'probe': rdsm.rdsm.residual_probe.weight.grad,
    }
    for name, gradient in gradients.items():
        if gradient is None or gradient.abs().sum().item() == 0.0:
            raise RuntimeError('{} receives no gradient'.format(name))

    optimizer = utility.make_optimizer(optimizer_args(), rdsm)
    group_lrs = [group['lr'] for group in optimizer.param_groups]
    group_params = [
        sum(parameter.numel() for parameter in group['params'])
        for group in optimizer.param_groups
    ]
    if group_lrs != [1e-5, 1e-4]:
        raise RuntimeError('unexpected RDSM optimizer learning rates')
    if group_params != [baseline_params, added_params]:
        raise RuntimeError('unexpected RDSM optimizer parameter groups')

    print('device: {}'.format(device))
    print('baseline parameters: {:,}'.format(baseline_params))
    print('RDSM training parameters: {:,}'.format(rdsm_params))
    print('added training parameters: {:,} ({:.3f}%)'.format(
        added_params, 100.0 * added_params / baseline_params
    ))
    print('training-only probe parameters: {:,}'.format(probe_params))
    print('added inference parameters: {:,} ({:.3f}%)'.format(
        inference_added, 100.0 * inference_added / baseline_params
    ))
    print('initial max output difference: {:.1f}'.format(max_abs_difference))
    print('residual/demand losses: {:.6f}/{:.6f}'.format(
        residual_loss.item(), demand_loss.item()
    ))
    print('gradient sums: {}'.format({
        name: gradient.abs().sum().item()
        for name, gradient in gradients.items()
    }))
    print('optimizer group learning rates: {}'.format(group_lrs))
    print('RDSM validation passed')


if __name__ == '__main__':
    main()
