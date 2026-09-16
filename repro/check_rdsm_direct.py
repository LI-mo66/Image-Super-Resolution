#!/usr/bin/env python3
"""Validate RDSM-v2 identity initialization, gradients, and parameter budget."""
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
from model.lfmnrdsmdirect import Net as DirectNet  # noqa: E402
import utility  # noqa: E402


def optimizer_args():
    return SimpleNamespace(
        lr=1e-5, feedback_lr_mult=1.0, freq_lr_mult=1.0,
        rdsm_lr_mult=10.0, weight_decay=0, optimizer='ADAM',
        betas=(0.9, 0.999), epsilon=1e-8,
        decay='200-400-600-800', gamma=0.5,
    )


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = BaselineNet(scale=4)
    direct = DirectNet(scale=4, use_auxiliary=True)
    state = torch.load(
        LFMN_ROOT / 'model' / 'scale4_model_939.pt',
        map_location='cpu', weights_only=True
    )
    baseline.load_state_dict(state, strict=True)
    incompatible = direct.load_state_dict(state, strict=False)
    expected_missing = {key for key in direct.state_dict() if key.startswith('rdsm.')}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError('unexpected checkpoint compatibility result')

    baseline_params = sum(p.numel() for p in baseline.parameters())
    direct_params = sum(p.numel() for p in direct.parameters())
    added_params = direct_params - baseline_params
    probe_params = sum(p.numel() for p in direct.rdsm.residual_probe.parameters())
    if 100.0 * added_params / baseline_params > 1.0:
        raise RuntimeError('RDSM-v2 exceeds 1% training-parameter budget')

    baseline = baseline.to(device).eval()
    direct = direct.to(device).eval()
    lr = torch.rand(1, 3, 32, 32, device=device) * 255.0
    with torch.inference_mode():
        baseline_output = baseline(lr)
        direct_output = direct(lr)
        demand = direct.rdsm.demand(
            direct.fea(lr), direct.first_conv(lr), 0
        )
    max_difference = (baseline_output - direct_output).abs().max().item()
    if max_difference != 0.0 or not torch.equal(demand, torch.full_like(demand, 0.5)):
        raise RuntimeError('RDSM-v2 must start as the exact baseline with demand=0.5')

    direct.train()
    direct.zero_grad(set_to_none=True)
    hr = torch.rand(1, 3, 128, 128, device=device) * 255.0
    output = direct(lr)
    sr, _ = output
    l1 = F.l1_loss(sr, hr)
    residual = ResidualProbeLoss()(output, hr)
    demand_loss = DemandDistillationLoss()(output, hr)
    (l1 + 0.05 * residual + 0.05 * demand_loss).backward()
    router_gradient = direct.rdsm.predict.weight.grad
    if router_gradient is None or router_gradient.abs().sum().item() == 0.0:
        raise RuntimeError('direct router receives no first-step gradient')

    optimizer = utility.make_optimizer(optimizer_args(), direct)
    group_lrs = [group['lr'] for group in optimizer.param_groups]
    group_params = [sum(p.numel() for p in group['params']) for group in optimizer.param_groups]
    if group_lrs != [1e-5, 1e-4] or group_params != [baseline_params, added_params]:
        raise RuntimeError('unexpected RDSM-v2 optimizer groups')

    print('device: {}'.format(device))
    print('initial max output difference: {:.1f}'.format(max_difference))
    print('initial demand mean: {:.1f}'.format(demand.mean().item()))
    print('router first-step gradient sum: {:.6f}'.format(router_gradient.abs().sum().item()))
    print('added training parameters: {:,} ({:.3f}%)'.format(
        added_params, 100.0 * added_params / baseline_params
    ))
    print('added inference parameters: {:,} ({:.3f}%)'.format(
        added_params - probe_params,
        100.0 * (added_params - probe_params) / baseline_params
    ))
    print('optimizer group learning rates: {}'.format(group_lrs))
    print('RDSM-v2 validation passed')


if __name__ == '__main__':
    main()
