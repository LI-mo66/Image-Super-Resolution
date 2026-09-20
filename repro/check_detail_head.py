#!/usr/bin/env python3
"""Validate the N8 direct detail reconstruction path before training."""
import io
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmndetailhead import DirectDetailHead, Net as CandidateNet
import utility


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def optimizer_args():
    return SimpleNamespace(
        lr=2e-4, weight_decay=0, optimizer='ADAM', betas=(0.9, 0.999),
        epsilon=1e-8, decay='200-400-600-800', gamma=0.5,
        scheduler='cosine', scheduler_t_max=150, eta_min=1e-6,
        epochs=20, feedback_lr_mult=1, freq_lr_mult=1,
        rdsm_lr_mult=1, stage_diff_lr_mult=1, cross_window_lr_mult=1,
    )


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    candidate = CandidateNet(scale=4)
    baseline = BaselineNet(scale=4)
    baseline.load_state_dict(
        {key: value for key, value in candidate.state_dict().items()
         if not key.startswith('detail_head.')},
        strict=True,
    )
    baseline_params = sum(p.numel() for p in baseline.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    addition = candidate_params - baseline_params
    require(addition > 0, 'detail path did not add parameters')
    require(addition / baseline_params < 0.01,
            'detail path exceeds the one-percent parameter budget')

    candidate = candidate.to(device)
    baseline = baseline.to(device)
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    candidate.eval()
    baseline.eval()
    with torch.inference_mode():
        enabled = candidate(x)
        candidate.detail_enabled = False
        disabled = candidate(x)
        baseline_output = baseline(x)
        candidate.detail_enabled = True
    require(torch.equal(disabled, baseline_output),
            'disabled candidate does not exactly match its baseline trunk')
    require(not torch.equal(enabled, disabled),
            'detail path has no causal output effect')
    require(enabled.shape == (1, 3, 128, 128), 'unexpected output shape')
    require(torch.isfinite(enabled).all(), 'nonfinite candidate output')

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    target = torch.rand_like(enabled) * 255
    F.l1_loss(candidate(x), target).backward()
    gradient_sums = []
    for name, parameter in candidate.detail_head.named_parameters():
        require(parameter.grad is not None, f'{name} gradient missing')
        value = parameter.grad.abs().sum().item()
        require(value > 0 and torch.isfinite(parameter.grad).all(),
                f'{name} gradient invalid')
        gradient_sums.append(value)

    constant = torch.ones(1, 8, 9, 9, device=device)
    high_pass = DirectDetailHead.high_pass(constant)
    require(torch.equal(high_pass, torch.zeros_like(high_pass)),
            'fixed high-pass does not reject a constant feature map')

    optimizer = utility.make_optimizer(optimizer_args(), candidate)
    require(optimizer.scheduler.T_max == 150,
            'cosine horizon does not match the 150-epoch baseline')

    candidate.eval()
    checkpoint = io.BytesIO()
    torch.save(candidate.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = CandidateNet(scale=4).to(device).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    with torch.inference_mode():
        require(torch.equal(candidate(x), restored(x)),
                'checkpoint reload changed output')

    print('device:', device)
    print('baseline parameters:', baseline_params)
    print('candidate parameters:', candidate_params)
    print('added parameters: {} ({:.4f}%)'.format(
        addition, 100.0 * addition / baseline_params
    ))
    print('enabled/disabled mean absolute difference:',
          (enabled - disabled).abs().mean().item())
    print('detail-head gradient min/max:',
          min(gradient_sums), max(gradient_sums))
    print('cosine scheduler T_max:', optimizer.scheduler.T_max)
    print('baseline equivalence, causality, gradients, scheduler, reload: OK')
    print('Detail-head validation passed (not a performance result)')


if __name__ == '__main__':
    main()
