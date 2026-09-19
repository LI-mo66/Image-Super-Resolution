#!/usr/bin/env python3
"""Validate the N7 cross-axis LRSA replacement before training."""
import io
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmncrossaxis import CrossAxisDepthwiseConv, Net as CandidateNet
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
    baseline = BaselineNet(scale=4)
    candidate = CandidateNet(scale=4)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    reduction = baseline_params - candidate_params
    require(reduction == 10752, 'unexpected parameter reduction')

    mixers = [
        local_attn.layer[1].fn.dwconv
        for _, local_attn in candidate.blocks
    ]
    require(len(mixers) == 8, 'expected eight LRSA mixers')
    require(all(isinstance(mixer, CrossAxisDepthwiseConv) for mixer in mixers),
            'not every LRSA mixer was replaced')

    candidate = candidate.to(device)
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    candidate.eval()
    with torch.inference_mode():
        enabled = candidate(x)
        for mixer in mixers:
            mixer.enabled = False
        disabled = candidate(x)
        for mixer in mixers:
            mixer.enabled = True
    require(enabled.shape == (1, 3, 128, 128), 'unexpected output shape')
    require(torch.isfinite(enabled).all(), 'nonfinite candidate output')
    require(not torch.equal(enabled, disabled),
            'cross-axis replacement has no causal output effect')

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    F.l1_loss(candidate(x), torch.rand_like(enabled)).backward()
    gradient_sums = []
    for index, mixer in enumerate(mixers, 1):
        for name, parameter in mixer.named_parameters():
            require(parameter.grad is not None,
                    f'stage {index} {name} gradient missing')
            value = parameter.grad.abs().sum().item()
            require(value > 0 and torch.isfinite(parameter.grad).all(),
                    f'stage {index} {name} gradient invalid')
            gradient_sums.append(value)

    optimizer = utility.make_optimizer(optimizer_args(), candidate)
    require(optimizer.scheduler.T_max == 150,
            'cosine horizon does not match the 150-epoch baseline')
    require(abs(optimizer.get_lr() - 2e-4) < 1e-12,
            'unexpected initial learning rate')

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
    print('parameter reduction: {} ({:.4f}%)'.format(
        reduction, 100.0 * reduction / baseline_params
    ))
    print('enabled/disabled max difference:',
          (enabled - disabled).abs().max().item())
    print('cross-axis gradient min/max:',
          min(gradient_sums), max(gradient_sums))
    print('cosine scheduler T_max:', optimizer.scheduler.T_max)
    print('replacement, causality, gradients, scheduler, reload: OK')
    print('Cross-axis validation passed (not a performance result)')


if __name__ == '__main__':
    main()
