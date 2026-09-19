#!/usr/bin/env python3
"""Validate cross-window directional exchange before performance screening."""
from pathlib import Path
from types import SimpleNamespace
import io
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmncrosswindow import Net as CandidateNet
import utility


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def optimizer_args():
    return SimpleNamespace(
        lr=1e-5, weight_decay=0, optimizer='ADAM', betas=(0.9, 0.999),
        epsilon=1e-8, decay='200-400-600-800', gamma=0.5,
        scheduler='multistep', epochs=5, feedback_lr_mult=1,
        freq_lr_mult=1, rdsm_lr_mult=1, stage_diff_lr_mult=1,
        cross_window_lr_mult=10,
    )


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = BaselineNet(scale=4)
    candidate = CandidateNet(scale=4)
    state = torch.load(
        ROOT / 'LFMN/model/scale4_model_939.pt',
        map_location='cpu', weights_only=True,
    )
    baseline.load_state_dict(state, strict=True)
    result = candidate.load_state_dict(state, strict=False)
    expected_missing = {
        'cross_window.local.weight',
        'cross_window.horizontal.weight',
        'cross_window.vertical.weight',
        'cross_window.fuse.weight',
        'cross_window.stage_gain',
    }
    require(set(result.missing_keys) == expected_missing,
            'baseline checkpoint mapping mismatch')
    require(not result.unexpected_keys, 'unexpected baseline checkpoint keys')

    base_params = sum(p.numel() for p in baseline.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    added = candidate_params - base_params
    require(added == 6480, 'unexpected cross-window parameter count')

    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    with torch.inference_mode():
        baseline_output = baseline(x)
        enabled_output = candidate(x)
        candidate.cross_window.enabled = False
        disabled_output = candidate(x)
        candidate.cross_window.enabled = True
    require(torch.equal(baseline_output, enabled_output),
            'zero-start candidate differs from baseline')
    require(torch.equal(baseline_output, disabled_output),
            'disabled candidate differs from baseline')

    candidate.train()
    optimizer = utility.make_optimizer(optimizer_args(), candidate)
    require([group['lr'] for group in optimizer.param_groups] == [1e-5, 1e-4],
            'cross-window optimizer learning rates are incorrect')
    optimizer.zero_grad(set_to_none=True)
    F.l1_loss(candidate(x), torch.rand_like(baseline_output)).backward()
    gate = candidate.cross_window.stage_gain
    require(gate.grad is not None and torch.isfinite(gate.grad).all(),
            'stage-gain gradient is missing or nonfinite')
    require(gate.grad.abs().sum().item() > 0,
            'stage gain has zero first-step gradient')
    first_gate_gradient = gate.grad.abs().sum().item()
    optimizer.step()
    require(gate.abs().sum().item() > 0,
            'first optimizer step did not update stage gain')

    optimizer.zero_grad(set_to_none=True)
    F.l1_loss(candidate(x), torch.rand_like(baseline_output)).backward()
    mixer_gradients = [
        parameter.grad.abs().sum().item()
        for name, parameter in candidate.cross_window.named_parameters()
        if name != 'stage_gain' and parameter.grad is not None
    ]
    require(len(mixer_gradients) == 4 and all(v > 0 for v in mixer_gradients),
            'directional mixer did not receive second-step gradients')

    candidate.eval()
    with torch.inference_mode():
        learned_output = candidate(x)
        candidate.cross_window.enabled = False
        learned_disabled = candidate(x)
        candidate.cross_window.enabled = True
    require(not torch.equal(learned_output, learned_disabled),
            'learned cross-window path has no causal output effect')

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
                'checkpoint reload changed candidate output')

    print('device:', device)
    print('initial max output difference:',
          (baseline_output - enabled_output).abs().max().item())
    print('added parameters: {} ({:.4f}%)'.format(
        added, 100.0 * added / base_params
    ))
    print('first-step stage-gain gradient sum:', first_gate_gradient)
    print('second-step mixer gradient sums:', mixer_gradients)
    print('optimizer group learning rates:',
          [group['lr'] for group in optimizer.param_groups])
    print('identity, ON/OFF causality, gradients, optimizer, reload: OK')
    print('Cross-window validation passed (not a performance result)')


if __name__ == '__main__':
    main()
