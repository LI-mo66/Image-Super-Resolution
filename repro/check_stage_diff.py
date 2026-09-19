#!/usr/bin/env python3
"""Validate stage-difference aggregation before any performance run."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmnstagediff import Net as CandidateNet
import utility


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def optimizer_args():
    return SimpleNamespace(
        lr=1e-5, weight_decay=0, optimizer='ADAM', betas=(0.9, 0.999),
        epsilon=1e-8, decay='200-400-600-800', gamma=0.5,
        scheduler='multistep', epochs=5, feedback_lr_mult=1,
        freq_lr_mult=1, rdsm_lr_mult=1, stage_diff_lr_mult=10,
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
    require(
        set(result.missing_keys) == {'stage_diff.gates'}
        and not result.unexpected_keys,
        'baseline checkpoint mapping mismatch',
    )

    base_params = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_params = sum(parameter.numel() for parameter in candidate.parameters())
    added = candidate_params - base_params
    require(added == 7 * 48, 'unexpected stage-difference parameter count')

    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    with torch.inference_mode():
        baseline_output = baseline(x)
        enabled_output = candidate(x)
        candidate.stage_diff.enabled = False
        disabled_output = candidate(x)
        candidate.stage_diff.enabled = True
    require(torch.equal(baseline_output, enabled_output),
            'zero-start candidate differs from baseline')
    require(torch.equal(baseline_output, disabled_output),
            'disabled candidate differs from baseline')

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    output = candidate(x)
    F.l1_loss(output, torch.rand_like(output)).backward()
    gates = candidate.stage_diff.gates
    require(gates.grad is not None and torch.isfinite(gates.grad).all(),
            'stage-difference gate gradient is missing or nonfinite')
    require(gates.grad.abs().sum().item() > 0,
            'stage-difference gate has zero first-step gradient')

    optimizer = utility.make_optimizer(optimizer_args(), candidate)
    require([group['lr'] for group in optimizer.param_groups] == [1e-5, 1e-4],
            'stage-difference optimizer learning rates are incorrect')
    optimizer.step()
    require(candidate.stage_diff.gates.abs().sum().item() > 0,
            'optimizer step did not update stage-difference gates')
    candidate.eval()
    with torch.inference_mode():
        learned_output = candidate(x)
        candidate.stage_diff.enabled = False
        learned_disabled = candidate(x)
        candidate.stage_diff.enabled = True
    require(not torch.equal(learned_output, learned_disabled),
            'learned stage-difference path has no causal output effect')

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'candidate.pt'
        torch.save(candidate.state_dict(), checkpoint)
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
    print('first-step gate gradient sum:', gates.grad.abs().sum().item())
    print('optimizer group learning rates:',
          [group['lr'] for group in optimizer.param_groups])
    print('identity, ON/OFF causality, gradients, optimizer, reload: OK')
    print('Stage-difference validation passed (not a performance result)')


if __name__ == '__main__':
    main()
