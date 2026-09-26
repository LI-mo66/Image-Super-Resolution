#!/usr/bin/env python3
"""Offline structural, KD-gradient, identity, and reload checks for N16."""
import io
import json
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LFMN_ROOT = os.path.join(ROOT, 'LFMN')
if LFMN_ROOT not in sys.path:
    sys.path.insert(0, LFMN_ROOT)

from model.lfmn import Net as BaselineNet
from model.lfmnsrprv2 import BicubicAdjoint, Net as SRPRv2Net
from rgcrd.criterion import RGCRDCriterion


def criterion_args():
    return SimpleNamespace(
        rgcrd_mode='output', scale=[4], rgcrd_lambda_output=0.1,
        rgcrd_lambda_rel=0.25, rgcrd_lambda_evo=0.125,
        rgcrd_local_windows='4+8', rgcrd_global_grid=8,
        rgcrd_reliability_pixel=0.5, rgcrd_reliability_low=0.25,
        rgcrd_reliability_grad=0.25,
        rgcrd_reliability_temperature=0.25,
        rgcrd_reliability_margin=0.0,
        rgcrd_reliability_smooth=3,
    )


def nonzero_finite(gradient):
    return (
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
    )


def main():
    torch.manual_seed(7)
    baseline = BaselineNet(scale=4).eval()
    candidate = SRPRv2Net(scale=4).eval()
    missing, unexpected = candidate.load_state_dict(
        baseline.state_dict(), strict=False
    )
    assert not unexpected
    assert missing and all(
        name.startswith(('proximal.', 'stage_probe.')) for name in missing
    )
    baseline_params = sum(p.numel() for p in baseline.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    assert baseline_params == 759627, baseline_params
    assert candidate_params == 841563, candidate_params

    sample = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        baseline_output = baseline(sample)
        candidate_output = candidate(sample)
    torch.testing.assert_close(
        candidate_output, baseline_output, rtol=0, atol=0
    )

    operator = BicubicAdjoint(4)
    hr_probe = torch.randn(1, 3, 64, 64, dtype=torch.float64)
    lr_probe = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    adjoint_error = abs(float(
        (operator.down(hr_probe) * lr_probe).sum()
        - (hr_probe * operator.adjoint(lr_probe, (64, 64))).sum()
    ))
    assert adjoint_error <= 1e-9, adjoint_error

    criterion = RGCRDCriterion(criterion_args())
    assert sum(p.numel() for p in criterion.parameters()) == 0
    train_model = SRPRv2Net(scale=4).train()
    optimizer = torch.optim.SGD(train_model.parameters(), lr=1e-3)
    teacher = torch.rand(1, 3, 64, 64) * 255
    target = torch.rand(1, 3, 64, 64) * 255
    train_input = torch.rand(1, 3, 16, 16) * 255

    optimizer.zero_grad(set_to_none=True)
    student = train_model(train_input)
    kd_loss, stats = criterion(student, None, teacher, None, target)
    total = F.l1_loss(student, target) + kd_loss
    total.backward()
    named = dict(train_model.named_parameters())
    assert nonzero_finite(named['proximal.0.writeback.2.weight'].grad)
    assert all(torch.isfinite(value) for value in stats.values())
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    student = train_model(train_input)
    kd_loss, _ = criterion(student, None, teacher, None, target)
    (F.l1_loss(student, target) + kd_loss).backward()
    delayed_paths = (
        'stage_probe.weight',
        'proximal.0.candidate.4.weight',
        'proximal.0.gate.weight',
    )
    assert all(nonzero_finite(named[name].grad) for name in delayed_paths)

    train_model.eval()
    with torch.no_grad():
        reference = train_model(sample)
    buffer = io.BytesIO()
    torch.save(train_model.state_dict(), buffer)
    buffer.seek(0)
    restored = SRPRv2Net(scale=4).eval()
    restored.load_state_dict(
        torch.load(buffer, map_location='cpu', weights_only=True), strict=True
    )
    with torch.no_grad():
        restored_output = restored(sample)
    torch.testing.assert_close(restored_output, reference, rtol=0, atol=0)

    print(json.dumps({
        'candidate': 'N16/SRPRv2+C1 interaction',
        'baseline_parameters': baseline_params,
        'srprv2_parameters': candidate_params,
        'initial_baseline_identity': 'passed',
        'output_kd_parameter_count': 0,
        'output_kd_first_step_writeback_gradient': 'passed',
        'delayed_state_path_gradients': 'passed',
        'adjoint_abs_error_float64': adjoint_error,
        'save_reload': 'passed',
    }, indent=2))


if __name__ == '__main__':
    main()
