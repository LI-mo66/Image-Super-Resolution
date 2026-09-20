#!/usr/bin/env python3
"""Offline structural checks for the training-only RGCRD implementation."""
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LFMN_ROOT = os.path.join(ROOT, 'LFMN')
if LFMN_ROOT not in sys.path:
    sys.path.insert(0, LFMN_ROOT)

from model.lfmn import Net as BaselineNet
from model.lfmnrgcrd import Net as RGCRDNet
from rgcrd.criterion import RGCRDCriterion


def criterion_args(mode='full'):
    return SimpleNamespace(
        rgcrd_mode=mode, scale=[4], rgcrd_lambda_output=0.1,
        rgcrd_lambda_rel=0.1, rgcrd_lambda_evo=0.05,
        rgcrd_local_windows='4+8', rgcrd_global_grid=4,
        rgcrd_reliability_pixel=0.5, rgcrd_reliability_low=0.25,
        rgcrd_reliability_grad=0.25,
        rgcrd_reliability_temperature=0.25,
        rgcrd_reliability_margin=0.0,
        rgcrd_reliability_smooth=3,
    )


def check_model_contract():
    torch.manual_seed(7)
    baseline = BaselineNet(scale=4)
    tapped = RGCRDNet(scale=4)
    assert baseline.state_dict().keys() == tapped.state_dict().keys()
    tapped.load_state_dict(baseline.state_dict(), strict=True)
    image = torch.rand(1, 3, 32, 32) * 255
    baseline.eval()
    tapped.eval()
    with torch.no_grad():
        expected = baseline(image)
        actual = tapped(image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    tapped.train()
    with torch.no_grad():
        sr, features = tapped(image)
    assert sr.shape == (1, 3, 128, 128)
    assert features['stage4'].shape == (1, 48, 32, 32)
    assert features['stage8'].shape == (1, 48, 32, 32)
    print('PASS model contract: identical state keys and evaluation output')


def check_criterion():
    torch.manual_seed(11)
    criterion = RGCRDCriterion(criterion_args())
    assert sum(parameter.numel() for parameter in criterion.parameters()) == 0
    hr = torch.rand(2, 3, 64, 64) * 255
    student_sr = (hr + torch.randn_like(hr) * 10).requires_grad_(True)
    teacher_sr = hr + torch.randn_like(hr) * 4
    student_features = {
        'stage4': torch.randn(2, 12, 16, 16, requires_grad=True),
        'stage8': torch.randn(2, 12, 16, 16, requires_grad=True),
    }
    teacher_features = {
        'stage4': torch.randn(2, 20, 16, 16),
        'stage8': torch.randn(2, 20, 16, 16),
    }
    loss, stats = criterion(
        student_sr, student_features, teacher_sr, teacher_features, hr
    )
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert student_features['stage4'].grad is not None
    assert student_features['stage8'].grad is not None
    assert all(torch.isfinite(value) for value in stats.values())

    better_gate, _, _ = criterion.reliability(
        hr + 20, hr, hr, (16, 16)
    )
    worse_gate, _, _ = criterion.reliability(
        hr, hr + 20, hr, (16, 16)
    )
    assert better_gate.mean() > 0.5
    assert worse_gate.mean() < 0.5
    print(
        'PASS criterion: parameter-free, finite gradients, directional gate '
        '({:.3f} > {:.3f})'.format(better_gate.mean(), worse_gate.mean())
    )


if __name__ == '__main__':
    check_model_contract()
    check_criterion()
