#!/usr/bin/env python3
"""N12/SRPRv2 structure, causal-path, and reload checks."""
import json
import io
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnsrprv2 import BicubicAdjoint, Net
from model.lfmn import Net as BaselineNet
from trainer import Trainer

DIAGNOSTIC_KEYS = {
    'residual_l2', 'backprojection_l2', 'observation_l2', 'state_l2',
    'state_delta_l2', 'feature_delta_l2', 'q_delta_l2', 'gate_mean',
    'gate_std', 'gate_saturation', 'data_consistency_ratio',
}


def check_forward(model, shape):
    x = torch.rand(*shape)
    with torch.no_grad():
        y = model(x)
    expected = (shape[0], 3, shape[2] * 4, shape[3] * 4)
    assert tuple(y.shape) == expected, (shape, tuple(y.shape), expected)
    assert torch.isfinite(y).all(), 'nonfinite output'
    diagnostics = model.diagnostics_snapshot()
    assert set(diagnostics) == DIAGNOSTIC_KEYS
    assert all(tuple(value.shape) == (8,) for value in diagnostics.values())
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    return x, y, diagnostics


def check_adjoint(height, width):
    op = BicubicAdjoint(4)
    hr = torch.randn(1, 3, height, width, dtype=torch.float64)
    lr = torch.randn(1, 3, height // 4, width // 4, dtype=torch.float64)
    lhs = (op.down(hr) * lr).sum()
    rhs = (hr * op.adjoint(lr, (height, width))).sum()
    error = float((lhs - rhs).abs())
    assert error <= 1e-9, error
    return error


def main():
    torch.manual_seed(1)
    model = Net(scale=4).eval()
    baseline = BaselineNet(scale=4).eval()
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(name.startswith(('proximal.', 'stage_probe.')) for name in missing)
    with torch.no_grad():
        baseline_input = torch.rand(1, 3, 16, 16)
        assert torch.equal(model(baseline_input), baseline(baseline_input)), \
            'zero-initialized N12 path changed the baseline output'
    results = []
    for shape in ((1, 3, 16, 24), (1, 3, 16, 16), (1, 3, 32, 24)):
        x, y, diagnostics = check_forward(model, shape)
        results.append({'input_shape': list(shape), 'output_shape': list(y.shape)})

    adjoint_errors = {
        f'{height}x{width}': check_adjoint(height, width)
        for height, width in ((32, 48), (64, 80), (128, 96))
    }

    train = Net(scale=4).train()
    assert callable(getattr(Trainer, 'prepare', None)), 'trainer.prepare is missing'
    train_x = torch.rand(1, 3, 16, 16)
    loss = train(train_x).abs().mean()
    loss.backward()
    proximal_grads = [
        p.grad for name, p in train.named_parameters()
        if name.startswith('proximal.')
    ]
    assert any(
        g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        for g in proximal_grads
    ), 'proximal gradients are zero or nonfinite'
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for name, p in train.named_parameters()
        if name.startswith('proximal.') and '.writeback.2.' in name
    ), 'writeback gradients are zero'

    torch.optim.SGD(train.parameters(), lr=1e-2).step()
    train.zero_grad(set_to_none=True)
    train(train_x).abs().mean().backward()
    for name in ('stage_probe.weight', 'proximal.0.candidate.4.weight',
                 'proximal.0.gate.weight'):
        grad = dict(train.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, \
            f'no state-to-output gradient after first update: {name}'

    probe_feat = torch.randn(1, 48, 16, 16, requires_grad=True)
    probe_lr = torch.rand(1, 3, 16, 16)
    probe_q = torch.zeros_like(probe_lr)
    probe_residual = probe_lr - train.observation.down(
        train._observation_estimate(probe_lr, probe_feat, probe_q)
    )
    probe_residual.square().mean().backward()
    assert probe_feat.grad is not None and probe_feat.grad.abs().sum() > 0, \
        'stage observation residual does not depend on current feature'

    changed = torch.zeros_like(x)
    changed[..., 2, 2] += .1
    with torch.no_grad():
        y_changed = model(x + changed)
    assert not torch.equal(y, y_changed), 'output does not depend on input'

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = Net(scale=4).eval()
    reloaded.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    with torch.no_grad():
        reloaded_y = reloaded(x)
    assert torch.equal(y, reloaded_y), 'save/reload changed output'

    payload = {
        'candidate': 'N12/SRPRv2',
        'parameters': sum(p.numel() for p in model.parameters()),
        'forward_checks': results,
        'adjoint_abs_error_float64': adjoint_errors,
        'gradient_check': 'passed',
        'baseline_identity_check': 'passed',
        'causal_input_check': 'passed',
        'stage_feature_to_residual_check': 'passed',
        'state_to_output_gradient_check': 'passed',
        'save_reload_check': 'passed',
        'diagnostic_keys': sorted(DIAGNOSTIC_KEYS),
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
