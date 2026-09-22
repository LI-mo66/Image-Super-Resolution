#!/usr/bin/env python3
"""N12/SRPRv2 structure and causal-path checks."""
import json
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnsrprv2 import Net, BicubicAdjoint


def main():
    torch.manual_seed(1)
    model = Net(scale=4).eval()
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (1, 3, 64, 64):
        raise AssertionError(tuple(y.shape))
    if not torch.isfinite(y).all():
        raise AssertionError('nonfinite output')
    op = BicubicAdjoint(4)
    hr = torch.randn(1, 3, 64, 80, dtype=torch.float64)
    lr = torch.randn(1, 3, 16, 20, dtype=torch.float64)
    lhs = (op.down(hr) * lr).sum()
    rhs = (hr * op.adjoint(lr, (64, 80))).sum()
    error = float((lhs - rhs).abs())
    if error > 1e-9:
        raise AssertionError(f'adjoint error {error}')
    train = Net(scale=4).train()
    train_x = torch.rand(1, 3, 16, 16)
    loss = train(train_x).abs().mean()
    loss.backward()
    grads = [p.grad for n, p in train.named_parameters() if n.startswith('proximal.')]
    if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
        raise AssertionError('proximal gradients are zero or nonfinite')
    changed = torch.zeros_like(x)
    changed[..., 2, 2] += .1
    with torch.no_grad():
        y_changed = model(x + changed)
    if torch.equal(y, y_changed):
        raise AssertionError('output does not depend on input')
    payload = {
        'candidate': 'N12/SRPRv2',
        'parameters': sum(p.numel() for p in model.parameters()),
        'input_shape': list(x.shape),
        'output_shape': list(y.shape),
        'adjoint_abs_error_float64': error,
        'gradient_check': 'passed',
        'causal_input_check': 'passed',
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
