#!/usr/bin/env python3
"""N11/SRPRv1 structural, adjoint, gradient, and reload checks."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = ROOT / 'LFMN'
sys.path.insert(0, str(LFMN_ROOT))
from model.lfmnsrprv1 import Net, PairedBicubic


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def check_adjoint():
    op = PairedBicubic(4)
    x = torch.randn(1, 3, 64, 64)
    r = torch.randn(1, 3, 16, 16)
    lhs = (op.down(x) * r).sum()
    rhs = (x * op.adjoint(r, (64, 64))).sum()
    error = float((lhs - rhs).abs())
    if error > 1e-5:
        raise AssertionError(f'adjoint error too large: {error}')
    return error


def main():
    torch.manual_seed(1)
    model = Net(scale=4).eval()
    params = count_parameters(model)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    if tuple(y.shape) != (1, 3, 256, 256):
        raise AssertionError(tuple(y.shape))
    if not torch.isfinite(y).all():
        raise AssertionError('nonfinite output')

    baseline = Net(scale=4).eval()
    baseline.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        y2 = baseline(x)
    if not torch.equal(y, y2):
        raise AssertionError('state reload changed output')

    train_model = Net(scale=4).train()
    train_x = torch.rand(1, 3, 32, 32)
    loss = train_model(train_x).abs().mean()
    loss.backward()
    new_grads = [p.grad for n, p in train_model.named_parameters()
                 if n.startswith('proximal.') and p.requires_grad]
    if not new_grads or not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in new_grads):
        raise AssertionError('SRPRv1 parameters did not receive finite nonzero gradients')

    payload = {
        'candidate': 'N11/SRPRv1',
        'parameters': params,
        'input_shape': list(x.shape),
        'output_shape': list(y.shape),
        'adjoint_abs_error': check_adjoint(),
        'gradient_check': 'passed',
        'reload_check': 'passed',
        'source_commit': __import__('subprocess').check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
