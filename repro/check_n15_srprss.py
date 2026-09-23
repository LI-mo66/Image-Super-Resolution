#!/usr/bin/env python3
"""N15/SRPR-SS structure, gradient, identity, and reload checks."""
import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet
from model.lfmnsrprss import BicubicAdjoint, Net, SharedProximalCore


def check_adjoint(height, width):
    op = BicubicAdjoint(4)
    hr = torch.randn(1, 3, height, width, dtype=torch.float64)
    lr = torch.randn(1, 3, height // 4, width // 4, dtype=torch.float64)
    error = float(abs((op.down(hr) * lr).sum() -
                      (hr * op.adjoint(lr, (height, width))).sum()))
    assert error <= 1e-9, error
    return error


def load_real_pair(root, image_id='0001'):
    hr_path = root / 'DIV2K_train_HR' / f'{image_id}.png'
    lr_path = root / 'DIV2K_train_LR_bicubic' / 'X4' / f'{image_id}x4.png'
    if not hr_path.is_file() or not lr_path.is_file():
        raise FileNotFoundError(f'missing pair: {hr_path} / {lr_path}')
    hr = np.asarray(Image.open(hr_path).convert('RGB'), dtype=np.float32)
    lr = np.asarray(Image.open(lr_path).convert('RGB'), dtype=np.float32)
    height = min(lr.shape[0], 32)
    width = min(lr.shape[1], 32)
    height -= height % 4
    width -= width % 4
    lr = torch.from_numpy(lr[:height, :width]).permute(2, 0, 1)[None] / 255
    hr = torch.from_numpy(hr[:height * 4, :width * 4]).permute(2, 0, 1)[None] / 255
    return lr, hr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, help='optional DIV2K directory')
    args = parser.parse_args()

    # Candidate construction must leave the post-model RNG stream equal to B0.
    torch.manual_seed(7)
    baseline_rng_model = BaselineNet(scale=4).eval()
    baseline_rng_after = torch.random.get_rng_state()
    torch.manual_seed(7)
    candidate_rng_model = Net(scale=4).eval()
    candidate_rng_after = torch.random.get_rng_state()
    assert torch.equal(baseline_rng_after, candidate_rng_after)
    baseline_state = baseline_rng_model.state_dict()
    candidate_state = candidate_rng_model.state_dict()
    assert all(torch.equal(value, candidate_state[key])
               for key, value in baseline_state.items())

    baseline_parameters = sum(p.numel() for p in baseline_rng_model.parameters())
    candidate_parameters = sum(p.numel() for p in candidate_rng_model.parameters())
    overhead = (candidate_parameters - baseline_parameters) / baseline_parameters
    assert overhead <= .025, overhead
    shared_cores = sum(isinstance(module, SharedProximalCore)
                       for module in candidate_rng_model.modules())
    assert shared_cores == 1, shared_cores
    assert not hasattr(candidate_rng_model, 'proximal'), \
        'eight independent N12 cores were accidentally retained'

    baseline = BaselineNet(scale=4).eval()
    candidate = Net(scale=4).eval()
    missing, unexpected = candidate.load_state_dict(
        baseline.state_dict(), strict=False
    )
    prefixes = ('proximal_core.', 'stage_probe.', 'stage_')
    assert missing and all(name.startswith(prefixes) for name in missing), missing
    assert not unexpected, unexpected
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        baseline_y = baseline(x)
        candidate_y = candidate(x)
    assert torch.equal(baseline_y, candidate_y), \
        'zero-initialized candidate path changed B0 output'
    assert not candidate.diagnostics_snapshot(), \
        'inference diagnostics must be disabled by default'
    candidate.set_diagnostics_enabled(True)
    with torch.no_grad():
        candidate(x)
    candidate.set_diagnostics_enabled(False)
    diagnostics = candidate.diagnostics_snapshot()
    assert diagnostics and all(tuple(value.shape) == (8,)
                               for value in diagnostics.values())
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert torch.equal(diagnostics['gate_mean'], torch.full((8,), .5))
    assert torch.equal(diagnostics['gate_std'], torch.zeros(8))
    assert torch.equal(diagnostics['gate_saturation'], torch.zeros(8))

    # The zero-output initialization deliberately delays gradients to the
    # shared state transform. Verify both the immediate and second-step paths.
    train = Net(scale=4).train()
    optimizer = torch.optim.SGD(train.parameters(), lr=1e-2)
    train_x = torch.rand(1, 3, 16, 16)
    optimizer.zero_grad(set_to_none=True)
    train(train_x).abs().mean().backward()
    named = dict(train.named_parameters())
    for name in ('proximal_core.writeback.2.weight',
                 'proximal_core.q_update.weight'):
        grad = named[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    train(train_x).abs().mean().backward()
    named = dict(train.named_parameters())
    required = (
        'proximal_core.candidate.4.weight', 'proximal_core.gate.weight',
        'stage_probe.weight', 'stage_candidate_scale',
        'stage_candidate_bias', 'stage_gate_bias',
        'stage_writeback_scale', 'stage_q_scale',
    )
    gradient_l1 = {}
    for name in required:
        grad = named[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        gradient_l1[name] = float(grad.abs().sum())
        assert gradient_l1[name] > 0, name

    checkpoint = io.BytesIO()
    torch.save(train.state_dict(), checkpoint)
    checkpoint.seek(0)
    reloaded = Net(scale=4).eval()
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location='cpu', weights_only=True),
        strict=True,
    )
    train.eval()
    with torch.no_grad():
        trained_y = train(train_x)
        restored_y = reloaded(train_x)
    assert torch.equal(trained_y, restored_y)
    assert not torch.equal(trained_y, BaselineNet(scale=4).eval()(train_x)), \
        'candidate remained an exact no-op after optimization'

    real = None
    if args.root:
        lr, hr = load_real_pair(args.root.resolve())
        real_model = Net(scale=4).train()
        real_optimizer = torch.optim.Adam(real_model.parameters(), lr=2e-4)
        real_optimizer.zero_grad(set_to_none=True)
        sr = real_model(lr)
        loss = F.l1_loss(sr, hr)
        loss.backward()
        real_optimizer.step()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in real_model.parameters())
        real = {
            'lr_shape': list(lr.shape), 'hr_shape': list(hr.shape),
            'sr_shape': list(sr.shape), 'loss': float(loss.detach()),
            'finite_backward': True,
        }

    payload = {
        'candidate': 'N15/SRPR-SS',
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
        'baseline_parameters': baseline_parameters,
        'candidate_parameters': candidate_parameters,
        'parameter_overhead_fraction': overhead,
        'shared_core_count': shared_cores,
        'baseline_identity': 'passed',
        'rng_stream_identity': 'passed',
        'neutral_gate_initialization': 'passed',
        'zero_overhead_diagnostics_default': 'passed',
        'adjoint_errors_float64': {
            f'{h}x{w}': check_adjoint(h, w)
            for h, w in ((32, 48), (64, 80), (128, 96))
        },
        'second_step_gradient_l1': gradient_l1,
        'diagnostic_keys': sorted(diagnostics),
        'save_reload': 'passed',
        'real_div2k': real,
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
