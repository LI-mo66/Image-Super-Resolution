#!/usr/bin/env python3
"""Check probe identity, actual stage gradients and checkpoint immutability."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro'))
sys.path.insert(0, str(ROOT / 'LFMN'))
import torch
from diagnose_f1_20e import StageProbe, state_digest
from model.lfmn import Net as B0
from model.lfmnf1 import Net as F1


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for label, cls in (('B0', B0), ('F1', F1)):
        torch.manual_seed(1)
        net = cls(scale=4).to(device).eval()
        sample = torch.randn(1, 3, 31, 35, device=device)
        before = state_digest(net)
        with torch.no_grad():
            plain = net(sample)
        probe = StageProbe(net, label)
        with torch.no_grad():
            observed = net(sample)
        torch.testing.assert_close(observed, plain, rtol=1e-5, atol=1e-5)
        assert len(probe.rows) == 8
        assert [row['stage'] for row in probe.rows] == list(range(1, 9))
        if label == 'F1':
            assert all(abs(row['direct_previous_retention'] - 1) < 1e-10 for row in probe.rows)
        else:
            assert all(0 <= row['direct_previous_retention'] <= 1 for row in probe.rows)
        probe.reset('gradient')
        sr = net(sample)
        loss = sr.square().mean()
        targets = probe.states + probe.deltas
        if label == 'F1':
            targets.append(net.residual_scales)
        gradients = torch.autograd.grad(loss, targets)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        probe.close()
        probe.close()
        assert state_digest(net) == before
        assert all(parameter.grad is None for parameter in net.parameters())
        with torch.no_grad():
            restored = net(sample)
        torch.testing.assert_close(restored, plain, rtol=1e-5, atol=1e-5)
        print(label + ' PROBE PASSED: identity, eight states, gradients, cleanup, unchanged model', flush=True)


if __name__ == '__main__':
    main()
