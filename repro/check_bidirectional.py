#!/usr/bin/env python3
"""Validate N1 identity initialization, compatibility, and gradients."""
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet
from model.lfmnbidirectional import Net as BidirectionalNet


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline = BaselineNet(scale=4)
    candidate = BidirectionalNet(scale=4)
    state = torch.load(ROOT / 'LFMN/model/scale4_model_939.pt',
                       map_location='cpu', weights_only=True)
    baseline.load_state_dict(state, strict=True)
    result = candidate.load_state_dict(state, strict=False)
    expected = {key for key in candidate.state_dict()
                if key.startswith('bidirectional.')}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise RuntimeError('unexpected checkpoint compatibility result')

    base_params = sum(p.numel() for p in baseline.parameters())
    candidate_params = sum(p.numel() for p in candidate.parameters())
    added = candidate_params - base_params
    if 100.0 * added / base_params > 3.0:
        raise RuntimeError('bidirectional branch exceeds 3% parameter budget')

    baseline = baseline.to(device).eval()
    candidate = candidate.to(device).eval()
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    with torch.inference_mode():
        y0 = baseline(x)
        y1 = candidate(x)
    difference = (y0 - y1).abs().max().item()
    if difference != 0.0:
        raise RuntimeError('N1 is not identity-initialized')

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    y = candidate(x)
    target = torch.rand_like(y)
    F.l1_loss(y, target).backward()
    branch_grad = sum(
        p.grad.abs().sum().item()
        for p in candidate.bidirectional.parameters() if p.grad is not None
    )
    if branch_grad == 0.0:
        raise RuntimeError('bidirectional branch receives no gradient')

    print('device:', device)
    print('initial max output difference: {:.1f}'.format(difference))
    print('baseline parameters:', base_params)
    print('added parameters: {} ({:.3f}%)'.format(added, 100 * added / base_params))
    print('branch gradient sum: {:.6f}'.format(branch_grad))
    print('bidirectional validation passed')


if __name__ == '__main__':
    main()
