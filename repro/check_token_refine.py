#!/usr/bin/env python3
"""Validate zero-parameter, evaluation-only TAB prototype refinement."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmntokenrefine import Net as RefineNet


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(
        ROOT / 'LFMN/model/scale4_model_939.pt',
        map_location='cpu', weights_only=True,
    )
    baseline = BaselineNet(scale=4)
    refine1 = RefineNet(scale=4, token_refine_iters=1)
    refine3 = RefineNet(scale=4, token_refine_iters=3)
    baseline.load_state_dict(state, strict=True)
    refine1.load_state_dict(state, strict=True)
    refine3.load_state_dict(state, strict=True)

    counts = [sum(p.numel() for p in model.parameters())
              for model in (baseline, refine1, refine3)]
    require(len(set(counts)) == 1, 'prototype refinement changed parameter count')
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    baseline = baseline.to(device).eval()
    refine1 = refine1.to(device).eval()
    refine3 = refine3.to(device).eval()
    for global_attention, _ in refine3.blocks:
        global_attention.eval_refine_iters = 0
    with torch.inference_mode():
        output0 = baseline(x)
        disabled3 = refine3(x)
    disabled_difference = (output0 - disabled3).abs().max().item()
    require(torch.equal(output0, disabled3),
            'disabled refinement differs from baseline evaluation')
    for global_attention, _ in refine3.blocks:
        global_attention.eval_refine_iters = 3
    buffers_before = {
        name: value.detach().cpu().clone()
        for name, value in refine3.named_buffers()
        if name.endswith('means') or name.endswith('initted')
    }
    baseline.eval()
    refine1.eval()
    refine3.eval()
    with torch.inference_mode():
        repeat0 = baseline(x)
        output1 = refine1(x)
        output3 = refine3(x)
        repeat3 = refine3(x)
    require(not torch.equal(output0, output1),
            'one-step refinement has no evaluation effect')
    require(not torch.equal(output1, output3),
            'three-step refinement matches one-step output')
    baseline_repeat_difference = (output0 - repeat0).abs().max().item()
    repeat_difference = (output3 - repeat3).abs().max().item()

    for name, before in buffers_before.items():
        after = dict(refine3.named_buffers())[name].cpu()
        require(torch.equal(before, after), f'evaluation mutated buffer {name}')

    print('device:', device)
    print('parameters:', counts[0], '(added: 0)')
    print('disabled-path max difference:', disabled_difference)
    print('eval baseline/refine1 max difference:',
          (output0 - output1).abs().max().item())
    print('eval refine1/refine3 max difference:',
          (output1 - output3).abs().max().item())
    print('baseline repeat max difference:', baseline_repeat_difference)
    print('repeat max difference:', repeat_difference)
    print('buffers unchanged: OK')
    print('Token-refinement validation passed (not a performance result)')


if __name__ == '__main__':
    main()
