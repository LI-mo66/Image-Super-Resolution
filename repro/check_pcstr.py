#!/usr/bin/env python3
"""Validate N9 A0/A1/C0 structure before any performance training."""
import io
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmnmixcontrol import Net as ControlNet, PriorConditionedConvMixer
from model.lfmnpcstr import Net as PCSTRNet, PriorConditionedSoftTokenRouter
from model.lfmnpcstrwide import Net as WidePCSTRNet


EXPECTED_PARAMETERS = {
    'baseline': 759_627,
    'pcstr': 733_259,
    'pcstr_wide': 759_827,
    'mix_control': 733_867,
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def gradient_check(module, feature, prior, label):
    module.train()
    module.zero_grad(set_to_none=True)
    output = module(feature, prior)
    target = torch.randn_like(output)
    F.l1_loss(output, target).backward()
    values = []
    for name, parameter in module.named_parameters():
        require(parameter.grad is not None, f'{label}.{name}: missing gradient')
        require(torch.isfinite(parameter.grad).all(),
                f'{label}.{name}: nonfinite gradient')
        magnitude = parameter.grad.abs().sum().item()
        require(magnitude > 0, f'{label}.{name}: zero gradient')
        values.append(magnitude)
    return min(values), max(values)


def analytic_route_macs(height=64, width=64):
    n = height * width
    c, cs, route, tokens, qk = 48, 32, 16, 32, 24
    pcstr = (
        n * ((c + cs) * route + route * tokens + c * c + 2 * tokens * c)
        + tokens * (2 * c * qk + 2 * c * c)
        + tokens * tokens * (qk + c)
    )
    hidden = 52
    control = n * (
        (c + cs) * hidden + 9 * hidden + hidden * c
    ) + (c + cs) * c
    return pcstr, control


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_classes = {
        'baseline': BaselineNet,
        'pcstr': PCSTRNet,
        'pcstr_wide': WidePCSTRNet,
        'mix_control': ControlNet,
    }
    models = {name: cls(scale=4) for name, cls in model_classes.items()}
    counts = {name: parameter_count(model) for name, model in models.items()}
    require(counts == EXPECTED_PARAMETERS,
            f'parameter counts changed: {counts!r}')
    require(abs(counts['pcstr_wide'] - counts['baseline']) <= 256,
            'A1 is not a near-parameter-matched baseline control')
    require(abs(counts['mix_control'] - counts['pcstr']) <= 1024,
            'C0 is not parameter-matched to A0')

    for model in models.values():
        model.to(device).eval()
    sample = torch.rand(1, 3, 32, 32, device=device) * 255
    with torch.inference_mode():
        outputs = {name: model(sample) for name, model in models.items()}
    for name, output in outputs.items():
        require(output.shape == (1, 3, 128, 128),
                f'{name}: unexpected output shape {tuple(output.shape)}')
        require(torch.isfinite(output).all(), f'{name}: nonfinite output')

    feature = torch.randn(2, 48, 16, 16, device=device)
    prior = torch.randn(2, 32, 16, 16, device=device)
    router = models['pcstr'].blocks[0][0]
    require(isinstance(router, PriorConditionedSoftTokenRouter),
            'A0 did not replace TAB with PCSTR')
    require(router.num_tokens == 32 and router.qk_dim == 24,
            'A0 K or Q/K dimension drifted from the frozen design')
    require(router.to_v.out_features == 48,
            'A0 value path is not full-width')

    with torch.no_grad():
        assignment = router.assignment_map(feature, prior)
        feature_changed = router.assignment_map(
            feature + 0.1 * torch.randn_like(feature), prior
        )
        prior_changed = router.assignment_map(
            feature, prior + 0.1 * torch.randn_like(prior)
        )
    require(assignment.shape == (2, 256, 32),
            'assignment has the wrong BxNxK shape')
    require(torch.allclose(
        assignment.sum(dim=-1), torch.ones_like(assignment[..., 0]),
        atol=1e-6, rtol=1e-6,
    ), 'assignment does not normalize over K')
    require(not torch.allclose(assignment, feature_changed),
            'assignment does not causally depend on Fm')
    require(not torch.allclose(assignment, prior_changed),
            'assignment does not causally depend on Fs')

    router.collect_routing_stats = True
    router.eval()
    with torch.no_grad():
        eval_output = router(feature, prior)
    stats = router.last_routing_stats
    require(stats is not None, 'routing diagnostics were not collected')
    require(stats['effective_tokens_mean'].item() > 8,
            'initial route has fewer than K/4 effective tokens')
    require(stats['mass_ratio'].item() < 100,
            'initial route is severely mass-imbalanced')
    router.collect_routing_stats = False
    router.train()
    with torch.no_grad():
        train_output = router(feature, prior)
    require(torch.allclose(eval_output, train_output, atol=1e-6, rtol=1e-6),
            'PCSTR train/eval data flows differ')

    pcstr_gradients = gradient_check(router, feature, prior, 'pcstr')
    control = models['mix_control'].blocks[0][0]
    require(isinstance(control, PriorConditionedConvMixer),
            'C0 did not install the ordinary mixer')
    control_gradients = gradient_check(control, feature, prior, 'control')

    if device.type == 'cuda':
        router.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            mixed_precision = router(feature, prior)
        require(torch.isfinite(mixed_precision).all(),
                'PCSTR produced nonfinite AMP output')

    candidate = models['pcstr'].eval()
    checkpoint = io.BytesIO()
    torch.save(candidate.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = PCSTRNet(scale=4).to(device).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    with torch.inference_mode():
        require(torch.equal(candidate(sample), restored(sample)),
                'strict checkpoint reload changed A0 output')

    pcstr_macs, control_macs = analytic_route_macs()
    relative_mac_gap = abs(control_macs - pcstr_macs) / pcstr_macs
    require(relative_mac_gap < 0.02,
            'C0 analytic mixer MACs are not matched to A0')

    print('device:', device)
    print('parameter counts:', counts)
    print('A0/C0 analytic route MACs at LR 64x64:',
          pcstr_macs, control_macs)
    print('A0/C0 analytic route MAC gap: {:.3f}%'.format(
        100.0 * relative_mac_gap
    ))
    print('routing diagnostics:', {
        key: float(value) for key, value in stats.items()
    })
    print('PCSTR gradient min/max:', pcstr_gradients)
    print('control gradient min/max:', control_gradients)
    print('shape, causality, train/eval, AMP, reload, gradients: OK')
    print('N9 structural validation passed (not a performance result)')


if __name__ == '__main__':
    main()
