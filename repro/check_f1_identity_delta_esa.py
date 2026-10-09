#!/usr/bin/env python3
"""Gate 0/1 checks for the F1 identity-preserved delta ESA model."""
import argparse
import hashlib
from importlib import import_module
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = ROOT / 'LFMN'
EXPECTED_CHECKPOINT_SHA256 = (
    '44999471d8cc2d5f7dbf10d354766e08a5d23a84200a1060dfa9a9ed7a7711dd'
)
if str(LFMN_ROOT) not in sys.path:
    sys.path.insert(0, str(LFMN_ROOT))

from model.lfmn import Net as BaselineNet  # noqa: E402
from model.lfmnf1 import Net as F1Net  # noqa: E402


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def load_registered_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise AssertionError('{} contains non-finite values'.format(name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint', type=Path,
        default=ROOT / 'LFMN/model/scale4_model_939.pt',
    )
    parser.add_argument(
        '--device', choices=('auto', 'cpu', 'cuda'), default='auto'
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(
            'checkpoint SHA256 mismatch: expected {}, got {}'.format(
                EXPECTED_CHECKPOINT_SHA256, checkpoint_hash
            )
        )
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device(
        'cuda' if args.device == 'cuda' or (
            args.device == 'auto' and torch.cuda.is_available()
        ) else 'cpu'
    )

    torch.manual_seed(1)
    baseline = BaselineNet(scale=4)
    candidate = F1Net(scale=4)
    dynamic_module = import_module('model.lfmnf1')
    dynamic_candidate = dynamic_module.make_model(SimpleNamespace(scale=[4]))
    if not isinstance(dynamic_candidate, F1Net):
        raise AssertionError('--model LFMNF1 does not resolve to the F1 model')
    baseline_parameters = parameter_count(baseline)
    candidate_parameters = parameter_count(candidate)
    if candidate_parameters - baseline_parameters != 8:
        raise AssertionError('F1 must add exactly eight scalar parameters')
    if tuple(candidate.residual_scales.shape) != (8,):
        raise AssertionError('unexpected residual scale shape')
    if not torch.equal(candidate.residual_scales.detach(), torch.ones(8)):
        raise AssertionError('residual scales must initialize to one')

    state = load_registered_checkpoint(checkpoint)
    baseline.load_state_dict(state, strict=True)
    inherited = candidate.load_state_dict(state, strict=False)
    if inherited.missing_keys != ['residual_scales']:
        raise AssertionError(
            'unexpected missing keys: {}'.format(inherited.missing_keys)
        )
    if inherited.unexpected_keys:
        raise AssertionError(
            'unexpected checkpoint keys: {}'.format(inherited.unexpected_keys)
        )
    shared_elements = sum(value.numel() for value in state.values())
    candidate_shared_elements = sum(
        value.numel() for key, value in candidate.state_dict().items()
        if key != 'residual_scales'
    )
    if candidate_shared_elements != shared_elements:
        raise AssertionError('shared checkpoint element coverage is not 100%')

    candidate = candidate.to(device)
    candidate.eval()
    previous = torch.randn(1, 48, 31, 35, device=device)
    delta = torch.randn_like(previous)
    with torch.no_grad():
        original_scale = candidate.residual_scales[0].item()
        candidate.residual_scales[0].zero_()
        identity = candidate.update_stage(previous, delta, 0)
        if not torch.equal(identity, previous):
            raise AssertionError('zero residual scale must give an exact identity')
        candidate.residual_scales[0].fill_(original_scale)
        expected = previous + candidate.esas[0](delta)
        actual = candidate.update_stage(previous, delta, 0)
        if not torch.equal(actual, expected):
            raise AssertionError('unit scale does not match the registered formula')

        sample = torch.randn(1, 3, 31, 35, device=device)
        output = candidate(sample)
        if tuple(output.shape) != (1, 3, 124, 140):
            raise AssertionError('unexpected x4 output shape: {}'.format(output.shape))
        assert_finite(output, 'evaluation output')

    candidate.train()
    candidate.zero_grad(set_to_none=True)
    training_input = torch.randn(1, 3, 28, 32, device=device)
    training_output = candidate(training_input)
    loss = training_output.square().mean()
    assert_finite(loss, 'loss')
    loss.backward()
    scale_grad = candidate.residual_scales.grad
    if scale_grad is None:
        raise AssertionError('residual scales are disconnected from the loss')
    assert_finite(scale_grad, 'residual scale gradients')
    if not torch.all(scale_grad.abs() > 0):
        raise AssertionError('every residual scale must receive non-zero gradient')
    esa_grad_sum = sum(
        parameter.grad.detach().abs().sum().item()
        for esa in candidate.esas
        for parameter in esa.parameters()
        if parameter.grad is not None
    )
    if not esa_grad_sum > 0:
        raise AssertionError('ESA parameters did not receive gradient')

    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-4)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group['params']
    }
    if id(candidate.residual_scales) not in optimizer_ids:
        raise AssertionError('residual scales are absent from the optimizer')

    candidate_cpu = candidate.cpu().eval()
    with tempfile.TemporaryDirectory(prefix='.f1_reload_', dir=ROOT) as directory:
        saved = Path(directory) / 'f1_state.pt'
        torch.save(candidate_cpu.state_dict(), saved)
        reloaded = F1Net(scale=4)
        reloaded.load_state_dict(load_registered_checkpoint(saved), strict=True)
        if not torch.equal(
            reloaded.residual_scales.detach(),
            candidate_cpu.residual_scales.detach(),
        ):
            raise AssertionError('strict reload changed residual scales')

    print(json.dumps({
        'status': 'passed',
        'device': str(device),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash,
        'baseline_parameters': baseline_parameters,
        'candidate_parameters': candidate_parameters,
        'added_parameters': candidate_parameters - baseline_parameters,
        'checkpoint_missing_keys': inherited.missing_keys,
        'checkpoint_unexpected_keys': inherited.unexpected_keys,
        'shared_checkpoint_element_coverage': 1.0,
        'identity_at_zero_scale': True,
        'unit_scale_formula': True,
        'non_square_x4_shape': list(output.shape),
        'all_scale_gradients_nonzero': True,
        'esa_gradient_sum': esa_grad_sum,
        'strict_candidate_reload': True,
        'dynamic_model_entrypoint': 'model.lfmnf1',
        'training_or_psnr_claim': 'not run',
    }, indent=2))


if __name__ == '__main__':
    main()
