#!/usr/bin/env python3
"""Local structural, migration, gradient, reload and latency checks for N24."""
import argparse
import hashlib
import json
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmnsrprv2 import Net as V1Net
from model.lfmnn24ffnprune import Net as N24Net, _ffn_modules


EXPECTED_PARAMETERS = {96: 841563, 88: 825819, 80: 810075}


def percentile(values, probability):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(
        probability * len(ordered)
    ) - 1))
    return ordered[index]


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed(model, sample, device):
    synchronize(device)
    start = time.perf_counter()
    model(sample)
    synchronize(device)
    return 1000.0 * (time.perf_counter() - start)


def alternating_latency(models, sample, device, warmup, repeats):
    names = list(models)
    values = {name: [] for name in names}
    with torch.inference_mode():
        for iteration in range(warmup):
            for offset in range(len(names)):
                models[names[(iteration + offset) % len(names)]](sample)
        synchronize(device)
        for iteration in range(repeats):
            for offset in range(len(names)):
                name = names[(iteration + offset) % len(names)]
                values[name].append(timed(models[name], sample, device))
    return {
        name: {
            'median_ms': statistics.median(samples),
            'p90_ms': percentile(samples, 0.9),
            'samples_ms': samples,
        }
        for name, samples in values.items()
    }


def load_source(checkpoint):
    if checkpoint is None:
        torch.manual_seed(2401)
        return V1Net(scale=4).state_dict(), 'seeded_random_v1'
    return torch.load(checkpoint, map_location='cpu', weights_only=True), str(checkpoint)


def check_structure(state):
    results = {}
    for hidden, expected in EXPECTED_PARAMETERS.items():
        model = N24Net(scale=4, ffn_hidden=hidden)
        report = model.load_pruned_from_v1(state)
        count = sum(parameter.numel() for parameter in model.parameters())
        assert count == expected, (hidden, count, expected)
        ffns = list(_ffn_modules(model))
        assert len(ffns) == 16
        for _, ffn in ffns:
            assert ffn.fc1.in_features == 48
            assert ffn.fc1.out_features == hidden
            assert ffn.fc2.in_features == hidden
            assert ffn.fc2.out_features == 48
            conv = ffn.dwconv.depthwise_conv[0]
            assert conv.in_channels == conv.out_channels == conv.groups == hidden
        assert report['target_parameter_coverage'] == 1.0
        expected_mismatches = 0 if hidden == 96 else 80
        assert report['target_expected_shape_mismatches'] == expected_mismatches
        encoded_indices = json.dumps(
            report['selected_indices'], sort_keys=True
        ).encode('utf-8')
        results[str(hidden)] = {
            'parameters': count,
            'migration': {
                key: value for key, value in report.items()
                if key != 'selected_indices'
            },
            'selected_index_sets': len(report['selected_indices']),
            'selected_indices_sha256': hashlib.sha256(encoded_indices).hexdigest(),
        }
    return results


def check_exact_width_96(state, device):
    baseline = V1Net(scale=4).to(device).eval()
    baseline.load_state_dict(state, strict=True)
    candidate = N24Net(scale=4, ffn_hidden=96).to(device).eval()
    candidate.load_pruned_from_v1(state)
    torch.manual_seed(2402)
    sample = torch.rand(1, 3, 29, 35, device=device) * 255
    with torch.inference_mode():
        expected = baseline(sample)
        actual = candidate(sample)
    difference = (expected - actual).abs().max().item()
    assert difference == 0.0
    return {'shape': list(actual.shape), 'max_abs_difference': difference}


def check_rng_preservation(state):
    torch.manual_seed(2404)
    V1Net(scale=4)
    expected_after_construction = torch.rand(8)
    torch.manual_seed(2404)
    candidate = N24Net(scale=4, ffn_hidden=88)
    actual_after_construction = torch.rand(8)
    assert torch.equal(expected_after_construction, actual_after_construction)

    torch.manual_seed(2405)
    expected_after_migration = torch.rand(8)
    torch.manual_seed(2405)
    candidate.load_pruned_from_v1(state)
    actual_after_migration = torch.rand(8)
    assert torch.equal(expected_after_migration, actual_after_migration)
    return {
        'construction_matches_v1': True,
        'migration_preserves_cpu_rng': True,
    }


def check_learning_and_reload(state, device):
    model = N24Net(scale=4, ffn_hidden=88).to(device).train()
    report = model.load_pruned_from_v1(state)
    torch.manual_seed(2403)
    sample = torch.rand(1, 3, 31, 37, device=device) * 255
    target = torch.rand(1, 3, 124, 148, device=device) * 255
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    optimizer.zero_grad(set_to_none=True)
    output = model(sample)
    assert output.shape == target.shape
    assert torch.isfinite(output).all()
    loss = torch.nn.functional.l1_loss(output, target)
    assert torch.isfinite(loss)
    loss.backward()

    ffn_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if '.mlp.fn.' in name or '.layer.1.fn.' in name
    ]
    assert ffn_parameters
    assert all(parameter.grad is not None for parameter in ffn_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in ffn_parameters)
    optimizer_ids = [
        id(parameter)
        for group in optimizer.param_groups for parameter in group['params']
    ]
    model_ids = [id(parameter) for parameter in model.parameters()]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == set(model_ids)

    watched = model.blocks[0][0].mlp.fn.fc1.weight
    before = watched.detach().clone()
    optimizer.step()
    update = (watched.detach() - before).abs().max().item()
    assert update > 0.0

    model.eval()
    with torch.inference_mode():
        reference = model(sample)
    with tempfile.TemporaryDirectory(
        prefix='.n24_reload_', dir=ROOT
    ) as directory:
        path = Path(directory) / 'model.pt'
        torch.save(model.state_dict(), path)
        reloaded = N24Net(scale=4, ffn_hidden=88).to(device).eval()
        result = reloaded.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        assert not result.missing_keys and not result.unexpected_keys
        with torch.inference_mode():
            restored = reloaded(sample)
    reload_difference = (reference - restored).abs().max().item()
    assert reload_difference == 0.0
    return {
        'input_shape': list(sample.shape),
        'output_shape': list(output.shape),
        'loss': loss.item(),
        'ffn_parameter_tensors_with_finite_gradients': len(ffn_parameters),
        'optimizer_parameter_tensors': len(optimizer_ids),
        'watched_parameter_max_update': update,
        'reload_max_abs_difference': reload_difference,
        'migration_parameter_coverage': report['target_parameter_coverage'],
        'real_div2k_batch_checked': False,
        'c1_teacher_checked': False,
        'amp_checked': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--sizes', type=int, nargs='*', default=[64, 128])
    parser.add_argument('--latency-hidden', type=int, choices=(88, 80), default=88)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError('warmup>=1 and repeats>=2 are required')

    device = torch.device(args.device)
    state, source = load_source(args.checkpoint)
    report = {
        'candidate': 'N24 V1 ConvFFN structured pruning',
        'source': source,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'dtype': 'float32',
        'structure': check_structure(state),
        'rng_preservation': check_rng_preservation(state),
        'width_96_exact_equivalence': check_exact_width_96(state, device),
        'learning_and_reload': check_learning_and_reload(state, device),
        'latency': {},
        'performance_claim': 'none; engineering checks are not PSNR evidence',
    }

    for size in args.sizes:
        baseline = V1Net(scale=4).to(device).eval()
        baseline.load_state_dict(state, strict=True)
        candidate = N24Net(
            scale=4, ffn_hidden=args.latency_hidden
        ).to(device).eval()
        candidate.load_pruned_from_v1(state)
        sample = torch.rand(1, 3, size, size, device=device) * 255
        report['latency'][str(size)] = alternating_latency(
            {'v1': baseline, f'n24_h{args.latency_hidden}': candidate},
            sample,
            device,
            args.warmup,
            args.repeats,
        )
        del baseline, candidate, sample
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    shown = json.dumps(report, indent=2, ensure_ascii=False)
    print(shown)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(shown + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
