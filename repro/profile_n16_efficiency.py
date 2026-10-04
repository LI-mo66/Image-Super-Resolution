#!/usr/bin/env python3
"""Matched inference efficiency audit for N16 epoch-150 checkpoints."""
import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmnsrprv2 import Net as SRPRv2Net

MODELS = {
    'c1': (BaselineNet, 'c1/model/model_150.pt'),
    'srpr_c1': (SRPRv2Net, 'srpr_c1/model/model_150.pt'),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, probability):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(
        probability * len(ordered)
    ) - 1))
    return ordered[index]


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def validate_source(source):
    source = source.resolve()
    status = source / 'wrapper_exit_status.txt'
    if not status.is_file() or status.read_text().strip() != '0':
        raise ValueError('epoch-150 source wrapper status is not zero')
    summary_path = source / 'summary_150e.json'
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('decision') != 'PROMOTE_TO_LONG_RUN_VALIDATION':
        raise ValueError('epoch-150 source did not pass its registered gate')

    checkpoints = {}
    for name, (model_class, relative_path) in MODELS.items():
        checkpoint = source / relative_path
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model_class(scale=4).load_state_dict(state, strict=True)
        checkpoints[name] = {
            'path': str(checkpoint), 'sha256': sha256(checkpoint)
        }
    return source, checkpoints


def load_model(name, source, device):
    model_class, relative_path = MODELS[name]
    model = model_class(scale=4)
    state = torch.load(
        source / relative_path, map_location='cpu', weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def timed_forward(model, sample, device):
    synchronize(device)
    start = time.perf_counter()
    output = model(sample)
    synchronize(device)
    return output, 1000.0 * (time.perf_counter() - start)


def measure_memory_and_cold(name, source, sample, device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = load_model(name, source, device)
    allocated_before = (
        torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    )
    with torch.inference_mode():
        output, cold_ms = timed_forward(model, sample, device)
    if not torch.isfinite(output).all():
        raise RuntimeError(f'{name} produced nonfinite output')
    peak = (
        torch.cuda.max_memory_allocated(device) if device.type == 'cuda'
        else None
    )
    result = {
        'cold_first_forward_ms': cold_ms,
        'peak_allocated_mib': peak / (1024 ** 2) if peak is not None else None,
        'incremental_peak_mib': (
            (peak - allocated_before) / (1024 ** 2)
            if peak is not None else None
        ),
        'output_shape': list(output.shape),
        'output_finite': True,
    }
    del output, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def measure_latency(source, sample, device, warmup, repeats):
    models = {name: load_model(name, source, device) for name in MODELS}
    names = list(models)
    timings = {name: [] for name in names}
    with torch.inference_mode():
        for iteration in range(warmup):
            offset = iteration % len(names)
            for index in range(len(names)):
                models[names[(index + offset) % len(names)]](sample)
        synchronize(device)
        for iteration in range(repeats):
            offset = iteration % len(names)
            for index in range(len(names)):
                name = names[(index + offset) % len(names)]
                _, elapsed = timed_forward(models[name], sample, device)
                timings[name].append(elapsed)
    result = {
        name: {
            'median_ms': statistics.median(timings[name]),
            'p90_ms': percentile(timings[name], 0.9),
            'samples_ms': timings[name],
        }
        for name in names
    }
    del models
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def shown(value, digits=3):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if any(size < 28 for size in args.sizes):
        raise ValueError('all LR input sizes must be >=28')
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError('warmup>=1 and repeats>=2 are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required unless --device cpu is explicit')

    source, checkpoints = validate_source(args.source)
    device = torch.device(args.device)
    parameter_counts = {
        name: sum(p.numel() for p in model_class(scale=4).parameters())
        for name, (model_class, _) in MODELS.items()
    }
    metadata = {
        'audit': 'N16 matched epoch-150 inference efficiency',
        'source': str(source), 'checkpoints': checkpoints,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'batch': 1, 'dtype': 'float32', 'scale': 4,
        'sizes_lr': args.sizes, 'warmup': args.warmup,
        'repeats': args.repeats, 'parameters': parameter_counts,
        'results': {},
        'mac_flop_status': 'not_measured_no_dependency_safe_exact_counter',
    }
    print(json.dumps({k: v for k, v in metadata.items() if k != 'results'}, indent=2))
    if args.check_only:
        return

    torch.manual_seed(1)
    for size in args.sizes:
        sample = torch.rand(1, 3, size, size, device=device) * 255
        size_result = {
            name: measure_memory_and_cold(name, source, sample, device)
            for name in MODELS
        }
        latency = measure_latency(source, sample, device, args.warmup, args.repeats)
        for name in MODELS:
            size_result[name].update(latency[name])
        metadata['results'][str(size)] = size_result

    args.output.mkdir(parents=True)
    json_path = args.output / 'efficiency_profile.json'
    json_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = [
        'N16 matched epoch-150 inference efficiency',
        f'device={metadata["gpu"] or device}; torch={torch.__version__}; cuda={torch.version.cuda}',
        f'batch=1; dtype=float32; scale=4; warmup={args.warmup}; repeats={args.repeats}',
        '| LR input | model | params | cold ms | median ms | P90 ms | peak MiB | incremental peak MiB |',
        '|---:|---|---:|---:|---:|---:|---:|---:|',
    ]
    for size in args.sizes:
        for name in MODELS:
            result = metadata['results'][str(size)][name]
            lines.append(
                f'| {size}x{size} | {name} | {parameter_counts[name]} | '
                f'{shown(result["cold_first_forward_ms"])} | '
                f'{shown(result["median_ms"])} | {shown(result["p90_ms"])} | '
                f'{shown(result["peak_allocated_mib"], 1)} | '
                f'{shown(result["incremental_peak_mib"], 1)} |'
            )
    base, candidate = parameter_counts['c1'], parameter_counts['srpr_c1']
    lines.extend([
        f'parameter_delta={candidate - base:+d} ({100.0 * (candidate - base) / base:+.2f}%)',
        'KD teacher is training-only and contributes zero inference parameters.',
        'MAC/FLOPs are not reported: no dependency-safe exact counter for all custom operations is assumed.',
        f'json={json_path}',
    ])
    text_path = args.output / 'efficiency_profile.txt'
    text_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
