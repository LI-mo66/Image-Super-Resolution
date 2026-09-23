#!/usr/bin/env python3
"""Profile LFMN and N13/SRPRv3 with the same input and Conv2d MAC rule."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet  # noqa: E402
from model.lfmnsrprv3 import Net as SRPRv3Net  # noqa: E402


def conv_macs(model, sample):
    total = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal total
        batch, out_channels, out_height, out_width = output.shape
        kernel_height, kernel_width = module.kernel_size
        in_per_group = module.in_channels // module.groups
        total += (batch * out_channels * out_height * out_width
                  * in_per_group * kernel_height * kernel_width)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()
    return total


def profile(model, sample, warmup, repeats):
    device = sample.device
    model.eval()
    macs = conv_macs(model, sample)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(sample)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - start) * 1000)
    ordered = sorted(times)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'conv2d_macs': macs,
        'flops_2x_macs': 2 * macs,
        'median_ms': statistics.median(times),
        'p95_ms': ordered[p95_index],
        'allocated_peak_mib': (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == 'cuda' else None
        ),
        'reserved_peak_mib': (
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == 'cuda' else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = torch.rand(1, 3, args.size, args.size, device=device) * 255
    rows = []
    for name, model in (
        ('LFMN', BaselineNet(scale=4)),
        ('LFMNSRPRV3', SRPRv3Net(scale=4)),
    ):
        row = {'model': name, 'device': str(device),
               'input_shape': list(sample.shape), 'precision': 'float32'}
        row.update(profile(model.to(device), sample, args.warmup, args.repeats))
        rows.append(row)
    result = {
        'mac_definition': 'Conv2d only; 1 MAC is one multiply-accumulate; FLOPs=2*MAC',
        'rows': rows,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
