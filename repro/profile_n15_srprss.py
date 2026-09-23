#!/usr/bin/env python3
"""Matched B0/N12/N15 parameter, Conv2d MAC, memory, and latency profile."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet
from model.lfmnsrprss import Net as SRPRSSNet


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def profile(model, sample, warmup, repeats, device):
    hooks, macs = [], 0

    def hook(layer, inputs, output):
        nonlocal macs
        output = output[0] if isinstance(output, tuple) else output
        b, c, h, w = output.shape
        macs += (b * c * h * w * (layer.in_channels // layer.groups) *
                 layer.kernel_size[0] * layer.kernel_size[1])

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        sync(device)
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                hooks.append(module.register_forward_hook(hook))
        model(sample)
        sync(device)
        for item in hooks:
            item.remove()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            model(sample)
            sync(device)
            times.append((time.perf_counter() - start) * 1000)
    peak = {'allocated_mib': None, 'reserved_mib': None}
    if device.type == 'cuda':
        peak = {
            'allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
        }
    ordered = sorted(times)
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'conv2d_macs': macs, 'conv2d_flops': 2 * macs,
        'median_ms': statistics.median(times),
        'p95_ms': ordered[max(0, int(.95 * len(ordered)) - 1)],
        **peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = torch.rand(1, 3, args.size, args.size, device=device)
    rows = []
    for name, cls in (('LFMN', BaselineNet), ('SRPR-SS', SRPRSSNet)):
        model = cls(scale=4).to(device).eval()
        rows.append({
            'model': name, 'device': str(device), 'precision': 'float32',
            'input_shape': list(sample.shape),
            **profile(model, sample, args.warmup, args.repeats, device),
        })
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    payload = {
        'mac_definition': '1 MAC = one multiply-accumulate; FLOPs = 2 * MAC',
        'rows': rows,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
