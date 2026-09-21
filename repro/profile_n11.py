#!/usr/bin/env python3
"""Record N11 parameter, MAC/FLOPs, memory, and latency metrics."""
import argparse
import json
import statistics
import time
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnsrprv1 import Net
from model.lfmn import Net as BaselineNet


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def params(model):
    return sum(p.numel() for p in model.parameters())


def conv_macs(module, shape):
    total = 0
    hooks = []
    def hook(layer, inputs, output):
        if not isinstance(layer, torch.nn.Conv2d):
            return
        out = output[0] if isinstance(output, tuple) else output
        batch, channels, height, width = out.shape
        total_layer = batch * channels * height * width
        total_layer *= layer.in_channels // layer.groups
        total_layer *= layer.kernel_size[0] * layer.kernel_size[1]
        total_layer += (batch * channels * height * width if layer.bias is not None else 0)
        nonlocal total
        total += int(total_layer)
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))
    device = next(module.parameters()).device
    sample = torch.rand(*shape, device=device)
    with torch.inference_mode():
        module(sample)
    for handle in hooks:
        handle.remove()
    return total


def measure(model, sample, warmup, repeats, device):
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        values = []
        for _ in range(repeats):
            sync(device)
            start = time.perf_counter()
            model(sample)
            sync(device)
            values.append((time.perf_counter() - start) * 1000)
        memory = {
            'allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20,
        } if device.type == 'cuda' else {'allocated_mib': None, 'reserved_mib': None}
    return {'median_ms': statistics.median(values), 'p95_ms': sorted(values)[max(0, int(.95 * len(values)) - 1)], **memory}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = torch.rand(1, 3, args.size, args.size, device=device)
    rows = []
    for name, cls in (('LFMN', BaselineNet), ('SRPRv1', Net)):
        model = cls(scale=4).to(device).eval()
        macs = conv_macs(model, (1, 3, args.size, args.size))
        row = {
            'model': name,
            'parameters': params(model),
            'macs': macs,
            'flops': 2 * macs,
            'mac_definition': '1 MAC = multiply-accumulate; FLOPs = 2 * MAC',
            'input_shape': [1, 3, args.size, args.size],
            'device': str(device),
            'precision': 'float32',
            **measure(model, sample, args.warmup, args.repeats, device),
        }
        rows.append(row)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    payload = {'profiler': 'same local Conv2d hook for both models', 'rows': rows}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
