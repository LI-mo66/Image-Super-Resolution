#!/usr/bin/env python3
"""Matched parameter, latency, and peak-memory profile for N9 candidates."""
import argparse
import math
import statistics
import time
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmnmixcontrol import Net as ControlNet
from model.lfmnpcstr import Net as PCSTRNet
from model.lfmnpcstrwide import Net as WidePCSTRNet


def percentile(values, probability):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(
        probability * len(ordered)
    ) - 1))
    return ordered[index]


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    args = parser.parse_args()
    if args.size < 28 or args.warmup < 1 or args.repeats < 2:
        raise ValueError('size>=28, warmup>=1, and repeats>=2 are required')

    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample = torch.rand(1, 3, args.size, args.size, device=device) * 255
    constructors = (
        ('baseline', BaselineNet),
        ('pcstr', PCSTRNet),
        ('pcstr_wide', WidePCSTRNet),
        ('mix_control', ControlNet),
    )
    peaks = {}
    with torch.inference_mode():
        # Measure memory one model at a time so unrelated model weights do not
        # mask small candidate differences.
        for name, constructor in constructors:
            model = constructor(scale=4).to(device).eval()
            model(sample)
            synchronize(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            output = model(sample)
            synchronize(device)
            if not torch.isfinite(output).all():
                raise RuntimeError(f'{name} produced nonfinite output')
            peaks[name] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == 'cuda' else float('nan')
            )
            del output, model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    models = {
        name: constructor(scale=4).to(device).eval()
        for name, constructor in constructors
    }
    timings = {name: [] for name in models}
    with torch.inference_mode():
        # Warm up in rotating order so no candidate receives a uniquely cold
        # device or kernel-cache state.
        names = list(models)
        for iteration in range(args.warmup):
            offset = iteration % len(names)
            for index in range(len(names)):
                models[names[(index + offset) % len(names)]](sample)
        synchronize(device)

        # Rotate model order each repeat to reduce clock/thermal order bias.
        for iteration in range(args.repeats):
            offset = iteration % len(names)
            for index in range(len(names)):
                name = names[(index + offset) % len(names)]
                synchronize(device)
                start = time.perf_counter()
                models[name](sample)
                synchronize(device)
                timings[name].append(
                    1000.0 * (time.perf_counter() - start)
                )

    print('| model | parameters | median ms | P90 ms | peak allocated MiB |')
    print('| --- | ---: | ---: | ---: | ---: |')
    for name, _ in constructors:
        count = sum(
            parameter.numel() for parameter in models[name].parameters()
        )
        median = statistics.median(timings[name])
        p90 = percentile(timings[name], 0.9)
        print('| {} | {} | {:.3f} | {:.3f} | {:.1f} |'.format(
            name, count, median, p90, peaks[name]
        ))
    print('\nDevice:', device)
    print('Input: 1x3x{}x{}; warmup={}; repeats={}'.format(
        args.size, args.size, args.warmup, args.repeats
    ))
    print('Latency/memory are device-specific; this script does not claim FLOPs.')


if __name__ == '__main__':
    main()
