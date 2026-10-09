#!/usr/bin/env python3
"""Matched FP32 OFF profiling with explicit PyTorch FLOP-counter coverage."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
import torch
from torch.utils.flop_counter import FlopCounterMode
from model.lfmn import Net as B0
from model.lfmnf1 import Net as F1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Matched latency and peak GPU memory profiling requires CUDA')
    payload = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'precision': 'FP32', 'batch': 1, 'self_ensemble': False,
               'warmup': 10, 'repeats': 50,
               'flops_convention': 'PyTorch FlopCounterMode counted operations; multiply-add=2 FLOPs',
               'coverage_limit': 'Includes supported conv/matmul/attention; excludes sorting, indexing, normalization, pooling, interpolation and most elementwise operations. Not an exact all-operator total.',
               'results': {}}
    for label, cls in [('B0', B0), ('F1', F1)]:
        path = args.group / (label + '_x4_seed1') / 'model' / ('model_{}.pt'.format(args.epoch))
        if not path.exists():
            path = args.group / (label + '_smoke_x4_seed1') / 'model' / ('model_{}.pt'.format(args.epoch))
        for size in (64, 128):
            torch.cuda.empty_cache()
            net = cls(scale=4).cuda().eval()
            net.load_state_dict(torch.load(path, map_location='cuda', weights_only=True), strict=True)
            sample = torch.zeros(1, 3, size, size, device='cuda')
            times = []
            with torch.inference_mode():
                for _ in range(10):
                    net(sample)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(50):
                    start = time.perf_counter()
                    output = net(sample)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                    if not torch.isfinite(output).all():
                        raise FloatingPointError('nonfinite profiled output')
                    del output
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
                with FlopCounterMode(display=False) as counter:
                    net(sample)
            payload['results'][label + '_LR' + str(size)] = {
                'parameters': sum(p.numel() for p in net.parameters()),
                'counted_flops': counter.get_total_flops(),
                'counted_multi_add_equivalent': counter.get_total_flops() / 2,
                'median_ms': statistics.median(times),
                'p90_ms': sorted(times)[44],
                'peak_allocated_mib': peak_allocated, 'peak_reserved_mib': peak_reserved,
                'additional_scale_multiplications': 8 * 48 * size * size if label == 'F1' else 0,
            }
            del net, sample
    target = args.group / 'resources.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
