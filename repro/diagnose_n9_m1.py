"""Trained routing and same-device efficiency audit, no optimization."""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as Baseline
from model.lfmnpcstr import Net as Candidate
from data.div2k import DIV2K
from run_n9_m1_server import dataset_args


def diagnose(baseline, candidate, data_root, output):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for efficiency audit')
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    dataset = DIV2K(dataset_args(data_root), name='DIV2K', train=True)
    x = torch.stack([dataset[index][0] for index in range(4)]).cuda()
    b, m = Baseline(scale=4).cuda().eval(), Candidate(scale=4).cuda().eval()
    b.load_state_dict(torch.load(baseline, map_location='cpu', weights_only=True), strict=True)
    m.load_state_dict(torch.load(candidate, map_location='cpu', weights_only=True), strict=True)
    for block in m.blocks:
        block[0].collect_routing_stats = True
    with torch.inference_mode():
        assert torch.isfinite(m(x)).all()
    routing = [{key: float(value) for key, value in block[0].last_routing_stats.items()}
               for block in m.blocks]
    for block in m.blocks:
        block[0].collect_routing_stats = False
    measurements = {}
    sample = x[:1]
    # Alternate ordering to reduce warm-up/clock-order bias; CUDA events
    # include device work, while peak allocation includes both resident nets.
    with torch.inference_mode():
        for _ in range(10):
            b(sample)
            m(sample)
        torch.cuda.synchronize()
        times = {'b0': [], 'm1': []}
        peaks = {'b0': [], 'm1': []}
        for index in range(30):
            order = [('b0', b), ('m1', m)]
            if index % 2:
                order.reverse()
            for name, net in order:
                torch.cuda.reset_peak_memory_stats()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                prediction = net(sample)
                end.record()
                end.synchronize()
                times[name].append(start.elapsed_time(end))
                peaks[name].append(torch.cuda.max_memory_allocated())
                del prediction
        for name in times:
            measurements[name] = {'median_ms': float(np.median(times[name])),
                                  'p90_ms': float(np.quantile(times[name], .9)),
                                  'peak_allocated_mib': float(max(peaks[name]) / 2**20)}
    latency_ratio = measurements['m1']['median_ms'] / measurements['b0']['median_ms']
    memory_ratio = measurements['m1']['peak_allocated_mib'] / measurements['b0']['peak_allocated_mib']
    result = {'routing': routing, 'measurements': measurements,
              'routing_pass': all(s['effective_tokens_mean'] > 8 and s['mass_ratio'] < 100
                                  and all(np.isfinite(list(s.values()))) for s in routing),
              'efficiency_pass': latency_ratio <= 1.1 and memory_ratio <= 1.1,
              'latency_ratio': latency_ratio, 'memory_ratio': memory_ratio,
              'gpu': torch.cuda.get_device_name(0), 'input': 'B1 LR64 FP32',
              'limitations': 'One-size engineering audit; full-image memory and total FLOPs remain unknown.'}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.baseline, args.candidate, args.data_root, args.output)


if __name__ == '__main__':
    main()
