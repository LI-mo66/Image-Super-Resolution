"""Paired FP32 inference timing; no optimizer or accuracy claims.

B0 uses an existing checkpoint strictly. N23 inherits only retained tensors;
its new SCC weights remain untrained. This is a pretraining resource gate,
not a final trained-model deployment benchmark.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn_exact_overlap import Net as BaselineNet
from model.lfmn_n23 import Net


def synchronize():
    torch.cuda.synchronize()


def resource_decision(sizes):
    if set(sizes) != {'64x64', '128x128', '96x160'}:
        raise ValueError('Incomplete resource matrix')
    return ('RESOURCE_GATE_PASS_PENDING_SERVER_CONFIRMATION'
            if all(s['latency_gate'] == 'PASS' for s in sizes.values())
            else 'RESOURCE_GATE_FAIL_NO_TRAINING')


def measure(net, image, warmup, repeats, cache_miss=False):
    for _ in range(warmup):
        output = net(image)
    synchronize()
    assert torch.isfinite(output).all()
    del output
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    wall, events = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        synchronize()
        t0 = time.perf_counter()
        if cache_miss:
            for block in net.blocks:
                block[1].fast_patches.clear_cache()
        start.record()
        output = net(image)
        end.record()
        synchronize()
        wall.append((time.perf_counter()-t0)*1000)
        events.append(start.elapsed_time(end))
        del output
    return {'wall_ms': wall, 'cuda_event_ms': events,
            'allocated_before_bytes': allocated_before,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}


def phases(net, image):
    stamps, handles = {}, []
    for i, block in enumerate(net.blocks):
        for j, module in enumerate(block):
            key = f'stage{i+1}_{"TAB" if j == 0 else "LRSA"}'
            def before(module, inputs, key=key):
                stamps[key] = [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]
                stamps[key][0].record()
            def after(module, inputs, output, key=key):
                stamps[key][1].record()
            handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        output = net(image)
        synchronize()
        return {k: v[0].elapsed_time(v[1]) for k, v in stamps.items()}
    finally:
        for handle in handles:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-checkpoint', type=Path, required=True)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--cache-miss', action='store_true',
                    help='Rebuild geometry every timed forward, including CPU work and transfers')
    ap.add_argument('--expected-sha256', default='e428004505ec01364f60a0812c879b6ffd1fc08d013e48ffe81a22904927023b')
    ap.add_argument('--output', type=Path, help='New ignored experiment/all_runs JSON artifact')
    args = ap.parse_args()
    assert args.warmup >= 5 and args.repeats >= 20 and args.rounds >= 2
    if args.output:
        args.output = args.output.resolve()
        args.output.relative_to((ROOT/'experiment/all_runs').resolve())
        assert not args.output.exists(), 'Refuse to overwrite existing evidence'
    checkpoint_hash = hashlib.sha256(args.baseline_checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == args.expected_sha256, 'Wrong baseline checkpoint'
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(23)
    baseline = BaselineNet(scale=4).eval()
    state = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=True)
    baseline.load_state_dict(state, strict=True)
    initialized = [v for k, v in state.items() if k.endswith('.initted')]
    assert len(initialized) == 8 and all(bool(v.item()) for v in initialized)
    candidate = Net(scale=4).eval()
    candidate_state = candidate.state_dict()
    retained = {k: v for k, v in state.items()
                if k in candidate_state and '.1.layer.0.fn.' not in k}
    candidate_state.update(retained)
    candidate.load_state_dict(candidate_state, strict=True)
    assert all(torch.equal(v, candidate.state_dict()[k]) for k, v in retained.items())
    assert all(block[1].normalize_overlap for net in (baseline, candidate) for block in net.blocks)
    del state, candidate_state, retained
    models = {'b0': baseline, 'n23': candidate}
    report = {'purpose': 'PRETRAINING_RESOURCE_GATE_NOT_PSNR',
              'checkpoint_sha256': checkpoint_hash,
              'n23_weights': 'retained B0 tensors plus untrained SCC, no optimizer steps',
              'environment': {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                              'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                              'precision': 'FP32, TF32 off', 'cpu_threads': 4},
              'protocol': {'batch': 1, 'scale': 4, 'warmup': args.warmup,
                           'repeats_per_round': args.repeats, 'rounds': args.rounds,
                           'order': 'b0/n23 then n23/b0 alternating',
                           'geometry_cache': 'miss_every_forward' if args.cache_miss else 'warm_hit',
                           'overlap': 'exact_coverage_v1', 'gate_wall_limit': 1.10},
              'sizes': {}}
    raw_measurements = {}
    with torch.no_grad():
        for height, width in [(64, 64), (128, 128), (96, 160)]:
            torch.manual_seed(23)
            image = torch.rand(1, 3, height, width, device='cuda')*255
            measurements = {'b0': [], 'n23': []}
            diagnostic = {}
            for round_index in range(args.rounds):
                order = ('b0', 'n23') if round_index % 2 == 0 else ('n23', 'b0')
                for label in order:
                    net = models[label].cuda()
                    print(f'PROGRESS {height}x{width} round{round_index+1} {label}', flush=True)
                    result = measure(net, image, args.warmup, args.repeats, args.cache_miss)
                    measurements[label].append(result)
                    if round_index == args.rounds-1:
                        diagnostic[label] = phases(net, image)
                    net.cpu()
                    gc.collect()
                    torch.cuda.empty_cache()
            summary = {}
            for label, rounds in measurements.items():
                wall = [v for r in rounds for v in r['wall_ms']]
                gpu = [v for r in rounds for v in r['cuda_event_ms']]
                summary[label] = {
                    'params': sum(p.numel() for p in models[label].parameters()),
                    'wall_median_ms': float(np.median(wall)), 'wall_p90_ms': float(np.quantile(wall, .9)),
                    'cuda_event_median_ms': float(np.median(gpu)),
                    'round_wall_medians_ms': [statistics.median(r['wall_ms']) for r in rounds],
                    'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rounds),
                    'peak_increment_bytes': max(r['peak_allocated_bytes']-r['allocated_before_bytes'] for r in rounds)}
            ratio = summary['n23']['wall_median_ms']/summary['b0']['wall_median_ms']
            report['sizes'][f'{height}x{width}'] = {'summary': summary,
                'wall_ratio_n23_over_b0': ratio, 'latency_gate': 'PASS' if ratio <= 1.10 else 'FAIL',
                'round_wall_ratios': [summary['n23']['round_wall_medians_ms'][i]/summary['b0']['round_wall_medians_ms'][i]
                                      for i in range(args.rounds)],
                'p90_ratio_n23_over_b0': summary['n23']['wall_p90_ms']/summary['b0']['wall_p90_ms'],
                'phase_diagnostic_ms_not_deployment_timing': diagnostic}
            raw_measurements[f'{height}x{width}'] = measurements
            del image
            torch.cuda.empty_cache()
    report['decision'] = resource_decision(report['sizes'])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**report, 'raw_measurements': raw_measurements}, indent=2), encoding='utf-8')
        report['raw_evidence_path'] = str(args.output)
    print('RESULT_JSON '+json.dumps(report), flush=True)
    if report['decision'] == 'RESOURCE_GATE_FAIL_NO_TRAINING':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
