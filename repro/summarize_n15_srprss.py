#!/usr/bin/env python3
"""Summarize and apply the pre-registered N15 20-epoch gate."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def curve(run, filename):
    value = torch.load(run / filename, map_location='cpu', weights_only=True).float()
    return (value[:, 0, 0] if value.ndim == 3 else value.flatten()).numpy()


def images(path):
    rows = torch.load(path, map_location='cpu', weights_only=True)
    return {row['filename']: row for row in rows}


def bootstrap_ci(values, seed=1, samples=10000):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (samples, len(values)), replace=True).mean(1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline-reference', type=Path, required=True)
    args = parser.parse_args()
    candidate, baseline = args.run / 'srprss', args.baseline_reference
    cfg = dict(line.split(': ', 1) for line in
               (candidate / 'config.txt').read_text().splitlines() if ': ' in line)
    expected = {
        'model': 'LFMNSRPRSS', 'epochs': '20',
        'data_range': '1-800/801-900', 'scale': '[4]',
        'patch_size': '256', 'batch_size': '4', 'seed': '1',
        'lr': '0.0002', 'scheduler': 'cosine',
        'scheduler_t_max': '150', 'eta_min': '1e-06',
        'loss': '1*L1', 'pre_train': '',
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'{key}: expected {value!r}, got {cfg.get(key)!r}')
    cp, cs = curve(candidate, 'psnr_log.pt')[:20], curve(candidate, 'ssim_log.pt')[:20]
    bp, bs = curve(baseline, 'psnr_log.pt')[:20], curve(baseline, 'ssim_log.pt')[:20]
    if min(map(len, (cp, cs, bp, bs))) < 20:
        raise ValueError('both runs require 20 complete epochs')
    ci_rows = images(candidate / 'per_image_metrics' / 'epoch_0020.pt')
    bi_rows = images(baseline / 'per_image_metrics' / 'epoch_0020.pt')
    if set(ci_rows) != set(bi_rows):
        raise ValueError('per-image filenames differ')
    names = sorted(ci_rows)
    deltas = np.array([ci_rows[n]['psnr'] - bi_rows[n]['psnr'] for n in names])
    epoch_delta, ssim_delta = cp - bp, cs - bs
    ci = bootstrap_ci(deltas)
    win_rate = float((deltas > 0).mean())
    profile = json.loads((args.run / 'profile_metrics.json').read_text())
    parameter_by_name = {row['model']: row['parameters'] for row in profile['rows']}
    parameter_overhead = (parameter_by_name['SRPR-SS'] /
                          parameter_by_name['LFMN'] - 1)
    gate = {
        'final_psnr_positive': bool(epoch_delta[-1] > 0),
        'last5_psnr_at_least_0.010': bool(epoch_delta[-5:].mean() >= .010),
        'per_image_ci_lower_positive': bool(ci[0] > 0),
        'per_image_win_rate_at_least_0.60': bool(win_rate >= .60),
        'final_ssim_nonnegative': bool(ssim_delta[-1] >= 0),
        'parameter_overhead_at_most_0.025': bool(parameter_overhead <= .025),
        'mechanism_diagnostics_present': (candidate / 'mechanism_diagnostics.pt').is_file(),
    }
    decision = 'GO_40' if all(gate.values()) else 'NO_GO'
    summary = {
        'candidate': 'N15/SRPR-SS', 'decision': decision,
        'final_psnr_delta': float(epoch_delta[-1]),
        'last5_psnr_delta': float(epoch_delta[-5:].mean()),
        'positive_epoch_ratio': float((epoch_delta > 0).mean()),
        'final_ssim_delta': float(ssim_delta[-1]),
        'per_image_mean_delta': float(deltas.mean()),
        'per_image_median_delta': float(statistics.median(deltas)),
        'per_image_win_rate': win_rate,
        'per_image_bootstrap_95ci': ci,
        'parameter_overhead_fraction': parameter_overhead,
        'gate': gate,
    }
    print(json.dumps(summary, indent=2))
    (args.run / 'n15_decision.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
