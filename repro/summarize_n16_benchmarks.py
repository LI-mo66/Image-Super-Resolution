#!/usr/bin/env python3
"""Paired five-benchmark summary for fixed N16 epoch-40 checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
EXPECTED_COUNTS = {'Set5': 5, 'Set14': 14, 'B100': 100,
                   'Urban100': 100, 'Manga109': 109}


def metrics(run):
    path = run / 'per_image_metrics/epoch_0000.pt'
    rows = torch.load(path, map_location='cpu', weights_only=True)
    result = {
        (row['dataset'], int(row['scale']), row['filename']): (
            float(row['psnr']), float(row['ssim'])
        )
        for row in rows
    }
    expected = sum(EXPECTED_COUNTS.values())
    if len(result) != expected:
        raise ValueError('{}: expected {} images, got {}'.format(
            path, expected, len(result)
        ))
    return result


def bootstrap_ci(values, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    c1 = metrics(args.run_dir / 'c1')
    srpr = metrics(args.run_dir / 'srpr_c1')
    if c1.keys() != srpr.keys():
        raise ValueError('C1 and SRPR+C1 benchmark image keys differ')

    rows = []
    all_deltas = []
    for index, dataset in enumerate(DATASETS):
        keys = sorted(key for key in c1 if key[0] == dataset and key[1] == 4)
        if len(keys) != EXPECTED_COUNTS[dataset]:
            raise ValueError('{} image count mismatch'.format(dataset))
        base_psnr = np.array([c1[key][0] for key in keys])
        cand_psnr = np.array([srpr[key][0] for key in keys])
        base_ssim = np.array([c1[key][1] for key in keys])
        cand_ssim = np.array([srpr[key][1] for key in keys])
        delta = cand_psnr - base_psnr
        ci = bootstrap_ci(delta, seed=index + 1)
        all_deltas.extend(delta.tolist())
        rows.append({
            'dataset': dataset,
            'images': len(keys),
            'c1_psnr': float(base_psnr.mean()),
            'srpr_c1_psnr': float(cand_psnr.mean()),
            'psnr_delta': float(delta.mean()),
            'c1_ssim': float(base_ssim.mean()),
            'srpr_c1_ssim': float(cand_ssim.mean()),
            'ssim_delta': float((cand_ssim - base_ssim).mean()),
            'median_delta': float(np.median(delta)),
            'win_rate': float((delta > 0).mean()),
            'bootstrap_95ci': [float(ci[0]), float(ci[1])],
        })

    by_name = {row['dataset']: row for row in rows}
    positive = sum(row['psnr_delta'] > 0 for row in rows)
    no_large_regression = all(row['psnr_delta'] > -0.01 for row in rows)
    texture_positive = (
        by_name['Urban100']['psnr_delta'] > 0
        and by_name['Manga109']['psnr_delta'] > 0
    )
    if positive >= 4 and texture_positive and no_large_regression:
        decision = 'BROAD_EXTERNAL_SUPPORT'
    elif positive >= 3 and no_large_regression:
        decision = 'MIXED_EXTERNAL_SUPPORT'
    else:
        decision = 'WEAK_EXTERNAL_SUPPORT'
    pooled = np.asarray(all_deltas, dtype=np.float64)
    pooled_ci = bootstrap_ci(pooled, seed=17)
    payload = {
        'comparison': 'epoch40 SRPRv2+C1 minus epoch40 B0+C1',
        'checkpoint_selection': 'fixed preregistered epoch40 endpoint',
        'self_ensemble': False,
        'datasets': rows,
        'positive_datasets': positive,
        'texture_datasets_positive': texture_positive,
        'pooled_image_mean_delta': float(pooled.mean()),
        'pooled_image_win_rate': float((pooled > 0).mean()),
        'pooled_bootstrap_95ci': [float(pooled_ci[0]), float(pooled_ci[1])],
        'decision': decision,
    }
    print('N16 fixed epoch-40 five-benchmark evaluation')
    print('comparison=SRPRv2+C1 minus B0+C1; self_ensemble=False')
    print('| dataset | C1 PSNR | SRPR+C1 PSNR | delta | SSIM delta | wins | 95% CI |')
    print('|---|---:|---:|---:|---:|---:|---|')
    for row in rows:
        print('| {dataset} | {c1_psnr:.6f} | {srpr_c1_psnr:.6f} | '
              '{psnr_delta:+.6f} | {ssim_delta:+.7f} | {win_rate:.1%} | '
              '[{bootstrap_95ci[0]:+.6f},{bootstrap_95ci[1]:+.6f}] |'.format(**row))
    print('positive_datasets={}/5'.format(positive))
    print('texture_datasets_positive={}'.format(texture_positive))
    print('pooled_image_mean_delta={:+.6f}'.format(pooled.mean()))
    print('pooled_image_win_rate={:.1%}'.format((pooled > 0).mean()))
    print('pooled_bootstrap_95ci=[{:+.6f},{:+.6f}]'.format(*pooled_ci))
    print('decision={}'.format(decision))
    (args.run_dir / 'benchmark_summary.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    main()
