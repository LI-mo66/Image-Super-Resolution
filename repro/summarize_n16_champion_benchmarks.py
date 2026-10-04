#!/usr/bin/env python3
"""Summarize the fixed N16 epoch-1000 benchmarks and classify A/B/C."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
EXPECTED_COUNTS = {'Set5': 5, 'Set14': 14, 'B100': 100,
                   'Urban100': 100, 'Manga109': 109}
LFMN_PAPER = {
    'Set5': (32.64, 0.9004), 'Set14': (28.95, 0.7893),
    'B100': (27.80, 0.7438), 'Urban100': (26.91, 0.8087),
    'Manga109': (31.45, 0.9193),
}


def bootstrap_ci(values, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    path = args.run_dir / 'champion/per_image_metrics/epoch_0000.pt'
    rows = torch.load(path, map_location='cpu', weights_only=True)
    metrics = {
        (row['dataset'], int(row['scale']), row['filename']):
        (float(row['psnr']), float(row['ssim']))
        for row in rows
    }
    if len(metrics) != sum(EXPECTED_COUNTS.values()):
        raise ValueError('expected 328 unique image rows, got {}'.format(len(metrics)))

    results = []
    all_psnr = []
    for index, dataset in enumerate(DATASETS):
        keys = sorted(key for key in metrics if key[0] == dataset and key[1] == 4)
        if len(keys) != EXPECTED_COUNTS[dataset]:
            raise ValueError('{} image count mismatch'.format(dataset))
        psnr = np.array([metrics[key][0] for key in keys])
        ssim = np.array([metrics[key][1] for key in keys])
        ci = bootstrap_ci(psnr, seed=index + 31)
        reference_psnr, reference_ssim = LFMN_PAPER[dataset]
        all_psnr.extend(psnr.tolist())
        results.append({
            'dataset': dataset, 'images': len(keys),
            'psnr': float(psnr.mean()), 'ssim': float(ssim.mean()),
            'psnr_bootstrap_95ci': [float(ci[0]), float(ci[1])],
            'lfmn_paper_psnr': reference_psnr,
            'lfmn_paper_ssim': reference_ssim,
            'psnr_minus_lfmn_paper': float(psnr.mean() - reference_psnr),
            'ssim_minus_lfmn_paper': float(ssim.mean() - reference_ssim),
        })
    by_name = {row['dataset']: row for row in results}
    above = sum(row['psnr_minus_lfmn_paper'] >= 0 for row in results)
    texture_above = (
        by_name['Urban100']['psnr_minus_lfmn_paper'] >= 0
        and by_name['Manga109']['psnr_minus_lfmn_paper'] >= 0
    )
    no_material_regression = all(
        row['psnr_minus_lfmn_paper'] >= -0.02 for row in results
    )
    if above >= 4 and texture_above and no_material_regression:
        category = 'A'
        decision = 'BROADLY_EXCEEDS_LFMN_PAPER_REFERENCE'
    elif above >= 3 or texture_above:
        category = 'B'
        decision = 'USEFUL_BUT_NOT_BROADLY_DOMINANT'
    else:
        category = 'C'
        decision = 'LONG_RUN_DOES_NOT_CLOSE_THE_ABSOLUTE_GAP'
    payload = {
        'candidate': 'N16 frozen champion: SRPRv2+C1',
        'checkpoint_selection': 'fixed epoch 1000 endpoint',
        'evaluation': 'x4, Y-channel PSNR/SSIM, no self-ensemble',
        'reference': 'LFMN paper reported x4 values; not a paired retraining',
        'datasets': results, 'datasets_at_or_above_lfmn_paper': above,
        'texture_datasets_at_or_above_lfmn_paper': texture_above,
        'no_dataset_below_reference_by_more_than_0.02db': no_material_regression,
        'category': category, 'decision': decision,
    }
    (args.run_dir / 'benchmark_summary.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print('N16 fixed epoch-1000 five-benchmark evaluation')
    print('self_ensemble=False; reference=LFMN paper (not paired retraining)')
    print('| dataset | N16 PSNR/SSIM | LFMN paper | PSNR gap | SSIM gap |')
    print('|---|---:|---:|---:|---:|')
    for row in results:
        print('| {dataset} | {psnr:.4f}/{ssim:.4f} | '
              '{lfmn_paper_psnr:.2f}/{lfmn_paper_ssim:.4f} | '
              '{psnr_minus_lfmn_paper:+.4f} | '
              '{ssim_minus_lfmn_paper:+.4f} |'.format(**row))
    print('datasets_at_or_above_lfmn_paper={}/5'.format(above))
    print('texture_datasets_at_or_above_lfmn_paper={}'.format(texture_above))
    print('category={}'.format(category))
    print('decision={}'.format(decision))


if __name__ == '__main__':
    main()
