#!/usr/bin/env python3
"""Summarize the fixed N12 20-epoch paired screen."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def load_curve(run, name):
    path = run / name
    if not path.is_file():
        raise FileNotFoundError(path)
    value = torch.load(path, map_location='cpu', weights_only=True).float()
    curve = value[:, 0, 0].numpy() if value.ndim == 3 else value.flatten().numpy()
    return curve


def bootstrap_ci(values, seed=1, samples=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def load_images(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'empty or invalid per-image metrics: {path}')
    required = {'filename', 'psnr', 'ssim'}
    if any(not required.issubset(row) for row in rows):
        raise ValueError(f'invalid per-image metric keys: {path}')
    return {row['filename']: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline-reference', type=Path, required=True)
    args = parser.parse_args()
    cand = args.run / 'srprv2'
    b0 = args.baseline_reference
    config_path = cand / 'config.txt'
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = dict(line.split(': ', 1) for line in config_path.read_text().splitlines() if ': ' in line)
    expected = {
        'model': 'LFMNSRPRV2', 'epochs': '20', 'data_range': '1-800/801-900',
        'scale': '[4]', 'patch_size': '256', 'batch_size': '4', 'seed': '1',
        'lr': '0.0002', 'scheduler': 'cosine', 'scheduler_t_max': '150',
        'eta_min': '1e-06', 'loss': '1*L1', 'pre_train': '',
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'{key}: expected {value!r}, got {cfg.get(key)!r}')
    cp, cs = load_curve(cand, 'psnr_log.pt'), load_curve(cand, 'ssim_log.pt')
    bp, bs = load_curve(b0, 'psnr_log.pt'), load_curve(b0, 'ssim_log.pt')
    if min(len(cp), len(cs), len(bp), len(bs)) < 20:
        raise ValueError('both runs must contain at least 20 epochs')
    cp, cs, bp, bs = cp[:20], cs[:20], bp[:20], bs[:20]
    cand_images = load_images(cand / 'per_image_metrics' / 'epoch_0020.pt')
    base_images = load_images(b0 / 'per_image_metrics' / 'epoch_0020.pt')
    names = sorted(set(cand_images) & set(base_images))
    if len(names) != len(cand_images) or len(names) != len(base_images):
        raise ValueError('candidate and baseline per-image filenames do not match')
    deltas = np.array([cand_images[name]['psnr'] - base_images[name]['psnr'] for name in names])
    ci = bootstrap_ci(deltas)

    print('| epoch | SRPRv2 PSNR | B0 PSNR | delta | SRPRv2 SSIM | B0 SSIM | delta |')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for index in range(20):
        print(f'| {index + 1} | {cp[index]:.6f} | {bp[index]:.6f} | {cp[index] - bp[index]:+.6f} | {cs[index]:.6f} | {bs[index]:.6f} | {cs[index] - bs[index]:+.6f} |')
    epoch_delta = cp - bp
    print(f'\nfinal_delta={epoch_delta[-1]:+.6f}')
    print(f'last5_delta={epoch_delta[-5:].mean():+.6f}')
    print(f'positive_epoch_ratio={(epoch_delta > 0).mean():.3f}')
    print(f'per_image_mean_delta={deltas.mean():+.6f}')
    print(f'per_image_median_delta={statistics.median(deltas):+.6f}')
    print(f'per_image_win_rate={(deltas > 0).mean():.3f}')
    print(f'per_image_bootstrap_95ci=[{ci[0]:+.6f},{ci[1]:+.6f}]')
    if (cand / 'mechanism_diagnostics.pt').is_file():
        print('mechanism_diagnostics=present')
    else:
        raise FileNotFoundError(cand / 'mechanism_diagnostics.pt')


if __name__ == '__main__':
    main()
