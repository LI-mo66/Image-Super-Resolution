#!/usr/bin/env python3
"""Summarize the preregistered N9 20-epoch scratch comparison."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from summarize_candidate_screen import load_curve


COMMON_EXPECTED = {
    'epochs': '20', 'data_range': '1-800/801-900', 'scale': '[4]',
    'patch_size': '256', 'batch_size': '4', 'ext': 'img',
    'lr': '0.0002', 'scheduler': 'cosine', 'scheduler_t_max': '150',
    'eta_min': '1e-06', 'loss': '1*L1', 'seed': '1',
    'max_train_batches': '0', 'pre_train': '',
    'save_per_image_metrics': 'True',
}
RUNS = {
    'baseline': 'LFMN',
    'pcstr': 'LFMNPCSTR',
    'pcstr_wide': 'LFMNPCSTRWide',
    'mix_control': 'LFMNMixControl',
}


def read_config(run_dir):
    path = run_dir / 'config.txt'
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(
        line.split(': ', 1)
        for line in path.read_text(encoding='utf-8').splitlines()
        if ': ' in line
    )


def validate(run_dir, model):
    config = read_config(run_dir)
    expected = {**COMMON_EXPECTED, 'model': model}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f'{run_dir}: {key}={config.get(key)!r}; expected {value!r}'
            )


def load_per_image(run_dir, epoch, expected_count=100):
    path = run_dir / 'per_image_metrics' / f'epoch_{epoch:04d}.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = torch.load(path, map_location='cpu', weights_only=True)
    result = {}
    for row in rows:
        key = (row['dataset'], int(row['scale']), row['filename'])
        if key in result:
            raise ValueError(f'duplicate metric key {key!r} in {path}')
        result[key] = (float(row['psnr']), float(row['ssim']))
    if len(result) != expected_count:
        raise ValueError(
            f'{path}: expected {expected_count} images, found {len(result)}'
        )
    return result


def paired_arrays(baseline, candidate):
    if baseline.keys() != candidate.keys():
        raise ValueError('per-image keys do not match the baseline')
    keys = sorted(baseline)
    psnr = np.array([
        candidate[key][0] - baseline[key][0] for key in keys
    ], dtype=np.float64)
    ssim = np.array([
        candidate[key][1] - baseline[key][1] for key in keys
    ], dtype=np.float64)
    return keys, psnr, ssim


def bootstrap_mean_ci(values, repetitions=10_000):
    rng = np.random.default_rng(1)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, (0.025, 0.975)))


def high_frequency_energy(data_root, filename):
    stem = Path(filename).stem
    path = data_root / 'DIV2K' / 'DIV2K_valid_HR' / f'{stem}.png'
    if not path.is_file():
        matches = list((data_root / 'DIV2K' / 'DIV2K_valid_HR').glob(
            f'{stem}.*'
        ))
        if len(matches) != 1:
            raise FileNotFoundError(path)
        path = matches[0]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'failed to read {path}')
    laplacian = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    return float(np.abs(laplacian).mean())


def texture_groups(keys, data_root):
    energy = np.array([
        high_frequency_energy(data_root, key[2]) for key in keys
    ])
    ordered = np.argsort(energy, kind='stable')
    splits = np.array_split(ordered, 3)
    return dict(zip(('low', 'medium', 'high'), splits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('screen_dir', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()

    curves = {}
    per_image = {}
    for name, model in RUNS.items():
        run_dir = args.screen_dir / name
        validate(run_dir, model)
        curves[name] = {
            'psnr': load_curve(run_dir, 'psnr_log.pt').numpy(),
            'ssim': load_curve(run_dir, 'ssim_log.pt').numpy(),
        }
        if len(curves[name]['psnr']) != 20:
            raise ValueError(f'{run_dir}: expected 20 completed epochs')
        per_image[name] = {
            epoch: load_per_image(run_dir, epoch)
            for epoch in range(16, 21)
        }

    checkpoints = (5, 10, 15, 20)
    baseline_psnr = curves['baseline']['psnr']
    baseline_ssim = curves['baseline']['ssim']
    print('| model | e5 Δ | e10 Δ | e15 Δ | e20 Δ | last5 Δ | '
          'slope Δ/epoch | e20 SSIM Δ |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    decisions = {}
    for name in RUNS:
        psnr = curves[name]['psnr']
        ssim = curves[name]['ssim']
        deltas = psnr - baseline_psnr
        tail_delta = float(deltas[15:20].mean())
        slope_delta = float(np.polyfit(
            np.arange(16, 21), deltas[15:20], 1
        )[0])
        print('| {} | {} |'.format(
            name,
            ' | '.join(
                [f'{deltas[epoch - 1]:+.6f}' for epoch in checkpoints]
                + [f'{tail_delta:+.6f}', f'{slope_delta:+.6f}',
                   f'{ssim[19] - baseline_ssim[19]:+.6f}']
            ),
        ))
        decisions[name] = (tail_delta, slope_delta, deltas[15:20],
                           float(ssim[19] - baseline_ssim[19]))

    baseline_epoch20 = per_image['baseline'][20]
    baseline_keys = sorted(baseline_epoch20)
    groups = texture_groups(baseline_keys, args.data_root)
    print('\n| candidate | mean Δ | median Δ | positive | 95% CI | '
          'low/mid/high texture Δ | decision |')
    print('| --- | ---: | ---: | ---: | --- | --- | --- |')
    for name in RUNS:
        if name == 'baseline':
            continue
        keys, psnr_delta, ssim_delta = paired_arrays(
            baseline_epoch20, per_image[name][20]
        )
        if keys != baseline_keys:
            raise ValueError(f'{name}: sorted per-image keys changed')
        low, high = bootstrap_mean_ci(psnr_delta)
        texture = [float(psnr_delta[index].mean()) for index in groups.values()]
        tail, slope, tail_points, final_ssim = decisions[name]
        median = float(np.median(psnr_delta))
        positive = float((psnr_delta > 0).mean())
        if tail <= 0 and slope <= 0:
            decision = 'STOP: nonpositive tail and slope'
        elif final_ssim < -0.0001:
            decision = 'STOP: SSIM gate'
        elif tail >= 0.01 and median > 0 and slope >= -0.001:
            decision = 'EXTEND to 40'
        elif 0 < tail < 0.01 and slope >= 0.001 and (tail_points > 0).sum() >= 4:
            decision = 'GRAY EXTEND once'
        else:
            decision = 'NO automatic extension'
        print('| {} | {:+.6f} | {:+.6f} | {:.1%} | '
              '[{:+.6f}, {:+.6f}] | {:+.6f}/{:+.6f}/{:+.6f} | {} |'.format(
                  name, psnr_delta.mean(), median, positive, low, high,
                  *texture, decision,
              ))
        if ssim_delta.mean() < -0.0001:
            print(f'warning: {name} paired mean SSIM delta '
                  f'{ssim_delta.mean():+.6f}')

    print('\nThese are preregistered resource decisions, not proof of a paper claim.')


if __name__ == '__main__':
    main()
