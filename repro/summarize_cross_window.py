#!/usr/bin/env python3
"""Validate protocol and summarize N6 cross-window screening."""
import argparse
from pathlib import Path

import torch

from summarize_candidate_screen import load_curve

EXPECTED = {
    'epochs': '5', 'test_every': '1000', 'lr': '1e-05',
    'optimizer': 'ADAM', 'seed': '1', 'n_threads': '8',
    'batch_size': '4', 'patch_size': '256',
    'data_range': '1-800/801-810', 'loss': '1*L1',
    'scale': '[4]', 'self_ensemble': 'False', 'chop': 'False',
    'max_train_batches': '0', 'ext': 'img', 'no_augment': 'False',
    'precision': 'single', 'rgb_range': '255', 'weight_decay': '0',
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


def validate(run_dir, model, require_multiplier=False):
    config = read_config(run_dir)
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise ValueError(
                f'{run_dir}: {key}={config.get(key)!r}; expected {expected!r}'
            )
    if config.get('model') != model:
        raise ValueError(
            f'{run_dir}: model={config.get("model")!r}; expected {model!r}'
        )
    if Path(config.get('pre_train', '')).name != 'scale4_model_939.pt':
        raise ValueError(f'{run_dir}: pretrained source mismatch')
    if require_multiplier and config.get('cross_window_lr_mult') != '10.0':
        raise ValueError('candidate cross_window_lr_mult must be 10.0')


def load_run(name, run_dir):
    psnr = load_curve(run_dir, 'psnr_log.pt')
    ssim = load_curve(run_dir, 'ssim_log.pt')
    if len(psnr) != 5 or len(ssim) != 5:
        raise ValueError(f'{name}: expected five completed epochs')
    loss_path = run_dir / 'loss_log.pt'
    loss = torch.load(
        loss_path, map_location='cpu', weights_only=True
    ) if loss_path.is_file() else None
    print(f'\n{name}: {run_dir}')
    print('epoch | PSNR | SSIM | L1')
    for index in range(5):
        l1 = f'{loss[index, 0].item():.6f}' if loss is not None else 'MISSING'
        print(f'{index + 1} | {psnr[index]:.6f} | {ssim[index]:.6f} | {l1}')
    return psnr, ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()

    validate(args.baseline, 'LFMN')
    baseline_psnr, baseline_ssim = load_run('baseline', args.baseline)
    curves = [('baseline', baseline_psnr, baseline_ssim)]
    if args.candidate is not None:
        validate(args.candidate, 'LFMNCrossWindow', require_multiplier=True)
        candidate_psnr, candidate_ssim = load_run(
            'cross_window', args.candidate
        )
        curves.append(('cross_window', candidate_psnr, candidate_ssim))

    print('\nname | final | best | last3 | final delta | last3 delta | SSIM')
    for name, psnr, ssim in curves:
        print(
            f'{name} | {psnr[-1]:.6f} | {psnr.max():.6f} | '
            f'{psnr[-3:].mean():.6f} | '
            f'{psnr[-1] - baseline_psnr[-1]:+.6f} | '
            f'{psnr[-3:].mean() - baseline_psnr[-3:].mean():+.6f} | '
            f'{ssim[-1]:.6f}'
        )
    print('\nBaseline protocol check passed; no baseline training requested.')


if __name__ == '__main__':
    main()
