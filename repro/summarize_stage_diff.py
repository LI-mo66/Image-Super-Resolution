#!/usr/bin/env python3
"""Validate a reusable baseline and compare stage-difference screening."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from summarize_candidate_screen import load_curve

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'model': 'LFMN', 'epochs': '5', 'test_every': '1000', 'lr': '1e-05',
    'optimizer': 'ADAM', 'seed': '1', 'n_threads': '8', 'batch_size': '4',
    'patch_size': '256', 'data_range': '1-800/801-810', 'loss': '1*L1',
    'scale': '[4]', 'self_ensemble': 'False', 'chop': 'False',
    'max_train_batches': '0', 'ext': 'img', 'no_augment': 'False',
    'precision': 'single', 'rgb_range': '255', 'weight_decay': '0',
}


def read_config(path):
    return dict(
        line.split(': ', 1)
        for line in (path / 'config.txt').read_text().splitlines()
        if ': ' in line
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()

    baseline_config = read_config(args.baseline)
    for key, expected in EXPECTED.items():
        if baseline_config.get(key) != expected:
            raise ValueError(
                f'baseline mismatch: {key}={baseline_config.get(key)!r}; '
                f'expected {expected!r}'
            )
    if Path(baseline_config.get('pre_train', '')).name != 'scale4_model_939.pt':
        raise ValueError('baseline pretrained source mismatch')

    runs = [('baseline', args.baseline)]
    if args.candidate is not None:
        candidate_config = read_config(args.candidate)
        for key, expected in EXPECTED.items():
            if key == 'model':
                expected = 'LFMNStageDiff'
            if candidate_config.get(key) != expected:
                raise ValueError(
                    f'candidate mismatch: {key}={candidate_config.get(key)!r}; '
                    f'expected {expected!r}'
                )
        if candidate_config.get('stage_diff_lr_mult') != '10.0':
            raise ValueError('candidate stage_diff_lr_mult must be 10.0')
        runs.append(('stage_diff', args.candidate))

    curves = {}
    for name, path in runs:
        psnr = load_curve(path, 'psnr_log.pt')
        ssim = load_curve(path, 'ssim_log.pt')
        if len(psnr) != 5 or len(ssim) != 5:
            raise ValueError(f'{name}: expected five completed epochs')
        curves[name] = (psnr, ssim)
        loss_path = path / 'loss_log.pt'
        loss = torch.load(
            loss_path, map_location='cpu', weights_only=True
        ) if loss_path.exists() else None
        print(f'\n{name}: {path}')
        print('epoch | PSNR | SSIM | L1')
        for index in range(5):
            l1 = f'{loss[index, 0].item():.6f}' if loss is not None else 'MISSING'
            print(f'{index + 1} | {psnr[index]:.6f} | {ssim[index]:.6f} | {l1}')

    baseline_psnr = curves['baseline'][0]
    print('\nname | final | best | last3 | final delta | last3 delta | SSIM')
    for name, (psnr, ssim) in curves.items():
        print(
            f'{name} | {psnr[-1]:.6f} | {psnr.max():.6f} | '
            f'{psnr[-3:].mean():.6f} | {psnr[-1] - baseline_psnr[-1]:+.6f} | '
            f'{psnr[-3:].mean() - baseline_psnr[-3:].mean():+.6f} | '
            f'{ssim[-1]:.6f}'
        )

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ROOT / 'LFMN/model/scale4_model_939.pt'
        manifest = {
            'baseline': str(args.baseline.resolve()),
            'baseline_config': baseline_config,
            'commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
            ).strip(),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'decision_rule': 'Advance only if last-3 delta is about +0.01 dB or better; '
                             'otherwise stop and diagnose without a long run.',
        }
        args.manifest.write_text(
            json.dumps(manifest, indent=2), encoding='utf-8'
        )
    print('\nBaseline protocol check passed; no baseline training requested.')


if __name__ == '__main__':
    main()
