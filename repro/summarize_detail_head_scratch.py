#!/usr/bin/env python3
"""Compare N8's first 20 scratch epochs with the existing baseline."""
import argparse
from pathlib import Path

from summarize_candidate_screen import load_curve


BASELINE_EXPECTED = {
    'model': 'LFMN', 'epochs': '150', 'data_range': '1-800/801-900',
    'scale': '[4]', 'patch_size': '256', 'batch_size': '4',
    'n_threads': '8', 'ext': 'img', 'lr': '0.0002',
    'scheduler': 'cosine', 'eta_min': '1e-06', 'loss': '1*L1',
    'seed': '1', 'max_train_batches': '0', 'pre_train': '',
}
CANDIDATE_EXPECTED = {
    **BASELINE_EXPECTED,
    'model': 'LFMNDetailHead', 'epochs': '20',
    'scheduler_t_max': '150',
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


def validate(run_dir, expected):
    config = read_config(run_dir)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f'{run_dir}: {key}={config.get(key)!r}; expected {value!r}'
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()

    validate(args.baseline, BASELINE_EXPECTED)
    baseline_psnr = load_curve(args.baseline, 'psnr_log.pt')
    baseline_ssim = load_curve(args.baseline, 'ssim_log.pt')
    if len(baseline_psnr) != 150 or len(baseline_ssim) != 150:
        raise ValueError('expected the completed 150-epoch baseline curves')
    print('Existing scratch baseline protocol and 150-epoch curves: OK')
    if args.candidate is None:
        print('No baseline training requested.')
        return

    validate(args.candidate, CANDIDATE_EXPECTED)
    candidate_psnr = load_curve(args.candidate, 'psnr_log.pt')
    candidate_ssim = load_curve(args.candidate, 'ssim_log.pt')
    if len(candidate_psnr) != 20 or len(candidate_ssim) != 20:
        raise ValueError('expected 20 completed candidate epochs')

    print('\nepoch | baseline PSNR | candidate PSNR | delta | baseline SSIM | candidate SSIM')
    for index in range(20):
        print(
            f'{index + 1} | {baseline_psnr[index]:.6f} | '
            f'{candidate_psnr[index]:.6f} | '
            f'{candidate_psnr[index] - baseline_psnr[index]:+.6f} | '
            f'{baseline_ssim[index]:.6f} | {candidate_ssim[index]:.6f}'
        )
    baseline_tail = baseline_psnr[17:20].mean()
    candidate_tail = candidate_psnr[17:20].mean()
    print('\nname | epoch20 | best1-20 | last3 | epoch20 delta | last3 delta | epoch20 SSIM')
    print(
        f'baseline | {baseline_psnr[19]:.6f} | {baseline_psnr[:20].max():.6f} | '
        f'{baseline_tail:.6f} | +0.000000 | +0.000000 | '
        f'{baseline_ssim[19]:.6f}'
    )
    print(
        f'detail_head | {candidate_psnr[19]:.6f} | {candidate_psnr.max():.6f} | '
        f'{candidate_tail:.6f} | '
        f'{candidate_psnr[19] - baseline_psnr[19]:+.6f} | '
        f'{candidate_tail - baseline_tail:+.6f} | '
        f'{candidate_ssim[19]:.6f}'
    )


if __name__ == '__main__':
    main()
