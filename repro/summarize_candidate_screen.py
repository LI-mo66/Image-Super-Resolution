#!/usr/bin/env python3
"""Summarize matched candidate-screen PSNR/SSIM logs against the baseline."""
import argparse
from pathlib import Path

import torch


def load_curve(run_dir, filename):
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    values = torch.load(path, map_location='cpu', weights_only=True)
    if values.ndim != 3 or values.shape[1:] != (1, 1):
        raise ValueError('unexpected log shape {} in {}'.format(values.shape, path))
    return values[:, 0, 0].float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('screen_dir', type=Path)
    args = parser.parse_args()

    names = ('baseline', 'overlap_control', 'shared_frequency', 'hf_loss')
    curves = {}
    for name in names:
        run_dir = args.screen_dir / name
        curves[name] = {
            'psnr': load_curve(run_dir, 'psnr_log.pt'),
            'ssim': load_curve(run_dir, 'ssim_log.pt'),
        }

    baseline = curves['baseline']['psnr']
    print('| candidate | final PSNR | best PSNR | last-3 mean | final delta | last-3 delta | final SSIM |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    for name in names:
        psnr = curves[name]['psnr']
        ssim = curves[name]['ssim']
        if len(psnr) != len(baseline):
            raise ValueError('{} epoch count differs from baseline'.format(name))
        tail = min(3, len(psnr))
        final_delta = psnr[-1] - baseline[-1]
        tail_delta = psnr[-tail:].mean() - baseline[-tail:].mean()
        print(
            '| {} | {:.4f} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} | {:.5f} |'.format(
                name,
                psnr[-1].item(),
                psnr.max().item(),
                psnr[-tail:].mean().item(),
                final_delta.item(),
                tail_delta.item(),
                ssim[-1].item(),
            )
        )


if __name__ == '__main__':
    main()
