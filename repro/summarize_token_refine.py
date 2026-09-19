#!/usr/bin/env python3
"""Compare zero-training TAB prototype refinement with a saved baseline."""
import argparse
from pathlib import Path

from summarize_candidate_screen import load_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('refine1', type=Path)
    parser.add_argument('refine3', type=Path)
    args = parser.parse_args()

    baseline_psnr = load_curve(args.baseline, 'psnr_log.pt')
    baseline_ssim = load_curve(args.baseline, 'ssim_log.pt')
    if len(baseline_psnr) != 5 or len(baseline_ssim) != 5:
        raise ValueError('expected a completed five-epoch reusable baseline')
    baseline_p = baseline_psnr[-1]
    baseline_s = baseline_ssim[-1]

    print('Zero-training TAB prototype-refinement evaluation')
    print('| mode | PSNR | delta | SSIM | SSIM delta |')
    print('| --- | ---: | ---: | ---: | ---: |')
    print(f'| stored EMA (baseline) | {baseline_p:.6f} | +0.000000 | '
          f'{baseline_s:.6f} | +0.000000 |')
    for name, path in [('refine1', args.refine1), ('refine3', args.refine3)]:
        psnr = load_curve(path, 'psnr_log.pt')
        ssim = load_curve(path, 'ssim_log.pt')
        if len(psnr) != 1 or len(ssim) != 1:
            raise ValueError(f'{name}: expected one test-only evaluation')
        print(f'| {name} | {psnr[0]:.6f} | {psnr[0] - baseline_p:+.6f} | '
              f'{ssim[0]:.6f} | {ssim[0] - baseline_s:+.6f} |')


if __name__ == '__main__':
    main()
