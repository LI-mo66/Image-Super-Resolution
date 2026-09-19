#!/usr/bin/env python3
"""Summarize a paired test-only baseline/refine1 evaluation."""
import argparse
from pathlib import Path

from summarize_candidate_screen import load_curve


def load_single(run_dir, filename):
    curve = load_curve(run_dir, filename)
    if len(curve) != 1:
        raise ValueError(f'{run_dir}: expected one test-only evaluation')
    return curve[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('refine1', type=Path)
    args = parser.parse_args()

    baseline_p = load_single(args.baseline, 'psnr_log.pt')
    baseline_s = load_single(args.baseline, 'ssim_log.pt')
    refine_p = load_single(args.refine1, 'psnr_log.pt')
    refine_s = load_single(args.refine1, 'ssim_log.pt')

    print('Paired TAB prototype-refinement confirmation')
    print('| mode | PSNR | delta | SSIM | SSIM delta |')
    print('| --- | ---: | ---: | ---: | ---: |')
    print(f'| stored EMA (fresh eval) | {baseline_p:.6f} | +0.000000 | '
          f'{baseline_s:.6f} | +0.000000 |')
    print(f'| refine1 | {refine_p:.6f} | {refine_p - baseline_p:+.6f} | '
          f'{refine_s:.6f} | {refine_s - baseline_s:+.6f} |')


if __name__ == '__main__':
    main()
