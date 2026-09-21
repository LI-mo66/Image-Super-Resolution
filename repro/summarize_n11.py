#!/usr/bin/env python3
"""Summarize N11/SRPRv1 against an existing N9 from-scratch B0."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from summarize_candidate_screen import load_curve


def config(path):
    return dict(line.split(': ', 1) for line in path.read_text(encoding='utf-8').splitlines() if ': ' in line)


def stats(run):
    psnr = load_curve(run / 'psnr_log.pt').flatten().tolist()
    ssim = load_curve(run / 'ssim_log.pt').flatten().tolist()
    return psnr, ssim


def load_per_image(run_dir, epoch):
    path = run_dir / 'per_image_metrics' / f'epoch_{epoch:04d}.pt'
    if not path.is_file():
        return {}
    rows = torch.load(path, map_location='cpu', weights_only=True)
    return {(row['dataset'], int(row['scale']), row['filename']):
            (float(row['psnr']), float(row['ssim'])) for row in rows}


def paired_stats(b0, candidate, epoch):
    base = load_per_image(b0, epoch)
    cand = load_per_image(candidate, epoch)
    if not base or not cand or base.keys() != cand.keys():
        return None
    psnr = np.array([cand[k][0] - base[k][0] for k in sorted(base)])
    ssim = np.array([cand[k][1] - base[k][1] for k in sorted(base)])
    return {
        'mean_psnr_delta': float(psnr.mean()),
        'median_psnr_delta': float(np.median(psnr)),
        'positive_psnr_ratio': float((psnr > 0).mean()),
        'mean_ssim_delta': float(ssim.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('screen_dir', type=Path)
    p.add_argument('--baseline-reference', type=Path, required=True)
    args = p.parse_args()
    run = args.screen_dir / 'srprv1'
    if not (run / 'config.txt').is_file():
        raise FileNotFoundError(run / 'config.txt')
    sr_cfg = config(run / 'config.txt')
    expected = {
        'model': 'LFMNSRPRV1', 'epochs': '20', 'data_range': '1-800/801-900',
        'scale': '[4]', 'patch_size': '256', 'batch_size': '4', 'seed': '1',
        'lr': '0.0002', 'scheduler': 'cosine', 'scheduler_t_max': '150',
        'eta_min': '1e-06', 'loss': '1*L1', 'pre_train': '',
    }
    for key, value in expected.items():
        if sr_cfg.get(key) != value:
            raise ValueError(f'{key}: expected {value!r}, got {sr_cfg.get(key)!r}')
    psnr, ssim = stats(run)
    print('N11/SRPRv1 from-scratch run')
    print(f'epochs={len(psnr)} final_psnr={psnr[-1]:.6f} final_ssim={ssim[-1]:.6f}')
    tail = psnr[-5:]
    print(f'last5_psnr={sum(tail)/len(tail):.6f}')
    if args.baseline_reference.exists():
        ref_psnr, ref_ssim = stats(args.baseline_reference)
        print('\nExisting N9 from-scratch B0 paired comparison:')
        print(f'epochs={len(ref_psnr)} final_psnr={ref_psnr[-1]:.6f} final_ssim={ref_ssim[-1]:.6f}')
        print('| epoch | SRPRv1 PSNR | B0 PSNR | delta | SRPRv1 SSIM | B0 SSIM | paired per-image |')
        print('| ---: | ---: | ---: | ---: | ---: | ---: | --- |')
        for epoch in (5, 10, 15, 20):
            if epoch <= len(psnr) and epoch <= len(ref_psnr):
                paired = paired_stats(args.baseline_reference, run, epoch)
                paired_text = json.dumps(paired, sort_keys=True) if paired else 'not available'
                print(f'| {epoch} | {psnr[epoch-1]:.6f} | {ref_psnr[epoch-1]:.6f} | {psnr[epoch-1]-ref_psnr[epoch-1]:+.6f} | {ssim[epoch-1]:.6f} | {ref_ssim[epoch-1]:.6f} | {paired_text} |')
        print('This is a same-protocol paired comparison only if the supplied B0 config matches N9 protocol.')
    else:
        print('\nB0 reference not found; no comparison performed.')


if __name__ == '__main__':
    main()
