#!/usr/bin/env python3
"""Paired, texture-stratified diagnosis for the trained N8 detail head."""
import argparse
import csv
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from data import common
from model.lfmn import Net as BaselineNet
from model.lfmndetailhead import Net as DetailHeadNet
import utility


def load_checkpoint(model, path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def to_tensor(image, device):
    image = common.set_channel(image, n_channels=3)[0]
    return common.np2Tensor(image, rgb_range=255)[0].unsqueeze(0).to(device)


def texture_energy(hr):
    image = hr.detach().float()[0]
    luminance = (
        65.481 * image[0] + 128.553 * image[1] + 24.966 * image[2]
    ) / 255.0 + 16.0
    luminance = luminance[10:-10, 10:-10]
    horizontal = (luminance[:, 1:] - luminance[:, :-1]).abs().mean()
    vertical = (luminance[1:, :] - luminance[:-1, :]).abs().mean()
    return float((horizontal + vertical).item() / 2.0)


def confidence_interval(values, seed=1, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def rank_correlation(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    left_rank = np.empty(len(left), dtype=np.float64)
    right_rank = np.empty(len(right), dtype=np.float64)
    left_rank[np.argsort(left)] = np.arange(len(left), dtype=np.float64)
    right_rank[np.argsort(right)] = np.arange(len(right), dtype=np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def summarize(rows, label):
    delta = np.array([row['candidate_psnr'] - row['baseline_psnr'] for row in rows])
    branch = np.array([row['candidate_psnr'] - row['off_psnr'] for row in rows])
    ssim_delta = np.array([
        row['candidate_ssim'] - row['baseline_ssim'] for row in rows
    ])
    low, high = confidence_interval(delta)
    print(
        '{} | n={} | PSNR delta={:+.6f} | 95% CI=[{:+.6f}, {:+.6f}] | '
        'SSIM delta={:+.6f} | wins={} | branch ON-OFF={:+.6f}'.format(
            label, len(rows), delta.mean(), low, high, ssim_delta.mean(),
            int((delta > 0).sum()), branch.mean(),
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline_checkpoint', type=Path)
    parser.add_argument('candidate_checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--begin', type=int, default=801)
    parser.add_argument('--end', type=int, default=900)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    for path in (args.baseline_checkpoint, args.candidate_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    hr_root = args.data_root / 'DIV2K' / 'DIV2K_valid_HR'
    lr_root = args.data_root / 'DIV2K' / 'DIV2K_valid_LR_bicubic' / 'X4'
    if not hr_root.is_dir() or not lr_root.is_dir():
        raise FileNotFoundError('DIV2K validation data not found under {}'.format(
            args.data_root
        ))

    identifiers = list(range(args.begin, args.end + 1))
    if args.limit:
        identifiers = identifiers[:args.limit]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)
    baseline = load_checkpoint(BaselineNet(scale=4), args.baseline_checkpoint, device)
    candidate = load_checkpoint(DetailHeadNet(scale=4), args.candidate_checkpoint, device)
    rows = []

    with torch.inference_mode():
        for position, identifier in enumerate(identifiers, 1):
            hr_path = hr_root / '{:04d}.png'.format(identifier)
            lr_path = lr_root / '{:04d}x4.png'.format(identifier)
            if not hr_path.is_file() or not lr_path.is_file():
                raise FileNotFoundError('{} or {}'.format(hr_path, lr_path))
            lr = to_tensor(imageio.imread(lr_path), device)
            hr = to_tensor(imageio.imread(hr_path), device)
            height, width = lr.shape[-2:]
            hr = hr[..., :height * 4, :width * 4]

            baseline_sr = utility.quantize(baseline(lr), 255)
            candidate.detail_enabled = True
            candidate_sr = utility.quantize(candidate(lr), 255)
            candidate.detail_enabled = False
            off_sr = utility.quantize(candidate(lr), 255)
            candidate.detail_enabled = True

            row = {
                'image': '{:04d}'.format(identifier),
                'texture': texture_energy(hr),
                'baseline_psnr': utility.calc_psnr(
                    baseline_sr, hr, 4, 255, dataset=None
                ),
                'candidate_psnr': utility.calc_psnr(
                    candidate_sr, hr, 4, 255, dataset=None
                ),
                'off_psnr': utility.calc_psnr(
                    off_sr, hr, 4, 255, dataset=None
                ),
                'baseline_ssim': utility.calc_ssim(
                    baseline_sr, hr, 4, 255, dataset=None
                ),
                'candidate_ssim': utility.calc_ssim(
                    candidate_sr, hr, 4, 255, dataset=None
                ),
                'off_ssim': utility.calc_ssim(
                    off_sr, hr, 4, 255, dataset=None
                ),
            }
            rows.append(row)
            print(
                '[{}/{}] {} texture={:.6f} delta={:+.6f} ON-OFF={:+.6f}'.format(
                    position, len(identifiers), row['image'], row['texture'],
                    row['candidate_psnr'] - row['baseline_psnr'],
                    row['candidate_psnr'] - row['off_psnr'],
                ),
                flush=True,
            )

    print('\nPaired full-set means')
    print('baseline: {:.6f} PSNR / {:.6f} SSIM'.format(
        mean(rows, 'baseline_psnr'), mean(rows, 'baseline_ssim')
    ))
    print('candidate ON: {:.6f} PSNR / {:.6f} SSIM'.format(
        mean(rows, 'candidate_psnr'), mean(rows, 'candidate_ssim')
    ))
    print('candidate OFF: {:.6f} PSNR / {:.6f} SSIM'.format(
        mean(rows, 'off_psnr'), mean(rows, 'off_ssim')
    ))
    summarize(rows, 'all')

    if len(rows) >= 4:
        ordered = sorted(rows, key=lambda row: row['texture'])
        groups = np.array_split(np.array(ordered, dtype=object), 4)
        print('\nTexture quartiles (Q1 smoothest, Q4 most textured)')
        for index, group in enumerate(groups, 1):
            summarize(list(group), 'Q{}'.format(index))

    textures = [row['texture'] for row in rows]
    deltas = [row['candidate_psnr'] - row['baseline_psnr'] for row in rows]
    print('texture/delta Spearman correlation: {:+.6f}'.format(
        rank_correlation(textures, deltas)
    ))

    output = args.output
    if output is None:
        output = args.candidate_checkpoint.parent.parent / 'detail_diagnosis.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('per-image CSV:', output)


if __name__ == '__main__':
    main()
