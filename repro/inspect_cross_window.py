#!/usr/bin/env python3
"""Inspect learned N6 gains and its marginal ON/OFF validation effect."""
import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmncrosswindow import Net
import utility


def image_tensor(path, device):
    image = imageio.imread(path)
    if image.ndim == 2:
        image = image[:, :, None].repeat(3, axis=2)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--validation-end', type=int, default=810)
    args = parser.parse_args()
    if not 801 <= args.validation_end <= 900:
        raise ValueError('validation-end must be between 801 and 900')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(scale=4).to(device).eval()
    model.load_state_dict(torch.load(
        args.checkpoint, map_location=device, weights_only=True
    ), strict=True)
    gains = torch.tanh(model.cross_window.stage_gain.detach()).cpu()
    print('stage | mean_abs_gain | max_abs_gain')
    for index, gain in enumerate(gains, 1):
        print(f'{index} | {gain.abs().mean():.8f} | {gain.abs().max():.8f}')

    enabled_psnr, disabled_psnr = [], []
    enabled_ssim, disabled_ssim = [], []
    difference_abs, difference_sq, difference_max, difference_count = 0, 0, 0, 0
    lr_dir = args.data_root / 'DIV2K/DIV2K_valid_LR_bicubic/X4'
    hr_dir = args.data_root / 'DIV2K/DIV2K_valid_HR'
    with torch.inference_mode():
        for number in range(801, args.validation_end + 1):
            lr = image_tensor(lr_dir / f'{number:04d}x4.png', device)
            hr = image_tensor(hr_dir / f'{number:04d}.png', device)
            model.cross_window.enabled = True
            enabled_raw = model(lr)
            model.cross_window.enabled = False
            disabled_raw = model(lr)
            delta = enabled_raw - disabled_raw
            difference_abs += delta.abs().sum().item()
            difference_sq += delta.square().sum().item()
            difference_max = max(difference_max, delta.abs().max().item())
            difference_count += delta.numel()
            enabled = utility.quantize(enabled_raw, 255)
            disabled = utility.quantize(disabled_raw, 255)
            enabled_psnr.append(utility.calc_psnr(enabled, hr, 4, 255))
            disabled_psnr.append(utility.calc_psnr(disabled, hr, 4, 255))
            enabled_ssim.append(utility.calc_ssim(enabled, hr, 4, 255))
            disabled_ssim.append(utility.calc_ssim(disabled, hr, 4, 255))
    model.cross_window.enabled = True

    enabled_p = sum(enabled_psnr) / len(enabled_psnr)
    disabled_p = sum(disabled_psnr) / len(disabled_psnr)
    enabled_s = sum(enabled_ssim) / len(enabled_ssim)
    disabled_s = sum(disabled_ssim) / len(disabled_ssim)
    print('\nraw ON/OFF output difference')
    print('mean_abs: {:.8f}'.format(difference_abs / difference_count))
    print('rms: {:.8f}'.format((difference_sq / difference_count) ** 0.5))
    print('max_abs: {:.8f}'.format(difference_max))
    print('\nmode | PSNR | SSIM')
    print(f'ON | {enabled_p:.6f} | {enabled_s:.6f}')
    print(f'OFF | {disabled_p:.6f} | {disabled_s:.6f}')
    print(f'ON-OFF | {enabled_p - disabled_p:+.6f} | {enabled_s - disabled_s:+.6f}')


if __name__ == '__main__':
    main()
