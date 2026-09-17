#!/usr/bin/env python3
"""Server diagnostic: prior magnitude and same-checkpoint enabled/disabled A/B."""
import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnpriorupdate import Net
from utility import quantize, calc_psnr, calc_ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(scale=4).to(device).eval()
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    stats = []

    def inspect(module, inputs, delta):
        fs = inputs[0]
        rms = lambda t: t.square().mean().sqrt().item()
        changes = []
        for sfml in net.sfmls[4:]:
            b0, g0 = sfml(fs)
            b1, g1 = sfml(fs + delta)
            changes.append((rms(b1-b0), rms(g1-g0)))
        stats.append((rms(delta) / max(rms(fs), 1e-12), changes))

    net.prior_update.register_forward_hook(inspect)
    totals = [0.0, 0.0]
    def tensor(path):
        array = imageio.imread(path)
        return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        for index in range(801, 811):
            root = args.data_root / 'DIV2K'
            lr = tensor(root / f'DIV2K_valid_LR_bicubic/X4/{index:04d}x4.png')
            hr = tensor(root / f'DIV2K_valid_HR/{index:04d}.png')
            net.update_enabled = True
            on = quantize(net(lr), 255)
            net.update_enabled = False
            off = quantize(net(lr), 255)
            # Matches current DIV2K non-benchmark calc_psnr: RGB, shave scale+6.
            p_on, p_off = [calc_psnr(y, hr, 4, 255) for y in (on, off)]
            totals[0] += p_on
            totals[1] += p_off
            ratio, changes = stats[-1]
            print(f'{index}: on={p_on:.6f} off={p_off:.6f} delta={p_on-p_off:+.6f} '
                  f'delta_prior/prior RMS={ratio:.8f} '
                  f'SSIM_on={calc_ssim(on, hr, 4, 255):.6f} beta/gamma_delta_RMS={changes}')
    print('Mean on/off/delta:', totals[0]/10, totals[1]/10, (totals[0]-totals[1])/10)
    print('Same-checkpoint ablation only; not a separately trained baseline comparison.')


if __name__ == '__main__':
    main()
