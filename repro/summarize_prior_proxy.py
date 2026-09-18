#!/usr/bin/env python3
"""Compare matched scratch proxy-training curves."""
import argparse
from pathlib import Path
import torch


def load_curve(path, name):
    values = torch.load(path / name, map_location='cpu', weights_only=True)
    if values.ndim != 3 or values.shape[1:] != (1, 1):
        raise ValueError(f'unexpected {name} shape: {tuple(values.shape)}')
    return values[:, 0, 0].float()


def config(path):
    return dict(
        line.split(': ', 1)
        for line in (path / 'config.txt').read_text().splitlines()
        if ': ' in line
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('screen_dir', type=Path)
    args = parser.parse_args()
    runs = {
        'baseline': args.screen_dir / 'baseline',
        'prior_proxy': args.screen_dir / 'prior_proxy',
    }
    curves = {}
    configs = {}
    for name, path in runs.items():
        if not path.is_dir():
            raise FileNotFoundError(path)
        curves[name] = {
            'psnr': load_curve(path, 'psnr_log.pt'),
            'ssim': load_curve(path, 'ssim_log.pt'),
        }
        configs[name] = config(path)

    required = ('epochs', 'data_range', 'seed', 'scheduler', 'eta_min',
                'lr', 'batch_size', 'patch_size', 'loss', 'model')
    for key in required:
        if configs['baseline'].get(key) != configs['prior_proxy'].get(key):
            raise ValueError(
                f'protocol mismatch for {key}: '
                f"{configs['baseline'].get(key)!r} vs {configs['prior_proxy'].get(key)!r}"
            )
    psnr_base = curves['baseline']['psnr']
    psnr_candidate = curves['prior_proxy']['psnr']
    if len(psnr_base) != len(psnr_candidate):
        raise ValueError('epoch count differs between runs')

    print('Proxy screening summary (not a final performance claim)')
    print(f'epochs: {len(psnr_base)}; validation: {configs["baseline"].get("data_range")}')
    print('| run | final PSNR | best PSNR | best epoch | last10 mean | last20 mean | final SSIM |')
    print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    tail10 = min(10, len(psnr_base))
    tail20 = min(20, len(psnr_base))
    for name, values in curves.items():
        p, s = values['psnr'], values['ssim']
        best = int(torch.argmax(p).item())
        print(
            f'| {name} | {p[-1]:.4f} | {p[best]:.4f} | {best + 1} | '
            f'{p[-tail10:].mean():.4f} | {p[-tail20:].mean():.4f} | {s[-1]:.5f} |'
        )

    delta = psnr_candidate - psnr_base
    print(f'\nfinal delta: {delta[-1]:+.6f} dB')
    print(f'last10 delta: {delta[-tail10:].mean():+.6f} dB')
    print(f'last20 delta: {delta[-tail20:].mean():+.6f} dB')
    print(f'candidate wins last10: {(delta[-tail10:] > 0).sum().item()}/{tail10}')
    print(f'candidate wins last20: {(delta[-tail20:] > 0).sum().item()}/{tail20}')
    print(f'candidate wins all epochs: {(delta > 0).sum().item()}/{len(delta)}')
    signs = torch.sign(delta)
    reversals = ((signs[1:] * signs[:-1]) < 0).sum().item()
    print(f'curve sign reversals: {reversals}')
    print('\nA positive proxy delta is only a resource-allocation signal; verify with longer training and repeated seeds.')


if __name__ == '__main__':
    main()
