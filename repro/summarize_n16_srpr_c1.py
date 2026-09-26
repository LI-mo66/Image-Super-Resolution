#!/usr/bin/env python3
"""Summarize the paired 20-epoch N16 C1/SRPR+C1 interaction screen."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


RUNS = {
    'c1': ('LFMN', 'output'),
    'srpr_c1': ('LFMNSRPRV2', 'output'),
}
COMMON = {
    'data_range': '1-800/801-900',
    'scale': '[4]',
    'patch_size': '256',
    'batch_size': '4',
    'ext': 'img',
    'epochs': '20',
    'lr': '0.0002',
    'scheduler': 'cosine',
    'scheduler_t_max': '150',
    'eta_min': '1e-06',
    'loss': '1*L1',
    'seed': '1',
    'pre_train': '',
    'save_per_image_metrics': 'True',
    'rgcrd_lambda_output': '0.1',
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


def validate(run_dir, model, mode):
    config = read_config(run_dir)
    expected = {**COMMON, 'model': model, 'rgcrd_mode': mode}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                '{}: {}={!r}; expected {!r}'.format(
                    run_dir, key, config.get(key), value
                )
            )


def curve(run_dir, name):
    values = torch.load(
        run_dir / name, map_location='cpu', weights_only=True
    )
    if values.ndim != 3 or values.shape[1:] != (1, 1):
        raise ValueError('{} has unexpected shape {}'.format(name, values.shape))
    result = values[:, 0, 0].double().numpy()
    if len(result) != 20:
        raise ValueError('{} is incomplete: {} epochs'.format(run_dir, len(result)))
    return result


def per_image(run_dir):
    path = run_dir / 'per_image_metrics' / 'epoch_0020.pt'
    rows = torch.load(path, map_location='cpu', weights_only=True)
    result = {
        (row['dataset'], int(row['scale']), row['filename']): (
            float(row['psnr']), float(row['ssim'])
        )
        for row in rows
    }
    if len(result) != 100:
        raise ValueError('{}: expected 100 validation images'.format(path))
    return result


def confidence_interval(values):
    rng = np.random.default_rng(1)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    return np.quantile(values[indices].mean(axis=1), (0.025, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('screen_dir', type=Path)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()

    curves = {}
    images = {}
    for name, (model, mode) in RUNS.items():
        run_dir = args.screen_dir / name
        validate(run_dir, model, mode)
        curves[name] = {
            'psnr': curve(run_dir, 'psnr_log.pt'),
            'ssim': curve(run_dir, 'ssim_log.pt'),
        }
        images[name] = per_image(run_dir)

    if images['c1'].keys() != images['srpr_c1'].keys():
        raise ValueError('C1 and SRPR+C1 per-image keys differ')
    ordered_keys = sorted(images['c1'])
    image_psnr_delta = np.array([
        images['srpr_c1'][key][0] - images['c1'][key][0]
        for key in ordered_keys
    ])
    image_ssim_delta = np.array([
        images['srpr_c1'][key][1] - images['c1'][key][1]
        for key in ordered_keys
    ])
    psnr_delta = curves['srpr_c1']['psnr'] - curves['c1']['psnr']
    ssim_delta = curves['srpr_c1']['ssim'] - curves['c1']['ssim']
    ci_low, ci_high = confidence_interval(image_psnr_delta)

    gates = {
        'final_delta_ge_0.010': bool(psnr_delta[-1] >= 0.010),
        'last5_delta_ge_0.010': bool(psnr_delta[-5:].mean() >= 0.010),
        'positive_epochs_ge_15': bool((psnr_delta > 0).sum() >= 15),
        'bootstrap_ci_low_gt_0': bool(ci_low > 0),
        'win_rate_ge_0.60': bool((image_psnr_delta > 0).mean() >= 0.60),
        'final_ssim_nonnegative': bool(ssim_delta[-1] >= 0),
    }
    if all(gates.values()):
        decision = 'PROMOTE_TO_MATCHED_40E_CONTINUATION'
    elif (
        psnr_delta[-1] > 0
        and psnr_delta[-5:].mean() > 0
        and ci_high > 0
    ):
        decision = 'MARGINAL_ONE_ADDITIONAL_SEED_ONLY'
    else:
        decision = 'STOP_SRPR_ROUTE'

    payload = {
        'comparison': 'SRPRv2+C1 minus B0+C1',
        'epochs': 20,
        'final': {
            'c1_psnr': float(curves['c1']['psnr'][-1]),
            'srpr_c1_psnr': float(curves['srpr_c1']['psnr'][-1]),
            'psnr_delta': float(psnr_delta[-1]),
            'ssim_delta': float(ssim_delta[-1]),
        },
        'last5_psnr_delta': float(psnr_delta[-5:].mean()),
        'positive_epochs': int((psnr_delta > 0).sum()),
        'per_image': {
            'mean_psnr_delta': float(image_psnr_delta.mean()),
            'median_psnr_delta': float(np.median(image_psnr_delta)),
            'win_rate': float((image_psnr_delta > 0).mean()),
            'bootstrap_95ci': [float(ci_low), float(ci_high)],
            'mean_ssim_delta': float(image_ssim_delta.mean()),
        },
        'gates': gates,
        'decision': decision,
    }

    print('N16 SRPRv2 x C1 interaction screen')
    print('comparison=SRPRv2+C1 minus B0+C1')
    for epoch in (1, 5, 10, 15, 20):
        print(
            'epoch_{:02d}: c1={:.6f}; srpr_c1={:.6f}; delta={:+.6f}'.format(
                epoch,
                curves['c1']['psnr'][epoch - 1],
                curves['srpr_c1']['psnr'][epoch - 1],
                psnr_delta[epoch - 1],
            )
        )
    print('final_delta={:+.6f}'.format(psnr_delta[-1]))
    print('last5_delta={:+.6f}'.format(psnr_delta[-5:].mean()))
    print('positive_epochs={}/20'.format((psnr_delta > 0).sum()))
    print('final_ssim_delta={:+.7f}'.format(ssim_delta[-1]))
    print('per_image_mean_delta={:+.6f}'.format(image_psnr_delta.mean()))
    print('per_image_median_delta={:+.6f}'.format(np.median(image_psnr_delta)))
    print('per_image_win_rate={:.1%}'.format((image_psnr_delta > 0).mean()))
    print('per_image_bootstrap_95ci=[{:+.6f},{:+.6f}]'.format(
        ci_low, ci_high
    ))
    print('per_image_ssim_delta={:+.7f}'.format(image_ssim_delta.mean()))
    for name, passed in gates.items():
        print('gate:{}={}'.format(name, 'PASS' if passed else 'FAIL'))
    print('decision={}'.format(decision))

    output = args.json_output or args.screen_dir / 'summary.json'
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
