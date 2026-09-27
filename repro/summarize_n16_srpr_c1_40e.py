#!/usr/bin/env python3
"""Validate and summarize the matched N16 epoch-20 to epoch-40 continuation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


MODELS = {'c1': 'LFMN', 'srpr_c1': 'LFMNSRPRV2'}
COMMON = {
    'data_range': '1-800/801-900',
    'scale': '[4]',
    'patch_size': '256',
    'batch_size': '4',
    'ext': 'img',
    'epochs': '40',
    'lr': '0.0002',
    'scheduler': 'cosine',
    'scheduler_t_max': '150',
    'eta_min': '1e-06',
    'loss': '1*L1',
    'seed': '1',
    'pre_train': '',
    'resume': '20',
    'resume_data_epochs': '20',
    'save_per_image_metrics': 'True',
    'rgcrd_mode': 'output',
    'rgcrd_lambda_output': '0.1',
}


def read_config(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            result[key] = value
    return result


def curve(run, name):
    path = run / name
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.ndim != 3 or tuple(value.shape[1:]) != (1, 1):
        raise ValueError('{} has unexpected shape {}'.format(path, value.shape))
    return value[:, 0, 0].double().numpy()


def per_image(run, epoch):
    path = run / 'per_image_metrics' / 'epoch_{:04d}.pt'.format(epoch)
    rows = torch.load(path, map_location='cpu', weights_only=True)
    result = {
        (row['dataset'], int(row['scale']), row['filename']): (
            float(row['psnr']), float(row['ssim'])
        )
        for row in rows
    }
    if len(result) != 100:
        raise ValueError('{}: expected 100 images, got {}'.format(
            path, len(result)
        ))
    return result


def bootstrap_ci(values):
    rng = np.random.default_rng(1)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--source-20', type=Path, required=True)
    parser.add_argument('--json-output', type=Path)
    args = parser.parse_args()

    curves = {}
    source_curves = {}
    images = {}
    for name, model in MODELS.items():
        run = args.run_dir / name
        source = args.source_20 / name
        config = read_config(run / 'config.txt')
        source_config = read_config(source / 'config.txt')
        expected = {**COMMON, 'model': model, 'load': name}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    '{}: {}={!r}; expected {!r}'.format(
                        run, key, config.get(key), value
                    )
                )
        if config.get('n_threads') != source_config.get('n_threads'):
            raise ValueError('{} changed n_threads during continuation'.format(run))
        curves[name] = {
            'psnr': curve(run, 'psnr_log.pt'),
            'ssim': curve(run, 'ssim_log.pt'),
        }
        source_curves[name] = {
            'psnr': curve(source, 'psnr_log.pt'),
            'ssim': curve(source, 'ssim_log.pt'),
        }
        if len(curves[name]['psnr']) != 40:
            raise ValueError('{} does not contain exactly 40 epochs'.format(run))
        for metric in ('psnr', 'ssim'):
            if len(source_curves[name][metric]) != 20:
                raise ValueError('{} source is not exactly 20 epochs'.format(source))
            if not np.array_equal(
                curves[name][metric][:20], source_curves[name][metric]
            ):
                maximum = np.max(np.abs(
                    curves[name][metric][:20] - source_curves[name][metric]
                ))
                raise ValueError(
                    '{} first-20 {} history changed; max error={}'.format(
                        name, metric, maximum
                    )
                )
        scheduler = torch.load(
            run / 'scheduler.pt', map_location='cpu', weights_only=True
        )
        if int(scheduler.get('last_epoch', -1)) != 40:
            raise ValueError('{} scheduler is not at epoch 40'.format(run))
        if int(scheduler.get('T_max', -1)) != 150:
            raise ValueError('{} scheduler T_max changed'.format(run))
        images[name] = per_image(run, 40)

    if images['c1'].keys() != images['srpr_c1'].keys():
        raise ValueError('epoch-40 per-image keys differ')
    keys = sorted(images['c1'])
    image_psnr_delta = np.array([
        images['srpr_c1'][key][0] - images['c1'][key][0] for key in keys
    ])
    image_ssim_delta = np.array([
        images['srpr_c1'][key][1] - images['c1'][key][1] for key in keys
    ])
    psnr_delta = curves['srpr_c1']['psnr'] - curves['c1']['psnr']
    ssim_delta = curves['srpr_c1']['ssim'] - curves['c1']['ssim']
    ci_low, ci_high = bootstrap_ci(image_psnr_delta)
    post_resume = psnr_delta[20:40]

    gates = {
        'final_delta_ge_0.010': bool(psnr_delta[39] >= 0.010),
        'last5_delta_ge_0.010': bool(psnr_delta[35:40].mean() >= 0.010),
        'bootstrap_ci_low_gt_0': bool(ci_low > 0),
        'final_ssim_nonnegative': bool(ssim_delta[39] >= 0),
    }
    decision = (
        'PROMOTE_TO_150E_CONFIRMATION'
        if all(gates.values()) else 'STOP_SRPR_ROUTE'
    )
    payload = {
        'comparison': 'SRPRv2+C1 minus B0+C1',
        'continuation': 'epoch 20 to 40',
        'first20_history_exact': True,
        'final': {
            'c1_psnr': float(curves['c1']['psnr'][39]),
            'srpr_c1_psnr': float(curves['srpr_c1']['psnr'][39]),
            'psnr_delta': float(psnr_delta[39]),
            'ssim_delta': float(ssim_delta[39]),
        },
        'last5_psnr_delta': float(psnr_delta[35:40].mean()),
        'post_resume_positive_epochs': int((post_resume > 0).sum()),
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

    print('N16 matched epoch-20 to epoch-40 continuation')
    print('comparison=SRPRv2+C1 minus B0+C1')
    print('first20_history_exact=True')
    for epoch in (20, 25, 30, 35, 40):
        print(
            'epoch_{:02d}: c1={:.6f}; srpr_c1={:.6f}; delta={:+.6f}'.format(
                epoch,
                curves['c1']['psnr'][epoch - 1],
                curves['srpr_c1']['psnr'][epoch - 1],
                psnr_delta[epoch - 1],
            )
        )
    print('final_delta={:+.6f}'.format(psnr_delta[39]))
    print('last5_delta={:+.6f}'.format(psnr_delta[35:40].mean()))
    print('post_resume_positive_epochs={}/20'.format((post_resume > 0).sum()))
    print('final_ssim_delta={:+.7f}'.format(ssim_delta[39]))
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

    output = args.json_output or args.run_dir / 'summary_40e.json'
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
