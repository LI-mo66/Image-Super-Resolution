#!/usr/bin/env python3
"""Validate and summarize the preregistered N13 20-epoch screen."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


EXPECTED = {
    'epochs': '20', 'data_range': '1-800/801-900', 'scale': '[4]',
    'data_train': "['DIV2K']", 'data_test': "['DIV2K']", 'ext': 'img',
    'patch_size': '256', 'batch_size': '4', 'seed': '1', 'n_threads': '8',
    'rgb_range': '255', 'precision': 'single', 'no_augment': 'False',
    'self_ensemble': 'False', 'chop': 'False', 'test_every': '1000',
    'lr': '0.0002', 'scheduler': 'cosine', 'scheduler_t_max': '150',
    'eta_min': '1e-06', 'loss': '1*L1', 'pre_train': '',
    'optimizer': 'ADAM', 'betas': '(0.9, 0.999)', 'weight_decay': '0',
    'max_train_batches': '0', 'save_per_image_metrics': 'True',
}


def config(run):
    return dict(
        line.split(': ', 1)
        for line in (run / 'config.txt').read_text(encoding='utf-8').splitlines()
        if ': ' in line
    )


def validate(run, model):
    values = config(run)
    for key, expected in {**EXPECTED, 'model': model}.items():
        if values.get(key) != expected:
            raise ValueError(f'{run}: {key}={values.get(key)!r}, expected {expected!r}')


def curve(run, name):
    value = torch.load(run / name, map_location='cpu', weights_only=True).float()
    return value[:, 0, 0].numpy() if value.ndim == 3 else value.flatten().numpy()


def per_image(run, epoch=20):
    rows = torch.load(
        run / 'per_image_metrics' / f'epoch_{epoch:04d}.pt',
        map_location='cpu', weights_only=True,
    )
    return {
        (row['dataset'], int(row['scale']), row['filename']): row
        for row in rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--mechanism', type=Path, required=True)
    args = parser.parse_args()
    candidate, baseline = args.candidate.resolve(), args.baseline.resolve()
    validate(candidate, 'LFMNSRPRV3')
    validate(baseline, 'LFMN')
    cp, bp = curve(candidate, 'psnr_log.pt'), curve(baseline, 'psnr_log.pt')
    cs, bs = curve(candidate, 'ssim_log.pt'), curve(baseline, 'ssim_log.pt')
    if any(len(value) < 20 for value in (cp, bp, cs, bs)):
        raise ValueError('both runs must contain 20 epochs')
    ci, bi = per_image(candidate), per_image(baseline)
    if len(ci) != 100 or ci.keys() != bi.keys():
        raise ValueError('paired epoch-20 image sets must contain the same 100 images')
    deltas = np.array([
        float(ci[key]['psnr']) - float(bi[key]['psnr']) for key in sorted(ci)
    ])
    rng = np.random.default_rng(1)
    means = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(1)
    confidence = np.quantile(means, (0.025, 0.975))
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    mechanism = json.loads(args.mechanism.read_text(encoding='utf-8'))
    rows = {row['model']: row for row in profile['rows']}
    base_profile, candidate_profile = rows['LFMN'], rows['LFMNSRPRV3']
    result = {
        'candidate': str(candidate), 'baseline': str(baseline),
        'epoch20_psnr_delta': float(cp[19] - bp[19]),
        'last5_psnr_delta': float((cp[15:20] - bp[15:20]).mean()),
        'positive_epoch_ratio': float(((cp[:20] - bp[:20]) > 0).mean()),
        'epoch20_ssim_delta': float(cs[19] - bs[19]),
        'per_image_mean_delta': float(deltas.mean()),
        'per_image_median_delta': float(np.median(deltas)),
        'per_image_win_rate': float((deltas > 0).mean()),
        'per_image_bootstrap_95ci': [float(value) for value in confidence],
        'parameter_ratio': candidate_profile['parameters'] / base_profile['parameters'],
        'conv2d_mac_ratio': candidate_profile['conv2d_macs'] / base_profile['conv2d_macs'],
        'median_latency_ratio': candidate_profile['median_ms'] / base_profile['median_ms'],
        'feedback_on_off_mean_abs': mechanism['feedback_on_off_mean_abs'],
    }
    gates = {
        'last5_positive': bool(result['last5_psnr_delta'] > 0),
        'ci_lower_positive': bool(confidence[0] > 0),
        'win_rate_at_least_60pct': bool(result['per_image_win_rate'] >= 0.60),
        'ssim_nonnegative': bool(result['epoch20_ssim_delta'] >= 0),
        'parameters_not_above_baseline': bool(result['parameter_ratio'] <= 1.0),
        'conv2d_macs_not_above_baseline': bool(result['conv2d_mac_ratio'] <= 1.0),
        'median_latency_within_10pct': bool(result['median_latency_ratio'] <= 1.10),
        'feedback_active': bool(result['feedback_on_off_mean_abs'] > 0),
    }
    result['gates'] = gates
    result['decision'] = 'PASS_20_TO_40' if all(gates.values()) else 'NO_AUTOMATIC_EXTENSION'
    output = candidate / 'n13_20_vs_b0.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
