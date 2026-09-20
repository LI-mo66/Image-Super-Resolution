#!/usr/bin/env python3
"""Validate and summarize a same-protocol B0/C1/M0/M1 RGCRD screen."""
import argparse
from pathlib import Path

import numpy as np
import torch


RUNS = {
    'b0': ('LFMN', 'off'),
    'c1': ('LFMN', 'output'),
    'm0': ('LFMNRGCRD', 'relation'),
    'm1': ('LFMNRGCRD', 'full'),
}
COMMON = {
    'data_range': '1-800/801-900', 'scale': '[4]',
    'ext': 'img',
    'lr': '0.0002', 'scheduler': 'multistep',
    'decay': '200-400-600-800', 'loss': '1*L1', 'seed': '1',
    'pre_train': '', 'save_per_image_metrics': 'True',
}


def read_config(run_dir):
    path = run_dir / 'config.txt'
    return dict(
        line.split(': ', 1)
        for line in path.read_text(encoding='utf-8').splitlines()
        if ': ' in line
    )


def validate(run_dir, model, mode, epochs=None, protocol=None):
    config = read_config(run_dir)
    expected = {**COMMON, 'model': model, 'rgcrd_mode': mode}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                '{}: {}={!r}; expected {!r}'.format(
                    run_dir, key, config.get(key), value
                )
            )
    configured_epochs = int(config['epochs'])
    if epochs is not None and configured_epochs != epochs:
        raise ValueError('epoch protocol differs across runs')
    current_protocol = {
        key: config[key]
        for key in ('patch_size', 'batch_size', 'n_threads')
    }
    if protocol is not None and current_protocol != protocol:
        raise ValueError('batch/patch/worker protocol differs across runs')
    return configured_epochs, current_protocol


def curve(run_dir, name):
    values = torch.load(run_dir / name, map_location='cpu', weights_only=True)
    if values.ndim != 3 or values.shape[1:] != (1, 1):
        raise ValueError('{} has unexpected shape {}'.format(name, values.shape))
    return values[:, 0, 0].double().numpy()


def per_image(run_dir, epoch):
    path = run_dir / 'per_image_metrics' / 'epoch_{:04d}.pt'.format(epoch)
    rows = torch.load(path, map_location='cpu', weights_only=True)
    result = {
        (row['dataset'], int(row['scale']), row['filename']):
        (float(row['psnr']), float(row['ssim']))
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
    args = parser.parse_args()
    epochs = None
    protocol = None
    curves = {}
    image_metrics = {}
    for name, (model, mode) in RUNS.items():
        run_dir = args.screen_dir / name
        epochs, protocol = validate(
            run_dir, model, mode, epochs, protocol
        )
        curves[name] = {
            'psnr': curve(run_dir, 'psnr_log.pt'),
            'ssim': curve(run_dir, 'ssim_log.pt'),
        }
        if len(curves[name]['psnr']) != epochs:
            raise ValueError('{} is incomplete'.format(run_dir))
        image_metrics[name] = per_image(run_dir, epochs)

    checkpoints = sorted(set((1, min(10, epochs), min(20, epochs), epochs)))
    baseline = curves['b0']['psnr']
    print('| group | {} | last5 PSNR delta | final SSIM delta |'.format(
        ' | '.join('e{} PSNR delta'.format(item) for item in checkpoints)
    ))
    print('| --- | {} | ---: | ---: |'.format(
        ' | '.join('---:' for _ in checkpoints)
    ))
    for name in RUNS:
        delta = curves[name]['psnr'] - baseline
        values = [delta[item - 1] for item in checkpoints]
        print('| {} | {} | {:+.6f} | {:+.6f} |'.format(
            name, ' | '.join('{:+.6f}'.format(item) for item in values),
            delta[-min(5, epochs):].mean(),
            curves[name]['ssim'][-1] - curves['b0']['ssim'][-1],
        ))

    baseline_rows = image_metrics['b0']
    print('\n| group | paired mean delta | median | win rate | 95% bootstrap CI |')
    print('| --- | ---: | ---: | ---: | --- |')
    paired_means = {}
    for name in ('c1', 'm0', 'm1'):
        rows = image_metrics[name]
        if rows.keys() != baseline_rows.keys():
            raise ValueError('{} per-image keys differ from B0'.format(name))
        delta = np.array([
            rows[key][0] - baseline_rows[key][0]
            for key in sorted(rows)
        ])
        low, high = confidence_interval(delta)
        paired_means[name] = delta.mean()
        print('| {} | {:+.6f} | {:+.6f} | {:.1%} | [{:+.6f}, {:+.6f}] |'.format(
            name, delta.mean(), np.median(delta), (delta > 0).mean(), low, high
        ))

    m1_final = curves['m1']['psnr'][-1]
    ordering = {
        name: m1_final > curves[name]['psnr'][-1]
        for name in ('b0', 'c1', 'm0')
    }
    print('\nM1 final-epoch ordering gates: {}'.format(', '.join(
        'M1>{}={}'.format(name.upper(), passed)
        for name, passed in ordering.items()
    )))
    if all(ordering.values()) and paired_means['m1'] > 0:
        print('SCREEN SIGNAL: eligible for longer validation; not yet a paper claim.')
    else:
        print('SCREEN SIGNAL: do not promote the full RGCRD configuration yet.')


if __name__ == '__main__':
    main()
