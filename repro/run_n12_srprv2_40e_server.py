#!/usr/bin/env python3
"""Guarded N12 epoch-20 to epoch-40 continuation against an existing B0."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_B0_20 = Path(
    '/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/'
    'scratch/n9_pcstr_20e_seed1/baseline'
)
EXPECTED = {
    'pre_train': '', 'data_range': '1-800/801-900', 'scale': '[4]',
    'data_train': "['DIV2K']", 'data_test': "['DIV2K']", 'ext': 'img',
    'patch_size': '256', 'batch_size': '4', 'seed': '1', 'n_threads': '8',
    'rgb_range': '255', 'precision': 'single', 'no_augment': 'False',
    'self_ensemble': 'False', 'chop': 'False', 'test_every': '1000',
    'max_train_batches': '0', 'save_per_image_metrics': 'True',
    'lr': '0.0002', 'scheduler': 'cosine', 'scheduler_t_max': '150',
    'eta_min': '1e-06', 'loss': '1*L1', 'optimizer': 'ADAM',
    'betas': '(0.9, 0.999)', 'weight_decay': '0',
}


def config(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            result[key] = value
    return result


def curve(run, name):
    value = torch.load(run / name, map_location='cpu', weights_only=True).float()
    return value[:, 0, 0].numpy() if value.ndim == 3 else value.flatten().numpy()


def validate_config(run, model):
    cfg = config(run / 'config.txt')
    for key, expected in {**EXPECTED, 'model': model}.items():
        if cfg.get(key) != expected:
            raise ValueError(f'{run}: {key} expected {expected!r}, got {cfg.get(key)!r}')
    return cfg


def require(run, names):
    for name in names:
        path = run / name
        if not path.is_file():
            raise FileNotFoundError(path)


def find_runs(roots, model_epoch):
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob(f'model/model_{model_epoch}.pt'):
            yield path.parent.parent.resolve()


def select_n12(args):
    if args.n12_run:
        candidates = [args.n12_run.resolve()]
    else:
        candidates = list(find_runs([ROOT / 'experiment/all_runs/n12'], 20))
        candidates = [run for run in candidates if config(run / 'config.txt').get('model') == 'LFMNSRPRV2']
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f'Expected one N12 run; pass --n12-run explicitly. Found: {candidates}')
    run = candidates[0]
    validate_config(run, 'LFMNSRPRV2')
    require(run, ('model/model_20.pt', 'optimizer.pt', 'scheduler.pt',
                  'psnr_log.pt', 'ssim_log.pt',
                  'per_image_metrics/epoch_0020.pt', 'mechanism_diagnostics.pt'))
    if len(curve(run, 'psnr_log.pt')) != 20:
        raise ValueError(f'N12 must contain exactly 20 epochs: {run}')
    scheduler = torch.load(run / 'scheduler.pt', map_location='cpu', weights_only=True)
    if int(scheduler.get('last_epoch', -1)) != 20:
        raise ValueError(f'N12 scheduler must be at epoch 20: {run}')
    diagnostics = torch.load(run / 'mechanism_diagnostics.pt', map_location='cpu', weights_only=True)
    columns, rows = diagnostics.get('columns', ()), diagnostics.get('rows', ())
    if not rows:
        raise ValueError('N12 mechanism diagnostics are empty')
    for key in ('state_delta_l2', 'feature_delta_l2', 'q_delta_l2'):
        index = columns.index(key)
        values = [float(row[index]) for row in rows if int(row[0]) == 20]
        if not values or not np.isfinite(values).all() or max(values) <= 0:
            raise ValueError(f'N12 epoch-20 {key} is absent, nonfinite or inactive')
    profile_path = run.parent / 'profile_metrics.json'
    require(run.parent, ('profile_metrics.json',))
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    print('N12 profile:', json.dumps(profile, ensure_ascii=False), flush=True)
    return run


def select_b0(args):
    b0_20 = args.b0_20.resolve()
    validate_config(b0_20, 'LFMN')
    require(b0_20, ('model/model_20.pt', 'psnr_log.pt', 'ssim_log.pt'))
    psnr20, ssim20 = curve(b0_20, 'psnr_log.pt')[:20], curve(b0_20, 'ssim_log.pt')[:20]
    if len(psnr20) != 20 or len(ssim20) != 20:
        raise ValueError('B0-20 reference has fewer than 20 epochs')
    if args.b0_40:
        candidates = [args.b0_40.resolve()]
    else:
        candidates = list(find_runs(args.search_root, 40))
    matches = []
    inspected = []
    for run in sorted(set(candidates)):
        try:
            validate_config(run, 'LFMN')
            require(run, ('model/model_40.pt', 'psnr_log.pt', 'ssim_log.pt',
                          'per_image_metrics/epoch_0040.pt'))
            psnr, ssim = curve(run, 'psnr_log.pt'), curve(run, 'ssim_log.pt')
            if len(psnr) < 40 or len(ssim) < 40:
                inspected.append((run, 'curve shorter than 40 epochs'))
            elif np.allclose(psnr[:20], psnr20, rtol=0, atol=1e-6) and \
                    np.allclose(ssim[:20], ssim20, rtol=0, atol=1e-6):
                matches.append(run)
                inspected.append((run, 'MATCH'))
            else:
                psnr_gap = float(np.max(np.abs(psnr[:20] - psnr20)))
                ssim_gap = float(np.max(np.abs(ssim[:20] - ssim20)))
                inspected.append((run, f'first-20 curve differs: max PSNR {psnr_gap:.6f}, '
                                       f'max SSIM {ssim_gap:.6f}'))
        except (FileNotFoundError, ValueError, KeyError) as error:
            inspected.append((run, f'{type(error).__name__}: {error}'))
            if args.b0_40 and not args.inspect_b0:
                raise
    if args.inspect_b0:
        if not inspected:
            print('No model_40.pt found in the search roots.', flush=True)
        for run, reason in inspected:
            print(f'{run}: {reason}', flush=True)
        return None
    if len(matches) != 1:
        raise ValueError(
            f'Expected one B0-40 with the exact N9 B0 first-20 curve; found {matches}. '
            'Run --inspect-b0 (optionally with --search-root /root/autodl-tmp) '
            'to see candidate paths and rejection reasons.'
        )
    return matches[0]


def summarize(n12, b0):
    cp, cs = curve(n12, 'psnr_log.pt'), curve(n12, 'ssim_log.pt')
    bp, bs = curve(b0, 'psnr_log.pt'), curve(b0, 'ssim_log.pt')
    if min(map(len, (cp, cs, bp, bs))) < 40:
        raise ValueError('N12 and B0 must both have 40 epochs')
    candidate = torch.load(n12 / 'per_image_metrics/epoch_0040.pt',
                           map_location='cpu', weights_only=True)
    baseline = torch.load(b0 / 'per_image_metrics/epoch_0040.pt',
                          map_location='cpu', weights_only=True)
    def keyed(rows):
        return {(row['dataset'], int(row['scale']), row['filename']): row for row in rows}
    cimg, bimg = keyed(candidate), keyed(baseline)
    if not cimg or cimg.keys() != bimg.keys():
        raise ValueError('N12/B0 epoch-40 per-image sets differ')
    deltas = np.array([cimg[key]['psnr'] - bimg[key]['psnr'] for key in sorted(cimg)])
    rng = np.random.default_rng(1)
    means = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    result = {
        'n12_run': str(n12), 'b0_run': str(b0),
        'epoch40_psnr_delta': float(cp[39] - bp[39]),
        'last5_psnr_delta': float((cp[35:40] - bp[35:40]).mean()),
        'epoch40_ssim_delta': float(cs[39] - bs[39]),
        'per_image_mean_delta': float(deltas.mean()),
        'per_image_win_rate': float((deltas > 0).mean()),
        'per_image_bootstrap_95ci': [float(x) for x in np.quantile(means, [.025, .975])],
    }
    output = n12 / 'n12_40_vs_b0.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n12-run', type=Path, help='N12 srprv2 output directory; auto-find if omitted')
    parser.add_argument('--b0-20', type=Path, default=DEFAULT_B0_20)
    parser.add_argument('--b0-40', type=Path, help='B0 40-epoch directory; auto-find if omitted')
    parser.add_argument('--search-root', type=Path, action='append', default=None)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--check-only', action='store_true',
                        help='validate and print both runs without starting training')
    parser.add_argument('--inspect-b0', action='store_true',
                        help='list B0-40 candidates and why each does or does not match')
    args = parser.parse_args()
    if args.search_root is None:
        args.search_root = [
            Path('/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs'),
            ROOT / 'experiment/all_runs',
        ]
    n12 = select_n12(args)
    b0 = select_b0(args)
    if args.inspect_b0:
        return
    if not (args.data_root / 'DIV2K/DIV2K_train_HR/0001.png').is_file():
        raise FileNotFoundError(args.data_root)
    print(f'Validated N12-20: {n12}\nValidated existing B0-40: {b0}', flush=True)
    if args.check_only:
        return
    command = [
        args.python_bin, 'main.py', '--dir_data', str(args.data_root),
        '--model', 'LFMNSRPRV2', '--data_train', 'DIV2K', '--data_test', 'DIV2K',
        '--data_range', '1-800/801-900', '--scale', '4', '--patch_size', '256',
        '--batch_size', '4', '--n_threads', '8', '--ext', 'img', '--epochs', '40',
        '--test_every', '1000', '--lr', '2e-4', '--scheduler', 'cosine',
        '--scheduler_t_max', '150', '--eta_min', '1e-6', '--loss', '1*L1',
        '--seed', '1', '--print_every', '100', '--save_per_image_metrics',
        '--experiment_root', str(n12.parent), '--load', n12.name,
        '--resume', '20', '--n12_resume_data_epochs', '20',
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    manifest = {
        'candidate': 'N12/SRPRv2', 'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'resume_from_epoch': 20, 'target_epoch': 40,
        'n12_run': str(n12), 'b0_20_reference': str(args.b0_20.resolve()),
        'b0_40_reference': str(b0), 'scheduler_t_max': 150,
        'data_stream_replay_epochs': 20,
    }
    (n12 / 'n12_40_resume_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT / 'LFMN', env=env, check=True)
    summarize(n12, b0)


if __name__ == '__main__':
    main()
