#!/usr/bin/env python3
"""Guarded server launcher for the N13/SRPRv3 20-epoch screen."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = Path(
    '/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/'
    'scratch/n9_pcstr_20e_seed1/baseline'
)
sys.path.insert(0, str(ROOT / 'repro'))
from summarize_n13_srprv3 import curve, per_image, validate  # noqa: E402


def run(command, *, cwd=ROOT, env=None):
    print('Running:', ' '.join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_name', nargs='?', default='n13/srprv3_20e_seed1')
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    validate(baseline, 'LFMN')
    if len(curve(baseline, 'psnr_log.pt')) != 20:
        raise ValueError('registered B0 must contain exactly 20 PSNR epochs')
    if len(per_image(baseline, 20)) != 100:
        raise ValueError('registered B0 must contain 100 epoch-20 image rows')
    if not (args.data_root / 'DIV2K/DIV2K_train_HR/0001.png').is_file():
        raise FileNotFoundError(args.data_root)
    group = (ROOT / 'experiment/all_runs' / args.run_name).resolve()
    candidate = group / 'srprv3'
    print(f'Validated baseline: {baseline}', flush=True)
    print(f'N13 output: {candidate}', flush=True)
    run([args.python_bin, 'repro/check_n13_srprv3.py'])
    run([
        args.python_bin, 'repro/check_n13_real_batch.py',
        '--data-root', args.data_root,
    ])
    if args.check_only:
        return
    if group.exists():
        raise FileExistsError(f'output already exists: {group}')
    group.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    profile_path = group / 'profile_metrics.json'
    run([
        args.python_bin, 'repro/profile_n13_srprv3.py',
        '--output', profile_path,
    ], env=env)
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
    ).strip()
    manifest = {
        'candidate': 'N13/SRPRv3', 'branch': 'codex/n13-srprv3',
        'commit': commit, 'baseline': str(baseline),
        'data_root': str(args.data_root.resolve()), 'run': str(candidate),
        'protocol': {
            'data_range': '1-800/801-900', 'scale': 4, 'patch_size': 256,
            'batch_size': 4, 'seed': 1, 'optimizer': 'Adam', 'lr': 2e-4,
            'scheduler': 'cosine', 'scheduler_t_max': 150,
            'eta_min': 1e-6, 'loss': '1*L1', 'epochs': 20,
        },
    }
    (group / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    run([
        args.python_bin, 'main.py', '--dir_data', args.data_root,
        '--model', 'LFMNSRPRV3', '--data_train', 'DIV2K',
        '--data_test', 'DIV2K', '--data_range', '1-800/801-900',
        '--scale', '4', '--patch_size', '256', '--batch_size', '4',
        '--n_threads', '8', '--ext', 'img', '--epochs', '20',
        '--test_every', '1000', '--lr', '2e-4', '--scheduler', 'cosine',
        '--scheduler_t_max', '150', '--eta_min', '1e-6', '--loss', '1*L1',
        '--seed', '1', '--print_every', '100', '--save_per_image_metrics',
        '--experiment_root', ROOT / 'experiment/all_runs',
        '--save', f'{args.run_name}/srprv3',
    ], cwd=ROOT / 'LFMN', env=env)
    mechanism_path = candidate / 'mechanism_epoch_0020.json'
    run([
        args.python_bin, 'repro/diagnose_n13_srprv3.py', candidate,
        '--data-root', args.data_root, '--epoch', '20',
    ], env=env)
    run([
        args.python_bin, 'repro/summarize_n13_srprv3.py', candidate,
        '--baseline', baseline, '--profile', profile_path,
        '--mechanism', mechanism_path,
    ])
    print(f'N13 screen completed: {candidate}', flush=True)


if __name__ == '__main__':
    main()
