#!/usr/bin/env python3
"""Guarded paired continuation of N16 C1 and SRPR+C1 from epoch 20 to 40."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODELS = {'c1': 'LFMN', 'srpr_c1': 'LFMNSRPRV2'}
SOURCE_COMMON = {
    'data_range': '1-800/801-900',
    'scale': '[4]',
    'data_train': "['DIV2K']",
    'data_test': "['DIV2K']",
    'patch_size': '256',
    'batch_size': '4',
    'ext': 'img',
    'epochs': '20',
    'test_every': '1000',
    'max_train_batches': '0',
    'lr': '0.0002',
    'scheduler': 'cosine',
    'scheduler_t_max': '150',
    'eta_min': '1e-06',
    'optimizer': 'ADAM',
    'betas': '(0.9, 0.999)',
    'weight_decay': '0',
    'rgb_range': '255',
    'precision': 'single',
    'no_augment': 'False',
    'self_ensemble': 'False',
    'chop': 'False',
    'loss': '1*L1',
    'seed': '1',
    'pre_train': '',
    'save_per_image_metrics': 'True',
    'rgcrd_mode': 'output',
    'rgcrd_lambda_output': '0.1',
    'rgcrd_teacher_microbatch': '1',
    'rgcrd_grad_diag_every': '1',
}
REQUIRED = (
    'config.txt', 'model/model_20.pt', 'optimizer.pt', 'scheduler.pt',
    'loss.pt', 'loss_log.pt', 'psnr_log.pt', 'ssim_log.pt',
    'rgcrd_log.pt', 'per_image_metrics/epoch_0020.pt',
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            result[key] = value
    return result


def curve(run, name):
    value = torch.load(
        run / name, map_location='cpu', weights_only=True
    )
    if value.ndim != 3 or tuple(value.shape[1:]) != (1, 1):
        raise ValueError('{} has unexpected shape {}'.format(name, value.shape))
    return value[:, 0, 0].double().numpy()


def validate_source_group(run, name, model):
    for relative in REQUIRED:
        path = run / relative
        if not path.is_file():
            raise FileNotFoundError(path)
    config = read_config(run / 'config.txt')
    expected = {**SOURCE_COMMON, 'model': model, 'load': ''}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                '{}: {}={!r}; expected {!r}'.format(
                    run, key, config.get(key), value
                )
            )
    try:
        n_threads = int(config['n_threads'])
    except (KeyError, ValueError) as error:
        raise ValueError('{} has invalid n_threads'.format(run)) from error
    if n_threads < 0:
        raise ValueError('{} has negative n_threads'.format(run))
    psnr = curve(run, 'psnr_log.pt')
    ssim = curve(run, 'ssim_log.pt')
    if len(psnr) != 20 or len(ssim) != 20:
        raise ValueError('{} must contain exactly 20 epochs'.format(run))
    scheduler = torch.load(
        run / 'scheduler.pt', map_location='cpu', weights_only=True
    )
    if int(scheduler.get('last_epoch', -1)) != 20:
        raise ValueError('{} scheduler is not at epoch 20'.format(run))
    if int(scheduler.get('T_max', -1)) != 150:
        raise ValueError('{} scheduler T_max is not 150'.format(run))
    optimizer = torch.load(
        run / 'optimizer.pt', map_location='cpu', weights_only=True
    )
    if not optimizer.get('state') or len(optimizer.get('param_groups', [])) != 1:
        raise ValueError('{} optimizer state is incomplete'.format(run))
    rgcrd = torch.load(
        run / 'rgcrd_log.pt', map_location='cpu', weights_only=True
    )
    rows = rgcrd.get('rows', [])
    if len(rows) != 20 or int(rows[-1][0]) != 20:
        raise ValueError('{} RGCRD log is not complete through epoch 20'.format(run))
    image_rows = torch.load(
        run / 'per_image_metrics/epoch_0020.pt',
        map_location='cpu', weights_only=True,
    )
    if len(image_rows) != 100:
        raise ValueError('{} epoch-20 metrics are not 100 images'.format(run))
    return {
        'model_sha256': sha256(run / 'model/model_20.pt'),
        'optimizer_sha256': sha256(run / 'optimizer.pt'),
        'scheduler_sha256': sha256(run / 'scheduler.pt'),
        'epoch20_psnr': float(psnr[-1]),
        'epoch20_ssim': float(ssim[-1]),
        'n_threads': n_threads,
    }


def preflight(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('source and continuation output must be separate')
    if output.exists():
        raise FileExistsError('continuation output already exists: {}'.format(output))
    if not (args.data_root / 'DIV2K/DIV2K_train_HR/0001.png').is_file():
        raise FileNotFoundError(args.data_root)
    if not (args.teacher_repo / 'models/network_swinir.py').is_file():
        raise FileNotFoundError(args.teacher_repo)
    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    status_path = source / 'wrapper_exit_status.txt'
    if not status_path.is_file() or status_path.read_text().strip() != '0':
        raise ValueError('source 20-epoch wrapper status is not zero')
    summary_path = source / 'summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('decision') != 'PROMOTE_TO_MATCHED_40E_CONTINUATION':
        raise ValueError('source did not pass the registered 20-epoch gate')
    groups = {
        name: validate_source_group(source / name, name, model)
        for name, model in MODELS.items()
    }
    if groups['c1']['n_threads'] != groups['srpr_c1']['n_threads']:
        raise ValueError('C1 and SRPR+C1 used different DataLoader worker counts')
    observed_delta = groups['srpr_c1']['epoch20_psnr'] - groups['c1']['epoch20_psnr']
    recorded_delta = float(summary['final']['psnr_delta'])
    if not np.isclose(observed_delta, recorded_delta, rtol=0, atol=1e-7):
        raise ValueError('source summary delta does not match stored curves')
    return source, output, groups


def copy_source(source, output, groups):
    output.mkdir(parents=True)
    for name in MODELS:
        shutil.copytree(source / name, output / name)
        for artifact, expected_hash in (
            ('model/model_20.pt', groups[name]['model_sha256']),
            ('optimizer.pt', groups[name]['optimizer_sha256']),
            ('scheduler.pt', groups[name]['scheduler_sha256']),
        ):
            actual_hash = sha256(output / name / artifact)
            if actual_hash != expected_hash:
                raise ValueError('{} copy hash mismatch'.format(artifact))
    shutil.copy2(source / 'summary.txt', output / 'summary_epoch20.txt')
    shutil.copy2(source / 'summary.json', output / 'summary_epoch20.json')


def training_command(args, output, name, model, n_threads):
    return [
        args.python_bin, 'main.py',
        '--dir_data', str(args.data_root),
        '--model', model,
        '--data_train', 'DIV2K', '--data_test', 'DIV2K',
        '--data_range', '1-800/801-900', '--scale', '4',
        '--patch_size', '256', '--batch_size', '4',
        '--n_threads', str(n_threads),
        '--ext', 'img', '--epochs', '40', '--test_every', '1000',
        '--lr', '2e-4', '--scheduler', 'cosine',
        '--scheduler_t_max', '150', '--eta_min', '1e-6',
        '--loss', '1*L1', '--seed', '1', '--print_every', '100',
        '--save_per_image_metrics',
        '--rgcrd_mode', 'output',
        '--rgcrd_teacher_repo', str(args.teacher_repo),
        '--rgcrd_teacher_checkpoint', str(args.teacher_checkpoint),
        '--rgcrd_teacher_microbatch', str(args.teacher_microbatch),
        '--rgcrd_lambda_output', '0.1',
        '--rgcrd_grad_diag_every', '1',
        '--experiment_root', str(output),
        '--load', name, '--resume', '20', '--resume_data_epochs', '20',
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--teacher-repo', type=Path,
                        default=ROOT / 'repro/swinir_ref')
    parser.add_argument('--teacher-checkpoint', type=Path, default=(
        ROOT / 'repro/teacher_weights/'
        '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'
    ))
    parser.add_argument('--teacher-microbatch', type=int, default=1)
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.teacher_repo = args.teacher_repo.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()

    source, output, groups = preflight(args)
    print(json.dumps({
        'preflight': 'passed',
        'source': str(source),
        'output': str(output),
        'groups': groups,
    }, indent=2), flush=True)
    if args.check_only:
        return

    copy_source(source, output, groups)
    manifest = {
        'candidate': 'N16/SRPRv2+C1 interaction',
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
        'source_20': str(source),
        'output_40': str(output),
        'resume_from_epoch': 20,
        'target_epoch': 40,
        'scheduler_t_max': 150,
        'data_stream_replay_epochs': 20,
        'source_hashes': groups,
    }
    (output / 'resume_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpu,
        PYTHONUNBUFFERED='1',
    )
    for name, model in MODELS.items():
        command = training_command(
            args, output, name, model, groups[name]['n_threads']
        )
        print('\nContinuing {}: {}'.format(name, ' '.join(command)), flush=True)
        subprocess.run(command, cwd=ROOT / 'LFMN', env=env, check=True)

    summary_command = [
        args.python_bin,
        str(ROOT / 'repro/summarize_n16_srpr_c1_40e.py'),
        str(output), '--source-20', str(source),
    ]
    result = subprocess.run(
        summary_command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    (output / 'summary_40e.txt').write_text(result.stdout, encoding='utf-8')
    print(result.stdout, end='', flush=True)


if __name__ == '__main__':
    main()
