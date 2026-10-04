#!/usr/bin/env python3
"""Start or exactly resume the frozen N16 champion 1000-epoch run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    'model': 'LFMNSRPRV2',
    'data_range': '1-800/801-900',
    'scale': '[4]',
    'patch_size': '256',
    'batch_size': '4',
    'n_threads': '8',
    'ext': 'img',
    'epochs': '1000',
    'test_every': '1000',
    'max_train_batches': '0',
    'lr': '0.0002',
    'scheduler': 'multistep',
    'decay': '200-400-600-800',
    'gamma': '0.5',
    'optimizer': 'ADAM',
    'weight_decay': '0',
    'loss': '1*L1',
    'seed': '1',
    'pre_train': '',
    'save_per_image_metrics': 'True',
    'rgcrd_mode': 'output',
    'rgcrd_lambda_output': '0.1',
    'rgcrd_teacher_microbatch': '1',
    'rgcrd_grad_diag_every': '10',
}


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


def curve_length(path):
    curve = torch.load(path, map_location='cpu', weights_only=True)
    if curve.ndim != 3 or tuple(curve.shape[1:]) != (1, 1):
        raise ValueError('{} has unexpected shape {}'.format(path, curve.shape))
    return len(curve)


def validate_resume(output, current_commit):
    manifest_path = output / 'long_run_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('source_commit') != current_commit:
        raise ValueError(
            'resume code commit differs: {} != {}'.format(
                current_commit, manifest.get('source_commit')
            )
        )
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('resume manifest protocol differs')
    config_path = output / 'config.txt'
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_config(config_path)
    for key, value in PROTOCOL.items():
        if config.get(key) != value:
            raise ValueError(
                '{}={!r}; expected {!r}'.format(key, config.get(key), value)
            )
    psnr_path = output / 'psnr_log.pt'
    ssim_path = output / 'ssim_log.pt'
    if not psnr_path.exists() and not ssim_path.exists():
        if any((output / 'model').glob('model_*.pt')):
            raise ValueError('checkpoints exist without metric history')
        return 0
    if not psnr_path.is_file() or not ssim_path.is_file():
        raise ValueError('only one metric history exists')
    psnr_epoch = curve_length(psnr_path)
    ssim_epoch = curve_length(ssim_path)
    if psnr_epoch != ssim_epoch or not 0 < psnr_epoch <= 1000:
        raise ValueError('invalid completed epoch count')
    if psnr_epoch == 1000:
        return psnr_epoch
    required = (
        output / 'model/model_{}.pt'.format(psnr_epoch),
        output / 'optimizer.pt', output / 'scheduler.pt',
        output / 'loss.pt', output / 'loss_log.pt',
        output / 'rgcrd_log.pt',
        output / 'per_image_metrics/epoch_{:04d}.pt'.format(psnr_epoch),
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    scheduler = torch.load(
        output / 'scheduler.pt', map_location='cpu', weights_only=True
    )
    if int(scheduler.get('last_epoch', -1)) != psnr_epoch:
        raise ValueError('scheduler epoch does not match metric history')
    if float(scheduler.get('gamma', -1)) != 0.5:
        raise ValueError('scheduler gamma is not 0.5')
    milestones = scheduler.get('milestones')
    if set(milestones) != {200, 400, 600, 800}:
        raise ValueError('scheduler milestones are not 200/400/600/800')
    optimizer = torch.load(
        output / 'optimizer.pt', map_location='cpu', weights_only=True
    )
    if not optimizer.get('state'):
        raise ValueError('optimizer state is empty')
    rgcrd = torch.load(
        output / 'rgcrd_log.pt', map_location='cpu', weights_only=True
    )
    if len(rgcrd.get('rows', [])) != psnr_epoch:
        raise ValueError('RGCRD log length does not match metric history')
    sys.path.insert(0, str(ROOT / 'LFMN'))
    from model.lfmnsrprv2 import Net as SRPRv2Net
    state = torch.load(
        output / 'model/model_{}.pt'.format(psnr_epoch),
        map_location='cpu', weights_only=True,
    )
    SRPRv2Net(scale=4).load_state_dict(state, strict=True)
    return psnr_epoch


def preflight(args):
    run_path = Path(args.run_name)
    if run_path.is_absolute() or '..' in run_path.parts:
        raise ValueError('run-name must be a safe relative path')
    data_root = args.data_root.resolve()
    teacher_repo = args.teacher_repo.resolve()
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    experiment_root = args.experiment_root.resolve()
    output = experiment_root / run_path
    if not (data_root / 'DIV2K/DIV2K_train_HR/0001.png').is_file():
        raise FileNotFoundError(data_root)
    if not (teacher_repo / 'models/network_swinir.py').is_file():
        raise FileNotFoundError(teacher_repo)
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(teacher_checkpoint)
    current_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
    ).strip()
    completed = (
        validate_resume(output, current_commit) if output.exists() else 0
    )
    return {
        'output': output,
        'data_root': data_root,
        'teacher_repo': teacher_repo,
        'teacher_checkpoint': teacher_checkpoint,
        'teacher_sha256': sha256(teacher_checkpoint),
        'source_commit': current_commit,
        'completed_epochs': completed,
    }


def training_command(args, checked):
    command = [
        args.python_bin, 'main.py',
        '--dir_data', str(checked['data_root']),
        '--model', 'LFMNSRPRV2',
        '--data_train', 'DIV2K', '--data_test', 'DIV2K',
        '--data_range', '1-800/801-900', '--scale', '4',
        '--patch_size', '256', '--batch_size', '4', '--n_threads', '8',
        '--ext', 'img', '--epochs', '1000', '--test_every', '1000',
        '--lr', '2e-4', '--scheduler', 'multistep',
        '--decay', '200-400-600-800', '--gamma', '0.5',
        '--loss', '1*L1', '--seed', '1', '--print_every', '100',
        '--save_per_image_metrics', '--rgcrd_mode', 'output',
        '--rgcrd_teacher_repo', str(checked['teacher_repo']),
        '--rgcrd_teacher_checkpoint', str(checked['teacher_checkpoint']),
        '--rgcrd_teacher_microbatch', '1', '--rgcrd_lambda_output', '0.1',
        '--rgcrd_grad_diag_every', '10',
        '--experiment_root', str(args.experiment_root.resolve()),
    ]
    completed = checked['completed_epochs']
    if completed:
        command.extend([
            '--load', args.run_name, '--resume', str(completed),
            '--resume_data_epochs', str(completed),
        ])
    else:
        command.extend(['--save', args.run_name])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--experiment-root', type=Path,
                        default=ROOT / 'experiment/all_runs')
    parser.add_argument('--teacher-repo', type=Path,
                        default=ROOT / 'repro/swinir_ref')
    parser.add_argument('--teacher-checkpoint', type=Path, default=(
        ROOT / 'repro/teacher_weights/'
        '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth'
    ))
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    checked = preflight(args)
    printable = {key: str(value) if isinstance(value, Path) else value
                 for key, value in checked.items()}
    print(json.dumps({'preflight': 'passed', **printable}, indent=2), flush=True)
    if args.check_only:
        return
    output = checked['output']
    if checked['completed_epochs'] == 1000:
        print('Training already complete at epoch 1000.', flush=True)
    else:
        if not output.exists():
            output.mkdir(parents=True)
            manifest = {
                'candidate': 'N16 frozen champion: SRPRv2+C1',
                'source_commit': checked['source_commit'],
                'teacher_checkpoint': str(checked['teacher_checkpoint']),
                'teacher_sha256': checked['teacher_sha256'],
                'protocol': PROTOCOL,
                'checkpoint_selection': 'fixed epoch 1000 endpoint',
            }
            (output / 'long_run_manifest.json').write_text(
                json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
            )
        command = training_command(args, checked)
        print('Training command: {}'.format(' '.join(command)), flush=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                   PYTHONUNBUFFERED='1')
        subprocess.run(command, cwd=ROOT / 'LFMN', env=env, check=True)

    result = subprocess.run([
        args.python_bin,
        str(ROOT / 'repro/summarize_n16_champion_1000e.py'),
        str(output),
    ], cwd=ROOT, check=True, text=True, capture_output=True)
    (output / 'summary_1000e.txt').write_text(result.stdout, encoding='utf-8')
    print(result.stdout, end='', flush=True)


if __name__ == '__main__':
    main()
