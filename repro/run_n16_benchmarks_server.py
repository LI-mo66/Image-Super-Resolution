#!/usr/bin/env python3
"""Evaluate fixed N16 epoch-40 C1/SRPR+C1 checkpoints on five benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
COUNTS = {'Set5': 5, 'Set14': 14, 'B100': 100,
          'Urban100': 100, 'Manga109': 109}
MODELS = {'c1': 'LFMN', 'srpr_c1': 'LFMNSRPRV2'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args):
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    status = source / 'wrapper_exit_status.txt'
    if not status.is_file() or status.read_text().strip() != '0':
        raise ValueError('epoch-40 source wrapper status is not zero')
    summary_path = source / 'summary_40e.json'
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('decision') != 'PROMOTE_TO_150E_CONFIRMATION':
        raise ValueError('epoch-40 source did not pass its registered gate')

    sys.path.insert(0, str(ROOT / 'LFMN'))
    from model.lfmn import Net as BaselineNet
    from model.lfmnsrprv2 import Net as SRPRv2Net
    classes = {'c1': BaselineNet, 'srpr_c1': SRPRv2Net}
    checkpoints = {}
    for name in MODELS:
        checkpoint = source / name / 'model/model_40.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        classes[name](scale=4).load_state_dict(state, strict=True)
        checkpoints[name] = {
            'path': str(checkpoint), 'sha256': sha256(checkpoint)
        }

    for dataset in DATASETS:
        root = args.data_root / 'benchmark' / dataset
        hr = sorted((root / 'HR').glob('*'))
        lr = sorted((root / 'LR_bicubic/X4').glob('*x4.png'))
        expected = COUNTS[dataset]
        if len(hr) != expected or len(lr) != expected:
            raise ValueError(
                '{}: expected {} HR/LR images, got {}/{}'.format(
                    dataset, expected, len(hr), len(lr)
                )
            )
    return source, output, checkpoints


def command(args, output, name, model, checkpoint):
    return [
        args.python_bin, 'main.py', '--dir_data', str(args.data_root),
        '--model', model, '--data_train', 'DIV2K',
        '--data_test', '+'.join(DATASETS), '--data_range', '1-800/801-900',
        '--scale', '4', '--n_threads', str(args.workers), '--ext', 'img',
        '--pre_train', checkpoint, '--test_only', '--save_per_image_metrics',
        '--experiment_root', str(output), '--save', name,
    ]


def run_logged(command_line, cwd, env, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command_line, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command_line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    source, output, checkpoints = validate(args)
    print(json.dumps({
        'preflight': 'passed', 'source': str(source),
        'output': str(output), 'checkpoints': checkpoints,
        'datasets': COUNTS, 'self_ensemble': False,
    }, indent=2), flush=True)
    if args.check_only:
        return
    output.mkdir(parents=True)
    (output / 'benchmark_manifest.json').write_text(json.dumps({
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
        ).strip(),
        'source_40': str(source), 'checkpoint_selection': 'epoch40 endpoint',
        'checkpoints': checkpoints, 'datasets': COUNTS,
        'scale': 4, 'self_ensemble': False,
    }, indent=2) + '\n', encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    for name, model in MODELS.items():
        group = output / name
        group.mkdir()
        run_logged(
            command(args, output, name, model, checkpoints[name]['path']),
            ROOT / 'LFMN', env, group / 'console.log'
        )
    summary_command = [
        args.python_bin, str(ROOT / 'repro/summarize_n16_benchmarks.py'),
        str(output),
    ]
    result = subprocess.run(
        summary_command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    (output / 'benchmark_summary.txt').write_text(
        result.stdout, encoding='utf-8'
    )
    print(result.stdout, end='', flush=True)


if __name__ == '__main__':
    main()
