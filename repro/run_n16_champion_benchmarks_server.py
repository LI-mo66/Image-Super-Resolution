#!/usr/bin/env python3
"""Evaluate the fixed epoch-1000 N16 champion on five SR benchmarks."""
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def locate_benchmark_root(data_root):
    candidates = (data_root / 'benchmark', data_root)
    for candidate in candidates:
        if all((candidate / dataset / 'HR').is_dir() for dataset in DATASETS):
            return candidate
    raise FileNotFoundError(
        'benchmark datasets were not found below {} or {}'.format(*candidates)
    )


def validate(args):
    source = args.source.resolve()
    output = args.output.resolve()
    summary_path = source / 'summary_1000e.json'
    checkpoint = source / 'model/model_1000.pt'
    if not summary_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError('epoch-1000 source is incomplete: {}'.format(source))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('decision') != 'READY_FOR_FIXED_EPOCH1000_BENCHMARKS':
        raise ValueError('epoch-1000 integrity summary did not pass')
    expected_hash = summary.get('checkpoint_sha256')
    actual_hash = sha256(checkpoint)
    if expected_hash != actual_hash:
        raise ValueError('epoch-1000 checkpoint hash differs from its summary')
    sys.path.insert(0, str(ROOT / 'LFMN'))
    from model.lfmnsrprv2 import Net as SRPRv2Net
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    SRPRv2Net(scale=4).load_state_dict(state, strict=True)

    benchmark_root = locate_benchmark_root(args.data_root.resolve())
    for dataset in DATASETS:
        root = benchmark_root / dataset
        hr = sorted((root / 'HR').glob('*'))
        lr = sorted((root / 'LR_bicubic/X4').glob('*x4.png'))
        expected = COUNTS[dataset]
        if len(hr) != expected or len(lr) != expected:
            raise ValueError(
                '{}: expected {} HR/LR images, got {}/{}'.format(
                    dataset, expected, len(hr), len(lr)
                )
            )
    if output.exists():
        completed = (
            (output / 'benchmark_summary.json').is_file()
            and (output / 'champion/per_image_metrics/epoch_0000.pt').is_file()
        )
        if not completed:
            raise FileExistsError('incomplete benchmark output exists: {}'.format(output))
    else:
        completed = False
    return source, output, checkpoint, actual_hash, benchmark_root, completed


def run_logged(command, cwd, env, log_path):
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def compatible_data_root(output, requested_root, benchmark_root):
    if benchmark_root == requested_root / 'benchmark':
        return requested_root
    layout = output / '_dataset_layout' / 'benchmark'
    layout.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        link = layout / dataset
        target = benchmark_root / dataset
        if link.exists() or link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError('benchmark compatibility link points elsewhere: {}'.format(link))
        else:
            link.symlink_to(target, target_is_directory=True)
    return layout.parent


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
    source, output, checkpoint, checkpoint_hash, benchmark_root, completed = (
        validate(args)
    )
    print(json.dumps({
        'preflight': 'passed', 'source': str(source), 'output': str(output),
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_selection': 'fixed epoch 1000 endpoint',
        'datasets': COUNTS, 'benchmark_root': str(benchmark_root),
        'self_ensemble': False, 'already_complete': completed,
    }, indent=2), flush=True)
    if args.check_only:
        return
    if not completed:
        output.mkdir(parents=True)
        group = output / 'champion'
        group.mkdir()
        command_data_root = compatible_data_root(
            output, args.data_root.resolve(), benchmark_root
        )
        (output / 'benchmark_manifest.json').write_text(json.dumps({
            'source_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
            ).strip(),
            'source_1000': str(source),
            'checkpoint_selection': 'fixed preregistered epoch1000 endpoint',
            'checkpoint': {'path': str(checkpoint), 'sha256': checkpoint_hash},
            'datasets': COUNTS, 'scale': 4, 'self_ensemble': False,
        }, indent=2) + '\n', encoding='utf-8')
        command = [
            args.python_bin, 'main.py', '--dir_data', str(command_data_root),
            '--model', 'LFMNSRPRV2', '--data_train', 'DIV2K',
            '--data_test', '+'.join(DATASETS), '--data_range', '1-800/801-900',
            '--scale', '4', '--n_threads', str(args.workers), '--ext', 'img',
            '--pre_train', str(checkpoint), '--test_only',
            '--save_per_image_metrics', '--experiment_root', str(output),
            '--save', 'champion',
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
                   PYTHONUNBUFFERED='1')
        run_logged(command, ROOT / 'LFMN', env, group / 'console.log')

    result = subprocess.run([
        args.python_bin,
        str(ROOT / 'repro/summarize_n16_champion_benchmarks.py'),
        str(output),
    ], cwd=ROOT, check=True, text=True, capture_output=True)
    (output / 'benchmark_summary.txt').write_text(result.stdout, encoding='utf-8')
    print(result.stdout, end='', flush=True)


if __name__ == '__main__':
    main()
