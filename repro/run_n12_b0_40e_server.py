#!/usr/bin/env python3
"""Continue the matching N9 B0 from epoch 20 to 40 in a separate output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import torch

from run_n12_srprv2_40e_server import (
    DEFAULT_B0_20, ROOT, curve, require, select_b0, validate_config,
)


DEFAULT_OUTPUT = ROOT / 'experiment/all_runs/n12/b0_40e_seed1_from_n9'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(args):
    source = args.b0_20.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('B0 output must be separate from the source directory')
    if output.exists():
        raise FileExistsError(f'B0 output already exists: {output}')
    cfg = validate_config(source, 'LFMN')
    if cfg.get('epochs') != '20':
        raise ValueError(f'N9 B0 must be the 20-epoch run: {cfg.get("epochs")!r}')
    require(source, ('model/model_20.pt', 'optimizer.pt', 'scheduler.pt',
                     'loss.pt', 'loss_log.pt', 'psnr_log.pt', 'ssim_log.pt',
                     'per_image_metrics/epoch_0020.pt'))
    if len(curve(source, 'psnr_log.pt')) != 20 or len(curve(source, 'ssim_log.pt')) != 20:
        raise ValueError('N9 B0 curves must contain exactly 20 epochs')
    scheduler = torch.load(source / 'scheduler.pt', map_location='cpu', weights_only=True)
    if scheduler.get('last_epoch') != 20 or scheduler.get('T_max') != 150:
        raise ValueError(f'N9 B0 scheduler is not at epoch 20 with T_max=150: {scheduler}')
    if not (args.data_root / 'DIV2K/DIV2K_train_HR/0001.png').is_file():
        raise FileNotFoundError(args.data_root)
    source_hash = sha256(source / 'model/model_20.pt')
    print(f'Validated N9 B0-20: {source}', flush=True)
    print(f'B0-20 model SHA256: {source_hash}', flush=True)
    print(f'Separate B0-40 output: {output}', flush=True)
    return source, output, source_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b0-20', type=Path, default=DEFAULT_B0_20)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--python-bin', default=sys.executable)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    source, output, source_hash = preflight(args)
    if args.check_only:
        return

    shutil.copytree(source, output)
    if sha256(output / 'model/model_20.pt') != source_hash:
        raise ValueError('Copied B0 model_20.pt hash differs from the source')
    manifest = {
        'candidate': 'N12/SRPRv2 B0 control',
        'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'original_b0_20': str(source), 'original_b0_20_model_sha256': source_hash,
        'resume_from_epoch': 20, 'target_epoch': 40,
        'scheduler_t_max': 150, 'data_stream_replay_epochs': 20,
        'output': str(output),
    }
    (output / 'b0_40_resume_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
    )
    command = [
        args.python_bin, 'main.py', '--dir_data', str(args.data_root),
        '--model', 'LFMN', '--data_train', 'DIV2K', '--data_test', 'DIV2K',
        '--data_range', '1-800/801-900', '--scale', '4', '--patch_size', '256',
        '--batch_size', '4', '--n_threads', '8', '--ext', 'img', '--epochs', '40',
        '--test_every', '1000', '--lr', '2e-4', '--scheduler', 'cosine',
        '--scheduler_t_max', '150', '--eta_min', '1e-6', '--loss', '1*L1',
        '--seed', '1', '--print_every', '100', '--save_per_image_metrics',
        '--experiment_root', str(output.parent), '--load', output.name,
        '--resume', '20', '--resume_data_epochs', '20',
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED='1')
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT / 'LFMN', env=env, check=True)
    select_b0(SimpleNamespace(b0_20=source, b0_40=output, search_root=[],
                              inspect_b0=False))
    print(f'Validated B0-40: {output}', flush=True)


if __name__ == '__main__':
    main()
