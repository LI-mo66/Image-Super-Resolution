#!/usr/bin/env python3
"""Validate a reusable baseline or summarize it against a new candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
from summarize_candidate_screen import load_curve

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'model': 'LFMN', 'epochs': '5', 'test_every': '1000', 'lr': '1e-05',
    'optimizer': 'ADAM', 'seed': '1', 'n_threads': '8', 'batch_size': '4',
    'patch_size': '256', 'data_range': '1-800/801-810', 'loss': '1*L1',
    'scale': '[4]', 'self_ensemble': 'False', 'chop': 'False',
    'max_train_batches': '0', 'ext': 'img', 'no_augment': 'False',
    'precision': 'single', 'rgb_range': '255', 'weight_decay': '0',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    config = dict(line.split(': ', 1) for line in
                  (args.baseline / 'config.txt').read_text().splitlines() if ': ' in line)
    for key, value in EXPECTED.items():
        if config.get(key) != value:
            raise ValueError(f'baseline mismatch: {key}={config.get(key)!r}; expected {value}')
    if Path(config.get('pre_train', '')).name != 'scale4_model_939.pt':
        raise ValueError('baseline pretrained source mismatch')
    curves = {}
    for name, path in [('baseline', args.baseline), ('prior_update', args.candidate)]:
        if path is None:
            continue
        p, s = load_curve(path, 'psnr_log.pt'), load_curve(path, 'ssim_log.pt')
        if len(p) != 5 or len(s) != 5:
            raise ValueError(f'{name}: expected five completed epochs')
        curves[name] = (p, s)
        loss_path = path / 'loss_log.pt'
        loss = torch.load(loss_path, map_location='cpu', weights_only=True) if loss_path.exists() else None
        if loss is not None and (loss.ndim != 2 or len(loss) != 5):
            raise ValueError('unexpected loss history shape')
        print(f'\n{name}: {path}')
        print('epoch | PSNR | SSIM | L1')
        for i in range(5):
            l1 = f'{loss[i, 0].item():.6f}' if loss is not None else 'MISSING'
            print(f'{i+1} | {p[i]:.6f} | {s[i]:.6f} | {l1}')
    print('\nname | final | best | last3 | final delta | last3 delta | SSIM')
    bp = curves['baseline'][0]
    for name, (p, s) in curves.items():
        print(f'{name} | {p[-1]:.6f} | {p.max():.6f} | {p[-3:].mean():.6f} | '
              f'{p[-1]-bp[-1]:+.6f} | {p[-3:].mean()-bp[-3:].mean():+.6f} | {s[-1]:.6f}')
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = ROOT / 'LFMN/model/scale4_model_939.pt'
        manifest = {
            'baseline': str(args.baseline.resolve()), 'baseline_config': config,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
            'historical_limit': 'Baseline historical checkpoint hash/environment not independently recorded; '
                                'independent DataLoader generator established in f157269 source and user run record.',
        }
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('\nBaseline protocol check passed; no baseline training requested.')


if __name__ == '__main__':
    main()
