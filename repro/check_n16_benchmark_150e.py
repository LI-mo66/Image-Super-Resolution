#!/usr/bin/env python3
"""Synthetic check for the fixed epoch-150 N16 benchmark evaluation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
COUNTS = {'Set5': 5, 'Set14': 14, 'B100': 100,
          'Urban100': 100, 'Manga109': 109}


def main():
    temporary_root = ROOT / 'experiment'
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        root = Path(temporary)
        run = root / 'benchmark_output'
        for group, delta in (('c1', 0.0), ('srpr_c1', 0.03)):
            metrics = run / group / 'per_image_metrics'
            metrics.mkdir(parents=True)
            rows = []
            for dataset, count in COUNTS.items():
                rows.extend({
                    'dataset': dataset, 'scale': 4,
                    'filename': '{:03d}'.format(index),
                    'psnr': 30.0 + delta, 'ssim': 0.9 + delta / 100,
                } for index in range(count))
            torch.save(rows, metrics / 'epoch_0000.pt')
        result = subprocess.run([
            sys.executable, str(ROOT / 'repro/summarize_n16_benchmarks.py'),
            str(run), '--checkpoint-epoch', '150',
        ], cwd=ROOT, text=True, capture_output=True, check=True)
        payload = json.loads(
            (run / 'benchmark_summary.json').read_text(encoding='utf-8')
        )
        assert payload['decision'] == 'BROAD_EXTERNAL_SUPPORT'
        assert payload['positive_datasets'] == 5
        assert payload['checkpoint_selection'] == (
            'fixed preregistered epoch150 endpoint'
        )
        assert abs(payload['pooled_image_mean_delta'] - 0.03) < 1e-10
        assert 'fixed epoch-150' in result.stdout

        source = root / 'source150'
        (source / 'wrapper_exit_status.txt').parent.mkdir(parents=True)
        (source / 'wrapper_exit_status.txt').write_text('0\n', encoding='utf-8')
        (source / 'summary_150e.json').write_text(json.dumps({
            'decision': 'PROMOTE_TO_LONG_RUN_VALIDATION'
        }), encoding='utf-8')
        sys.path.insert(0, str(ROOT / 'LFMN'))
        from model.lfmn import Net as BaselineNet
        from model.lfmnsrprv2 import Net as SRPRv2Net
        for group, model_class in (
            ('c1', BaselineNet), ('srpr_c1', SRPRv2Net)
        ):
            checkpoint_dir = source / group / 'model'
            checkpoint_dir.mkdir(parents=True)
            torch.save(
                model_class(scale=4).state_dict(),
                checkpoint_dir / 'model_150.pt',
            )
        data_root = root / 'datasets'
        for dataset, count in COUNTS.items():
            hr = data_root / 'benchmark' / dataset / 'HR'
            lr = data_root / 'benchmark' / dataset / 'LR_bicubic/X4'
            hr.mkdir(parents=True)
            lr.mkdir(parents=True)
            for index in range(count):
                (hr / '{:03d}.png'.format(index)).write_bytes(b'hr')
                (lr / '{:03d}x4.png'.format(index)).write_bytes(b'lr')
        sys.path.insert(0, str(ROOT / 'repro'))
        from run_n16_benchmarks_150e_server import validate
        _, _, checkpoints = validate(SimpleNamespace(
            source=source, output=root / 'unused_output',
            data_root=data_root,
        ))
        assert set(checkpoints) == {'c1', 'srpr_c1'}
        print(json.dumps({
            'n16_benchmark_150e_check': 'passed',
            'strict_epoch150_checkpoint_load': 'passed',
            'dataset_counts': 'passed',
            'paired_statistics': 'passed',
            'checkpoint_selection': payload['checkpoint_selection'],
            'decision': payload['decision'],
        }, indent=2))


if __name__ == '__main__':
    main()
