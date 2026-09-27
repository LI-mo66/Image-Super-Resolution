#!/usr/bin/env python3
"""Synthetic check of the paired N16 benchmark summarizer."""
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
        run = Path(temporary)
        for group, delta in (('c1', 0.0), ('srpr_c1', 0.02)):
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
            str(run),
        ], cwd=ROOT, text=True, capture_output=True, check=True)
        payload = json.loads(
            (run / 'benchmark_summary.json').read_text(encoding='utf-8')
        )
        assert payload['decision'] == 'BROAD_EXTERNAL_SUPPORT'
        assert payload['positive_datasets'] == 5
        assert abs(payload['pooled_image_mean_delta'] - 0.02) < 1e-10
        assert 'positive_datasets=5/5' in result.stdout

        source = run / 'source40'
        (source / 'wrapper_exit_status.txt').parent.mkdir(parents=True)
        (source / 'wrapper_exit_status.txt').write_text('0\n', encoding='utf-8')
        (source / 'summary_40e.json').write_text(json.dumps({
            'decision': 'PROMOTE_TO_150E_CONFIRMATION'
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
                checkpoint_dir / 'model_40.pt',
            )
        data_root = run / 'datasets'
        for dataset, count in COUNTS.items():
            hr = data_root / 'benchmark' / dataset / 'HR'
            lr = data_root / 'benchmark' / dataset / 'LR_bicubic/X4'
            hr.mkdir(parents=True)
            lr.mkdir(parents=True)
            for index in range(count):
                (hr / '{:03d}.png'.format(index)).write_bytes(b'hr')
                (lr / '{:03d}x4.png'.format(index)).write_bytes(b'lr')
        sys.path.insert(0, str(ROOT / 'repro'))
        from run_n16_benchmarks_server import validate
        _, _, checkpoints = validate(SimpleNamespace(
            source=source, output=run / 'benchmark_output',
            data_root=data_root,
        ))
        assert set(checkpoints) == {'c1', 'srpr_c1'}
        print(json.dumps({
            'n16_benchmark_summary_check': 'passed',
            'checkpoint_and_dataset_preflight': 'passed',
            'dataset_partitioning': 'passed',
            'paired_statistics': 'passed',
            'decision': payload['decision'],
        }, indent=2))


if __name__ == '__main__':
    main()
