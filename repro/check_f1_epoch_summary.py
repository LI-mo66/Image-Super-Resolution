#!/usr/bin/env python3
"""Check paired epoch arithmetic, missing data, image pairing and protocol guards."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro'))
from summarize_f1_screen import export


def main():
    scratch = ROOT / 'experiment'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='f1_summary_test_', dir=scratch) as temp:
        group = Path(temp)
        for label, offset in [('B0', 0.0), ('F1', 0.05)]:
            directory = group / (label + '_x4_seed1')
            images = directory / 'per_image_metrics'
            images.mkdir(parents=True)
            (directory / 'config.json').write_text(json.dumps({'seed': 1}), encoding='utf-8')
            with (directory / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['epoch', 'learning_rate', 'train_loss', 'validation_psnr', 'validation_ssim'])
                writer.writeheader()
                for epoch in range(1, 7):
                    writer.writerow(dict(epoch=epoch, learning_rate=2e-4, train_loss=1.0,
                                         validation_psnr=20 + epoch / 10 + offset, validation_ssim=0.8))
                    torch.save([dict(dataset='Set5', scale=4, filename=str(index),
                                     psnr=25 + epoch / 10 + offset + index / 100,
                                     ssim=0.8) for index in range(5)],
                               images / ('epoch_{:04d}.pt'.format(epoch)))
        args = argparse.Namespace(group=group, output=None, backfill_set5=False,
                                  data_root=None, workers=2, cpu=True)
        output = export(args)
        with (output / 'epoch_comparison.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 6
        assert all(abs(float(row['delta_set5_psnr']) - 0.05) < 1e-10 for row in rows)
        with (output / 'set5_per_image_comparison.csv').open(encoding='utf-8') as stream:
            assert len(list(csv.DictReader(stream))) == 30
        report = (output / 'comparison.md').read_text(encoding='utf-8')
        assert 'Last five epoch mean delta set5_psnr: +0.050000' in report
        resource = {'gpu': 'fixture', 'coverage_limit': 'fixture counter scope', 'results': {
            'B0_LR64': dict(parameters=759627, counted_flops=0, median_ms=1.0,
                           p90_ms=1.1, peak_allocated_mib=10.0, peak_reserved_mib=12.0)}}
        (group / 'resources.json').write_text(json.dumps(resource), encoding='utf-8')
        output = export(args)
        assert 'invalid counter' in (output / 'comparison.md').read_text(encoding='utf-8')
        missing = group / 'F1_x4_seed1/per_image_metrics/epoch_0006.pt'
        missing.unlink()
        output = export(args)
        with (output / 'epoch_comparison.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert rows[-1]['F1_set5_psnr'] == '' and rows[-1]['delta_set5_psnr'] == ''
        wrong = [dict(dataset='Set5', scale=4, filename=str(index + 10), psnr=25.0, ssim=0.8) for index in range(5)]
        torch.save(wrong, missing)
        try:
            export(args)
            raise AssertionError('mismatched image names were accepted')
        except ValueError as error:
            assert 'filenames differ' in str(error)
        (group / 'F1_x4_seed1/config.json').write_text(json.dumps({'seed': 2}), encoding='utf-8')
        try:
            export(args)
            raise AssertionError('mismatched seeds were accepted')
        except ValueError as error:
            assert 'protocol differs' in str(error)
    print('EPOCH SUMMARY PASSED: deltas, per-image rows, missing values, five-epoch window, pairing, protocol')


if __name__ == '__main__':
    main()
