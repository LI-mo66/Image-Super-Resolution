#!/usr/bin/env python3
"""Synthetic end-to-end check for the N16 150-epoch continuation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
MODELS = {'c1': 'LFMN', 'srpr_c1': 'LFMNSRPRV2'}
sys.path.insert(0, str(ROOT / 'repro'))
from run_n16_srpr_c1_150e_server import (  # noqa: E402
    SOURCE_COMMON, copy_source, preflight,
)


def write_curve(path, values):
    torch.save(torch.tensor(values, dtype=torch.float64).view(-1, 1, 1), path)


def config_text(name, model):
    values = {
        'data_range': '1-800/801-900', 'scale': '[4]',
        'patch_size': '256', 'batch_size': '4', 'n_threads': '8',
        'ext': 'img', 'epochs': '150', 'lr': '0.0002',
        'scheduler': 'cosine', 'scheduler_t_max': '150',
        'eta_min': '1e-06', 'loss': '1*L1', 'seed': '1',
        'pre_train': '', 'resume': '40', 'resume_data_epochs': '40',
        'save_per_image_metrics': 'True', 'rgcrd_mode': 'output',
        'rgcrd_lambda_output': '0.1', 'model': model, 'load': name,
    }
    return ''.join('{}: {}\n'.format(key, value) for key, value in values.items())


def source_config_text(name, model):
    values = {
        **SOURCE_COMMON, 'n_threads': '8', 'model': model, 'load': name
    }
    return ''.join('{}: {}\n'.format(key, value) for key, value in values.items())


def main():
    temporary_root = ROOT / 'experiment'
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        root = Path(temporary)
        source = root / 'source40'
        output = root / 'output150'
        continuation = root / 'copied150'
        base_psnr = [28.0 + 0.005 * epoch for epoch in range(150)]
        base_ssim = [0.80 + 0.0001 * epoch for epoch in range(150)]
        for name, model in MODELS.items():
            source_run = source / name
            output_run = output / name
            (source_run / 'per_image_metrics').mkdir(parents=True)
            (output_run / 'per_image_metrics').mkdir(parents=True)
            offset = 0.0 if name == 'c1' else 0.03
            source_offset = 0.0 if name == 'c1' else 0.031
            source_psnr = [value + source_offset for value in base_psnr[:40]]
            source_ssim = [
                value + source_offset / 100 for value in base_ssim[:40]
            ]
            continued_psnr = source_psnr + [
                value + offset for value in base_psnr[40:]
            ]
            continued_ssim = source_ssim + [
                value + offset / 100 for value in base_ssim[40:]
            ]
            write_curve(source_run / 'psnr_log.pt', source_psnr)
            write_curve(source_run / 'ssim_log.pt', source_ssim)
            (source_run / 'config.txt').write_text(
                source_config_text(name, model), encoding='utf-8'
            )
            (source_run / 'model').mkdir()
            (source_run / 'model/model_40.pt').write_bytes(
                ('checkpoint-' + name).encode('utf-8')
            )
            torch.save(
                {'state': {0: {'step': torch.tensor(1.)}},
                 'param_groups': [{'params': [0]}]},
                source_run / 'optimizer.pt',
            )
            torch.save(
                {'last_epoch': 40, 'T_max': 150},
                source_run / 'scheduler.pt',
            )
            torch.save(torch.ones(1), source_run / 'loss.pt')
            torch.save(torch.ones(1), source_run / 'loss_log.pt')
            torch.save(
                {'columns': ('epoch',),
                 'rows': [[float(epoch)] for epoch in range(1, 41)]},
                source_run / 'rgcrd_log.pt',
            )
            source_rows = [
                {
                    'dataset': 'DIV2K', 'scale': 4,
                    'filename': '{:04d}'.format(index),
                    'psnr': 30.0 + source_offset,
                    'ssim': 0.9 + source_offset / 100,
                }
                for index in range(100)
            ]
            torch.save(
                source_rows, source_run / 'per_image_metrics/epoch_0040.pt'
            )
            write_curve(output_run / 'psnr_log.pt', continued_psnr)
            write_curve(output_run / 'ssim_log.pt', continued_ssim)
            (output_run / 'config.txt').write_text(
                config_text(name, model), encoding='utf-8'
            )
            torch.save(
                {'last_epoch': 150, 'T_max': 150},
                output_run / 'scheduler.pt',
            )
            rows = [
                {
                    'dataset': 'DIV2K', 'scale': 4,
                    'filename': '{:04d}'.format(index),
                    'psnr': 30.0 + offset,
                    'ssim': 0.9 + offset / 100,
                }
                for index in range(100)
            ]
            torch.save(
                rows, output_run / 'per_image_metrics/epoch_0150.pt'
            )

        (source / 'wrapper_exit_status.txt').write_text('0\n', encoding='utf-8')
        (source / 'summary_40e.txt').write_text(
            'synthetic source\n', encoding='utf-8'
        )
        (source / 'summary_40e.json').write_text(json.dumps({
            'decision': 'PROMOTE_TO_150E_CONFIRMATION',
            'final': {'psnr_delta': 0.031},
        }), encoding='utf-8')
        data_root = root / 'datasets'
        (data_root / 'DIV2K/DIV2K_train_HR').mkdir(parents=True)
        (data_root / 'DIV2K/DIV2K_train_HR/0001.png').write_bytes(b'image')
        teacher_repo = root / 'swinir'
        (teacher_repo / 'models').mkdir(parents=True)
        (teacher_repo / 'models/network_swinir.py').write_text(
            '# synthetic\n', encoding='utf-8'
        )
        teacher_checkpoint = root / 'teacher.pth'
        teacher_checkpoint.write_bytes(b'teacher')
        preflight_args = SimpleNamespace(
            source=source, output=continuation, data_root=data_root,
            teacher_repo=teacher_repo,
            teacher_checkpoint=teacher_checkpoint,
        )
        checked_source, checked_output, groups = preflight(preflight_args)
        copy_source(checked_source, checked_output, groups)
        assert (continuation / 'c1/model/model_40.pt').is_file()
        assert (continuation / 'srpr_c1/optimizer.pt').is_file()

        command = [
            sys.executable,
            str(ROOT / 'repro/summarize_n16_srpr_c1_150e.py'),
            str(output), '--source-40', str(source),
        ]
        result = subprocess.run(
            command, cwd=ROOT, check=True, text=True, capture_output=True
        )
        payload = json.loads(
            (output / 'summary_150e.json').read_text(encoding='utf-8')
        )
        assert payload['first40_history_exact'] is True
        assert payload['decision'] == 'PROMOTE_TO_LONG_RUN_VALIDATION'
        assert abs(payload['final']['psnr_delta'] - 0.03) < 1e-10
        assert 'first40_history_exact=True' in result.stdout
        print(json.dumps({
            'n16_150e_summary_check': 'passed',
            'source_preflight_and_hash_copy': 'passed',
            'history_integrity_guard': 'passed',
            'paired_statistics': 'passed',
            'registered_decision': payload['decision'],
        }, indent=2))


if __name__ == '__main__':
    main()
