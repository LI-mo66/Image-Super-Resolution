#!/usr/bin/env python3
"""Synthetic integrity checks for the N16 formal 1000-epoch workflow."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import uuid

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scheduler_state(epoch):
    return {
        'milestones': Counter({200: 1, 400: 1, 600: 1, 800: 1}),
        'gamma': 0.5, 'base_lrs': [2e-4], 'last_epoch': epoch,
        'verbose': False, '_step_count': epoch + 1,
        '_get_lr_called_within_step': False, '_last_lr': [2e-4],
    }


def write_state(run, epoch, protocol, model_state, image_count=1):
    (run / 'model').mkdir(parents=True)
    (run / 'per_image_metrics').mkdir()
    config = ''.join('{}: {}\n'.format(key, value)
                     for key, value in protocol.items())
    (run / 'config.txt').write_text(config, encoding='utf-8')
    torch.save(torch.zeros(epoch, 1, 1), run / 'psnr_log.pt')
    torch.save(torch.zeros(epoch, 1, 1), run / 'ssim_log.pt')
    torch.save(model_state, run / 'model/model_{}.pt'.format(epoch))
    torch.save({'state': {0: {'step': torch.tensor(1.)}},
                'param_groups': [{'params': [0]}]}, run / 'optimizer.pt')
    torch.save(scheduler_state(epoch), run / 'scheduler.pt')
    torch.save({}, run / 'loss.pt')
    torch.save(torch.zeros(epoch, 1), run / 'loss_log.pt')
    torch.save({'rows': [[float(index)] for index in range(1, epoch + 1)]},
               run / 'rgcrd_log.pt')
    torch.save([{'index': index} for index in range(image_count)],
               run / 'per_image_metrics/epoch_{:04d}.pt'.format(epoch))


def main():
    runner = load_module(
        'n16_long_runner', ROOT / 'repro/run_n16_champion_1000e_server.py'
    )
    sys.path.insert(0, str(ROOT / 'LFMN'))
    from model.lfmnsrprv2 import Net as SRPRv2Net
    model_state = SRPRv2Net(scale=4).state_dict()
    commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True
    ).strip()
    check_parent = ROOT / 'repro/_n16_check_workspace'
    check_parent.mkdir(exist_ok=True)
    temp_root = check_parent / ('run_' + uuid.uuid4().hex)
    temp_root.mkdir()
    try:
        partial = temp_root / 'partial'
        write_state(partial, 40, runner.PROTOCOL, model_state)
        (partial / 'long_run_manifest.json').write_text(json.dumps({
            'source_commit': commit, 'protocol': runner.PROTOCOL,
        }), encoding='utf-8')
        assert runner.validate_resume(partial, commit) == 40

        pre_epoch = temp_root / 'pre_epoch'
        (pre_epoch / 'model').mkdir(parents=True)
        (pre_epoch / 'config.txt').write_text(
            ''.join('{}: {}\n'.format(key, value)
                    for key, value in runner.PROTOCOL.items()),
            encoding='utf-8',
        )
        (pre_epoch / 'long_run_manifest.json').write_text(json.dumps({
            'source_commit': commit, 'protocol': runner.PROTOCOL,
        }), encoding='utf-8')
        assert runner.validate_resume(pre_epoch, commit) == 0

        args = SimpleNamespace(
            python_bin='python', data_root=ROOT / 'datasets',
            experiment_root=ROOT / 'experiment/all_runs',
            teacher_repo=ROOT / 'repro/swinir_ref',
            teacher_checkpoint=ROOT / 'repro/teacher_weights/teacher.pth',
            run_name='n16/formal', gpu='0', check_only=False,
        )
        checked = {'data_root': ROOT / 'datasets',
                   'teacher_repo': ROOT / 'repro/swinir_ref',
                   'teacher_checkpoint': ROOT / 'teacher.pth',
                   'completed_epochs': 40}
        command = runner.training_command(args, checked)
        assert command[command.index('--epochs') + 1] == '1000'
        assert command[command.index('--scheduler') + 1] == 'multistep'
        assert command[command.index('--decay') + 1] == '200-400-600-800'
        assert command[command.index('--resume') + 1] == '40'
        assert command[command.index('--resume_data_epochs') + 1] == '40'

        complete = temp_root / 'complete'
        write_state(complete, 1000, runner.PROTOCOL, model_state, image_count=100)
        (complete / 'long_run_manifest.json').write_text(json.dumps({
            'source_commit': commit, 'protocol': runner.PROTOCOL,
        }), encoding='utf-8')
        torch.save(torch.linspace(28, 29, 1000).reshape(-1, 1, 1),
                   complete / 'psnr_log.pt')
        torch.save(torch.linspace(.82, .84, 1000).reshape(-1, 1, 1),
                   complete / 'ssim_log.pt')
        result = subprocess.run([
            sys.executable, str(ROOT / 'repro/summarize_n16_champion_1000e.py'),
            str(complete),
        ], cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE,
           stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(result.stdout)
        summary = json.loads((complete / 'summary_1000e.json').read_text())
        assert summary['decision'] == 'READY_FOR_FIXED_EPOCH1000_BENCHMARKS'
        assert summary['parameters'] == 841563
        assert 'epoch_1000' in result.stdout

        benchmark = temp_root / 'benchmark'
        metrics_dir = benchmark / 'champion/per_image_metrics'
        metrics_dir.mkdir(parents=True)
        reference = {
            'Set5': (5, 32.64, .9004), 'Set14': (14, 28.95, .7893),
            'B100': (100, 27.80, .7438), 'Urban100': (100, 26.91, .8087),
            'Manga109': (109, 31.45, .9193),
        }
        rows = []
        for dataset, (count, psnr, ssim) in reference.items():
            rows.extend({
                'dataset': dataset, 'scale': 4,
                'filename': '{}_{:03d}'.format(dataset, index),
                'psnr': psnr + .01, 'ssim': ssim + .0001,
            } for index in range(count))
        torch.save(rows, metrics_dir / 'epoch_0000.pt')
        subprocess.run([
            sys.executable,
            str(ROOT / 'repro/summarize_n16_champion_benchmarks.py'),
            str(benchmark),
        ], cwd=ROOT, check=True, text=True, capture_output=True)
        classification = json.loads(
            (benchmark / 'benchmark_summary.json').read_text(encoding='utf-8')
        )
        assert classification['category'] == 'A'
    finally:
        shutil.rmtree(temp_root)

    print(json.dumps({
        'n16_champion_1000e_check': 'passed',
        'resume_state_integrity': 'passed',
        'pre_epoch_restart': 'passed',
        'paper_protocol_command': 'passed',
        'fixed_epoch1000_summary': 'passed',
        'benchmark_abc_classification': 'passed',
    }, indent=2))


if __name__ == '__main__':
    main()
