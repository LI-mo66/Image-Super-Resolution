#!/usr/bin/env python3
"""Compare epoch trajectories and optionally backfill Set5 from saved weights."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from run_logging import launch, write_json


def read_epochs(directory):
    path = directory / 'metrics.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as stream:
        lines = stream.readlines()
    if lines and not lines[-1].endswith('\n'):
        lines.pop()
    rows = list(csv.DictReader(lines))
    result = {}
    previous = 0
    for row in rows:
        epoch = int(row['epoch'])
        if epoch <= previous:
            raise ValueError('duplicate or decreasing metric epoch: ' + str(path))
        previous = epoch
        for key, value in row.items():
            if key != 'epoch' and value:
                if not math.isfinite(float(value)):
                    raise ValueError('non-finite metric: ' + str(path))
        result[epoch] = row
    return result


def load_set5(path):
    import torch
    if not path.exists():
        return None
    rows = torch.load(path, map_location='cpu', weights_only=True)
    selected = [row for row in rows if row['dataset'] == 'Set5' and row['scale'] == 4]
    if not selected:
        return None
    if len(selected) != 5 or len({row['filename'] for row in selected}) != 5:
        raise ValueError('invalid five-image Set5 report: ' + str(path))
    for row in selected:
        if not all(math.isfinite(float(row[key])) for key in ('psnr', 'ssim')):
            raise ValueError('non-finite Set5 per-image metric: ' + str(path))
    return {row['filename']: row for row in selected}


def backfill(args, output, label, epoch):
    import torch
    source = args.group / (label + '_x4_seed1') / 'model' / ('model_{}.pt'.format(epoch))
    if not source.exists():
        raise FileNotFoundError('cannot backfill Set5 without checkpoint: ' + str(source))
    directory = output / 'backfill' / label / ('epoch_{:04d}'.format(epoch))
    model = 'LFMN' if label == 'B0' else 'LFMNF1'
    command = [sys.executable, '-u', str(ROOT / 'repro/f1_train_entry.py'),
               '--dir_data', str(args.data_root), '--model', model, '--scale', '4',
               '--test_only', '--pre_train', str(source), '--data_test', 'Set5',
               '--n_threads', str(args.workers), '--ext', 'img', '--save_per_image_metrics',
               '--experiment_root', str(directory.parent), '--save', directory.name]
    if args.cpu:
        command += ['--cpu']
    launch(command, directory, {'kind': 'Set5 backfill', 'model': model,
           'source_epoch': epoch, 'source_checkpoint': str(source),
           'self_ensemble': False, 'source_run_config': str(source.parents[1] / 'config.json'),
           'checkpoint_selection': 'all saved completed epochs; monitoring only',
           'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'torch_version': torch.__version__,
           'gpu_name': 'CPU' if args.cpu else torch.cuda.get_device_name(0),
           'metric_source_sha256': hashlib.sha256((ROOT / 'LFMN/utility.py').read_bytes()).hexdigest(),
           'metric_protocol': 'same utility.py benchmark Y/quantize255/crop4'}, ROOT / 'LFMN')
    return load_set5(directory / 'per_image_metrics/epoch_0000.pt')


def export(args):
    group = args.group.resolve()
    args.group = group
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output.resolve() if args.output else group / 'comparison_reports' / stamp
    output.mkdir(parents=True, exist_ok=False)
    histories = {label: read_epochs(group / (label + '_x4_seed1')) for label in ('B0', 'F1')}
    epochs = sorted(set(histories['B0']) | set(histories['F1']))
    if not epochs:
        raise ValueError('no completed training epochs found in ' + str(group))
    configs = {}
    for label in histories:
        path = group / (label + '_x4_seed1') / 'config.json'
        configs[label] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if all(configs.values()):
        keys = ('scale', 'seed', 'patch_size', 'batch_size', 'optimizer', 'lr', 'eta_min',
                'scheduler', 'scheduler_t_max', 'data_range', 'loss', 'self_ensemble',
                'pretrained_checkpoint', 'precision', 'rgb_range', 'workers', 'data_root',
                'epoch_evaluation')
        for key in keys:
            if configs['B0'].get(key) != configs['F1'].get(key):
                raise ValueError('B0/F1 protocol differs: ' + key)
        left_init = configs['B0'].get('initial_shared_state_sha256')
        right_init = configs['F1'].get('initial_shared_state_sha256')
        if left_init and right_init and left_init != right_init:
            raise ValueError('B0/F1 shared scratch initialization differs')
    if args.backfill_set5:
        metric_hash = hashlib.sha256((ROOT / 'LFMN/utility.py').read_bytes()).hexdigest()
        for config in configs.values():
            source = (config or {}).get('source_sha256', {}).get('LFMN/utility.py')
            if source and source != metric_hash:
                raise ValueError('metric source differs from original training/evaluation protocol')
    set5 = {label: {} for label in histories}
    for label, history in histories.items():
        directory = group / (label + '_x4_seed1')
        for epoch in history:
            rows = load_set5(directory / 'per_image_metrics' / ('epoch_{:04d}.pt'.format(epoch)))
            if rows is None and args.backfill_set5:
                if args.data_root is None:
                    raise ValueError('--backfill-set5 requires --data-root')
                rows = backfill(args, output, label, epoch)
            if rows is not None:
                set5[label][epoch] = rows
    comparison = []
    image_comparison = []
    for epoch in epochs:
        record = {'epoch': epoch}
        for label in ('B0', 'F1'):
            row = histories[label].get(epoch, {})
            for key in ('learning_rate', 'train_loss', 'validation_psnr', 'validation_ssim'):
                record[label + '_' + key] = row.get(key, '')
            images = set5[label].get(epoch)
            for metric in ('psnr', 'ssim'):
                value = sum(image[metric] for image in images.values()) / 5 if images else row.get('set5_' + metric, '')
                record[label + '_set5_' + metric] = value
        for metric in ('validation_psnr', 'validation_ssim', 'set5_psnr', 'set5_ssim'):
            a, b = record['B0_' + metric], record['F1_' + metric]
            record['delta_' + metric] = float(b) - float(a) if a != '' and b != '' else ''
        left, right = set5['B0'].get(epoch), set5['F1'].get(epoch)
        if left and right and set(left) != set(right):
            raise ValueError('Set5 filenames differ between B0/F1 at epoch {}'.format(epoch))
        for name in sorted(set(left or {}) | set(right or {})):
            image = {'epoch': epoch, 'filename': name}
            for label, images in [('B0', left), ('F1', right)]:
                for metric in ('psnr', 'ssim'):
                    image[label + '_' + metric] = images[name][metric] if images and name in images else ''
            for metric in ('psnr', 'ssim'):
                a, b = image['B0_' + metric], image['F1_' + metric]
                image['delta_' + metric] = float(b) - float(a) if a != '' and b != '' else ''
            image_comparison.append(image)
        comparison.append(record)
    write_csv(output / 'epoch_comparison.csv', comparison, list(comparison[0]))
    image_columns = ['epoch', 'filename', 'B0_psnr', 'B0_ssim', 'F1_psnr', 'F1_ssim', 'delta_psnr', 'delta_ssim']
    write_csv(output / 'set5_per_image_comparison.csv', image_comparison, image_columns)
    lines = ['# B0/F1 paired comparison', '',
             'Self-Ensemble OFF. Set5 is monitored only; checkpoint selection uses independent validation or fixed endpoints.', '',
             '| Epoch | B0 DIV2K | F1 DIV2K | Delta DIV2K | B0 Set5 | F1 Set5 | Delta Set5 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    keys = ['B0_validation_psnr', 'F1_validation_psnr', 'delta_validation_psnr',
            'B0_set5_psnr', 'F1_set5_psnr', 'delta_set5_psnr']
    for row in comparison:
        lines.append('| {} | {} |'.format(row['epoch'], ' | '.join(format_number(row[key]) for key in keys)))
    lines += ['', 'Missing entries mean pending/missing tests, never zero or interpolated values.', '',
              'SSIM, loss, learning rate and all five Set5 images per epoch are in the CSV files.', '',
              '## Last Common Epoch and Last Five Epochs', '']
    paired = sorted(set(histories['B0']) & set(histories['F1']))
    if paired:
        endpoint = paired[-1]
        lines.append('Last common completed epoch: {}. Fixed screening endpoint: 20.'.format(endpoint))
        for metric in ('validation_psnr', 'set5_psnr'):
            ending = [row for row in comparison if endpoint - 4 <= row['epoch'] <= endpoint]
            if len(ending) == 5 and all(row['delta_' + metric] != '' for row in ending):
                delta = sum(row['delta_' + metric] for row in ending) / 5
                lines.append('Last five epoch mean delta {}: {:+.6f} dB.'.format(metric, delta))
            else:
                lines.append('Last five epoch mean delta {}: unavailable (need five complete paired epochs).'.format(metric))
    else:
        lines.append('No paired epochs yet; relative performance cannot be judged.')
    benchmark_path = group / 'benchmark_comparison.csv'
    lines += ['', '## Fixed Epoch20 Benchmarks', '']
    if benchmark_path.exists():
        with benchmark_path.open(encoding='utf-8') as stream:
            benchmark_rows = list(csv.DictReader(stream))
        lines += ['| Dataset | B0 PSNR | F1 PSNR | Delta PSNR | B0 SSIM | F1 SSIM | Delta SSIM |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for row in benchmark_rows:
            lines.append('| {} | {} |'.format(row['dataset'], ' | '.join(format_number(row[key]) for key in
                         ('B0_psnr', 'F1_psnr', 'delta_psnr', 'B0_ssim', 'F1_ssim', 'delta_ssim'))))
    else:
        lines.append('Pending: Set5, Set14, B100, Urban100 and Manga109 fixed epoch20 evaluation.')
    lines += ['', '## Resource Comparison', '']
    resource_path = group / 'resources.json'
    if resource_path.exists():
        resource = json.loads(resource_path.read_text(encoding='utf-8'))
        lines.append('Device: {}. {}'.format(resource.get('gpu'), resource.get('coverage_limit')))
        lines += ['', '| LR size | Model | Params | Counted FLOPs | Median ms | P90 ms | Peak allocated MiB | Peak reserved MiB |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for size in (64, 128):
            for label in ('B0', 'F1'):
                row = resource.get('results', {}).get(label + '_LR' + str(size))
                if row:
                    lines.append('| {} | {} | {} |'.format(size, label, ' | '.join(format_number(row[key]) for key in
                                 ('parameters', 'counted_flops', 'median_ms', 'p90_ms', 'peak_allocated_mib', 'peak_reserved_mib'))))
    else:
        lines.append('Pending resource profiling; no efficiency claim.')
    lines += ['', 'No average across benchmark datasets. No best-Set5 checkpoint comparison.']
    if not all(configs.values()):
        lines.append('Protocol fairness remains unverified: one or both source config.json files are missing.')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output / 'manifest.json', {'source_group': str(group), 'source_configs': configs,
               'backfill_set5': args.backfill_set5, 'completed_epochs': {key: sorted(value) for key, value in histories.items()},
               'set5_available_epochs': {key: sorted(value) for key, value in set5.items()},
               'checkpoint_selection': 'fixed epoch20; no benchmark-based selection'})
    print('\n'.join(lines), flush=True)
    print('COMPARISON REPORT: ' + str(output), flush=True)
    return output


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def format_number(value):
    return '{:.6f}'.format(float(value)) if value != '' else 'pending'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backfill-set5', action='store_true')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError('workers must be nonnegative')
    if args.data_root:
        args.data_root = args.data_root.resolve()
    export(args)


if __name__ == '__main__':
    main()
