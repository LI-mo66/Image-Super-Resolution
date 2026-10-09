#!/usr/bin/env python3
"""Paired scratch x4 screening; stop at epoch 20 on a 150-epoch cosine horizon."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from run_logging import launch, write_json

BENCHMARKS = {'Set5': 5, 'Set14': 14, 'B100': 100, 'Urban100': 100, 'Manga109': 109}
PROTOCOL = dict(scale=4, seed=1, patch_size=256, batch_size=4, optimizer='ADAM',
                lr=2e-4, eta_min=1e-6, scheduler='cosine', scheduler_t_max=150,
                planned_epochs=150, stop_epoch=20, test_every=1000,
                data_range='1-800/801-900', loss='1*L1', self_ensemble=False,
                pretrained_checkpoint=None, weight_ema=False,
                tab_centroid_ema=True, precision='single', rgb_range=255)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = ['LFMN/model/lfmn.py', 'LFMN/model/lfmnf1.py', 'LFMN/trainer.py',
             'LFMN/utility.py', 'LFMN/option.py', 'LFMN/data/__init__.py',
             'LFMN/data/common.py', 'LFMN/loss/__init__.py', 'LFMN/run_logging.py',
             'repro/f1_train_entry.py', 'repro/run_f1_screen_server.py']
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def check_data(root):
    from PIL import Image
    pairs = []
    for split, ids in [('train', range(1, 801)), ('valid', range(801, 901))]:
        for index in ids:
            name = '{:04d}'.format(index)
            pairs.append((root / 'DIV2K' / ('DIV2K_' + split + '_HR') / (name + '.png'),
                          root / 'DIV2K' / ('DIV2K_' + split + '_LR_bicubic') / 'X4' / (name + 'x4.png')))
    for dataset, count in BENCHMARKS.items():
        folder = root / 'benchmark' / dataset
        hr_files = sorted(path for path in (folder / 'HR').glob('*') if path.is_file())
        if len(hr_files) != count:
            raise ValueError('{} needs {} HR images, found {}'.format(dataset, count, len(hr_files)))
        pairs.extend((hr, folder / 'LR_bicubic' / 'X4' / (hr.stem + 'x4.png')) for hr in hr_files)
    for hr, lr in pairs:
        if not hr.is_file() or not lr.is_file():
            raise FileNotFoundError('Missing pair: {} / {}'.format(hr, lr))
        with Image.open(hr) as image:
            hsize = image.size
        with Image.open(lr) as image:
            lsize = image.size
        if any(high // 4 != low for high, low in zip(hsize, lsize)):
            raise ValueError('x4 dimensions mismatch: {} / {}'.format(hr, lr))
    print('DATA PREFLIGHT: {} x4 pairs checked'.format(len(pairs)), flush=True)


def train_command(args, directory, model, smoke=False, resume_epoch=0, stop_epoch=20):
    command = [sys.executable, '-u', str(ROOT / 'repro/f1_train_entry.py'),
               '--dir_data', str(args.data_root), '--model', model,
               '--data_train', 'DIV2K', '--data_test', 'DIV2K',
               '--data_range', '1-1/801-801' if smoke else PROTOCOL['data_range'],
               '--scale', '4', '--patch_size', '256',
               '--batch_size', '4', '--n_threads', str(args.workers),
               '--ext', 'img', '--epochs', str(stop_epoch), '--test_every', '1' if smoke else '1000',
               '--lr', '2e-4', '--optimizer', 'ADAM', '--scheduler', 'cosine',
               '--scheduler_t_max', '150', '--eta_min', '1e-6', '--loss', '1*L1',
               '--seed', '1', '--print_every', '1' if smoke else '100',
               '--save_per_image_metrics', '--experiment_root', str(directory.parent),
               '--save', directory.name]
    if args.cpu:
        command += ['--cpu']
    if smoke:
        command += ['--max_train_batches', '1']
    if resume_epoch:
        command += ['--load', directory.name, '--resume', str(resume_epoch)]
    return command


def config(args, model, command, **extra):
    import torch
    result = dict(PROTOCOL, model=model, command=command, cwd=str(ROOT / 'LFMN'),
                dataset='DIV2K', validation='DIV2K 801-900',
                data_root=str(args.data_root), workers=args.workers,
                augmentation=True, gpu_name='CPU' if args.cpu else torch.cuda.get_device_name(0),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
                git_diff_summary=git('diff', '--stat'),
                source_sha256=source_hashes(),
                metric_protocol='utility.py; quantize255; DIV2K RGB/crop10; benchmark PSNR Y BT601-256/crop4; SSIM MATLAB Y/crop4',
                **extra)
    if extra.get('smoke'):
        result.update(patch_size=256, batch_size=4, test_every=1,
                      data_range='1-1/801-801', validation='DIV2K 801 only',
                      stop_epoch=int(command[command.index('--epochs') + 1]), max_train_batches=1)
    return result


def smoke(args, group):
    for label, model in [('B0', 'LFMN'), ('F1', 'LFMNF1')]:
        directory = group / (label + '_smoke_x4_seed1')
        command = train_command(args, directory, model, smoke=True, stop_epoch=1)
        launch(command, directory, config(args, model, command, smoke=True), ROOT / 'LFMN')
        command = train_command(args, directory, model, smoke=True, resume_epoch=1, stop_epoch=2)
        launch(command, directory, config(args, model, command, smoke=True,
               resume_checkpoint=str(directory / 'model/model_1.pt'), resume_start_epoch=1),
               ROOT / 'LFMN', resume=True)
        rows = list(csv.DictReader((directory / 'metrics.csv').open(encoding='utf-8')))
        if [int(row['epoch']) for row in rows] != [1, 2]:
            raise AssertionError('smoke resume metric timeline incorrect')
        log = (directory / 'train_log.txt').read_text(encoding='utf-8', errors='replace')
        if 'Restored RNG/DataLoader state at epoch 1' not in log:
            raise AssertionError('resume trace missing')
    print('GATE 2 PASSED: real DIV2K batch, validation, checkpoints and resume for B0/F1', flush=True)


def evaluate(args, group, label):
    import torch
    source = group / (label + '_x4_seed1') / 'model/model_20.pt'
    directory = group / (label + '_benchmarks_epoch20_OFF')
    if directory.exists():
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        directory = group / (label + '_benchmarks_epoch20_OFF_retry_' + stamp)
    model = 'LFMN' if label == 'B0' else 'LFMNF1'
    command = [sys.executable, '-u', str(ROOT / 'repro/f1_train_entry.py'),
               '--dir_data', str(args.data_root), '--model', model, '--scale', '4',
               '--test_only', '--pre_train', str(source), '--data_test', '+'.join(BENCHMARKS),
               '--n_threads', str(args.workers), '--ext', 'img', '--save_per_image_metrics',
               '--experiment_root', str(group), '--save', directory.name]
    if args.cpu:
        command += ['--cpu']
    launch(command, directory, config(args, model, command, test_only=True,
           checkpoint=str(source), checkpoint_selection='fixed epoch20'), ROOT / 'LFMN')
    rows = torch.load(directory / 'per_image_metrics/epoch_0000.pt', weights_only=True)
    with (directory / 'per_image_metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'scale', 'filename', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(rows)
    paths_file = group / 'evaluation_paths.json'
    paths = json.loads(paths_file.read_text(encoding='utf-8')) if paths_file.exists() else {}
    paths[label] = str(directory)
    write_json(paths_file, paths)
    return rows


def summarize(group, results):
    summary = []
    for dataset, count in BENCHMARKS.items():
        entry = {'dataset': dataset}
        pair_names = None
        for label in ('B0', 'F1'):
            rows = [row for row in results[label] if row['dataset'] == dataset]
            if len(rows) != count:
                raise ValueError('wrong evaluated count for ' + dataset)
            names = {row['filename'] for row in rows}
            if len(names) != count or (pair_names is not None and names != pair_names):
                raise ValueError('benchmark filename pairing differs for ' + dataset)
            pair_names = names
            for metric in ('psnr', 'ssim'):
                entry[label + '_' + metric] = sum(row[metric] for row in rows) / count
        entry['delta_psnr'] = entry['F1_psnr'] - entry['B0_psnr']
        entry['delta_ssim'] = entry['F1_ssim'] - entry['B0_ssim']
        summary.append(entry)
    with (group / 'benchmark_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ['dataset | B0 PSNR/SSIM | F1 PSNR/SSIM | delta PSNR/SSIM']
    for row in summary:
        lines.append('{dataset} | {B0_psnr:.6f}/{B0_ssim:.6f} | {F1_psnr:.6f}/{F1_ssim:.6f} | {delta_psnr:+.6f}/{delta_ssim:+.6f}'.format(**row))
    lines.append('\nSet5 per-image:')
    for label in ('B0', 'F1'):
        for row in results[label]:
            if row['dataset'] == 'Set5':
                lines.append('{} {} PSNR={:.6f} SSIM={:.6f}'.format(label, row['filename'], row['psnr'], row['ssim']))
    lines.append('\nSTOP AT EPOCH20. No cross-dataset average. Await user review before continuing.')
    output = '\n'.join(lines)
    (group / 'summary.txt').write_text(output, encoding='utf-8')
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'experiment/all_runs')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true', help='smoke/check only')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--resume', type=Path, help='resume an interrupted screening group, up to epoch20 only')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    if args.workers < 1:
        raise ValueError('Use workers >=1 for independent paired crop/augmentation streams')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('F1 screening is a single GPU, single training process protocol')
    check_data(args.data_root)
    subprocess.run([sys.executable, str(ROOT / 'repro/check_f1_identity_delta_esa.py'),
                    '--device', 'cpu' if args.cpu else 'cuda'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'repro/check_training_run_logging.py')], cwd=ROOT, check=True)
    if args.check_only:
        return
    if args.cpu and not args.smoke_only:
        raise ValueError('Full screening requires CUDA')
    if args.resume:
        group = args.resume.resolve()
        manifest = json.loads((group / 'protocol.json').read_text(encoding='utf-8'))
        if manifest['protocol'] != PROTOCOL or manifest['commit'] != git('rev-parse', 'HEAD'):
            raise ValueError('resume source commit or protocol changed')
        if manifest['workers'] != args.workers or manifest['data_root'] != str(args.data_root):
            raise ValueError('resume data root or worker count changed')
        if manifest['source_sha256'] != source_hashes():
            raise ValueError('resume source files changed despite same commit')
    else:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        group = args.output_root.resolve() / ('F1_pair_x4_seed1_' + stamp)
        group.mkdir(parents=True, exist_ok=False)
        write_json(group / 'protocol.json', dict(protocol=PROTOCOL, commit=git('rev-parse', 'HEAD'),
                   workers=args.workers, data_root=str(args.data_root), source_sha256=source_hashes()))
        print('RUN GROUP: ' + str(group), flush=True)
        smoke(args, group)
    if args.smoke_only:
        return
    for label, model in [('B0', 'LFMN'), ('F1', 'LFMNF1')]:
        directory = group / (label + '_x4_seed1')
        epoch = 0
        if directory.exists():
            rows = list(csv.DictReader((directory / 'metrics.csv').open(encoding='utf-8')))
            epoch = int(rows[-1]['epoch']) if rows else 0
            if not epoch:
                raise ValueError('run failed before first checkpoint; preserve it and start a new group')
        if epoch < 20:
            command = train_command(args, directory, model, resume_epoch=epoch)
            expected = None
            if label == 'F1' and not epoch:
                baseline_config = json.loads((group / 'B0_x4_seed1/config.json').read_text(encoding='utf-8'))
                expected = baseline_config['initial_shared_state_sha256']
            launch(command, directory, config(args, model, command, resume_start_epoch=epoch,
                   expected_initial_shared_state_sha256=expected,
                   resume_checkpoint=str(directory / 'model' / ('model_{}.pt'.format(epoch))) if epoch else None),
                   ROOT / 'LFMN', resume=bool(epoch))
    results = {}
    for label in ('B0', 'F1'):
        paths_file = group / 'evaluation_paths.json'
        paths = json.loads(paths_file.read_text(encoding='utf-8')) if paths_file.exists() else {}
        directory = Path(paths.get(label, str(group / (label + '_benchmarks_epoch20_OFF'))))
        if (directory / 'per_image_metrics/epoch_0000.pt').exists():
            import torch
            results[label] = torch.load(directory / 'per_image_metrics/epoch_0000.pt', weights_only=True)
        else:
            results[label] = evaluate(args, group, label)
    summarize(group, results)
    if not (group / 'resources.json').exists():
        directory = group / ('resource_audit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        command = [sys.executable, str(ROOT / 'repro/profile_f1_resources.py'), str(group)]
        launch(command, directory, {'kind': 'resource_audit', 'git_commit': git('rev-parse', 'HEAD')}, ROOT)


if __name__ == '__main__':
    main()
