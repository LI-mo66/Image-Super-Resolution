"""M1-only N9 screening with fail-closed reuse of an existing B0.

Run --prepare-only first to audit and re-evaluate B0 without training.
All outputs are new; the original B0 directory is read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from data.div2k import DIV2K
from model.lfmn import Net as Baseline
import utility
from summarize_n9_scratch import read_config, load_per_image, paired_arrays
from summarize_n9_scratch import bootstrap_mean_ci, texture_groups

EXPECTED = {
    'model': 'LFMN', 'data_train': "['DIV2K']", 'data_test': "['DIV2K']",
    'data_range': '1-800/801-900', 'scale': '[4]', 'patch_size': '256',
    'batch_size': '4', 'n_threads': '8', 'seed': '1', 'ext': 'img',
    'rgb_range': '255', 'no_augment': 'False', 'precision': 'single',
    'optimizer': 'ADAM', 'betas': '(0.9, 0.999)', 'epsilon': '1e-08',
    'weight_decay': '0', 'gclip': '0', 'lr': '0.0002',
    'scheduler': 'cosine', 'eta_min': '1e-06', 'epochs': '150',
    'test_every': '1000', 'loss': '1*L1', 'pre_train': '',
    'max_train_batches': '0', 'chop': 'False', 'self_ensemble': 'False',
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def curve(directory, name):
    value = torch.load(directory / name, map_location='cpu', weights_only=True)
    if value.ndim != 3 or tuple(value.shape[1:]) != (1, 1):
        raise ValueError(f'invalid curve shape: {directory / name}')
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite curve')
    return value[:, 0, 0].double().numpy()


def audit_baseline(directory):
    config = read_config(directory)
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise ValueError(f'B0 protocol mismatch: {key}: {config.get(key)!r} != {expected!r}')
    horizon = int(config.get('scheduler_t_max', '0')) or int(config['epochs'])
    if horizon != 150:
        raise ValueError('B0 cosine horizon must be 150')
    state = torch.load(directory / 'scheduler.pt', map_location='cpu', weights_only=True)
    for key, expected in {'T_max': 150, 'eta_min': 1e-6, 'last_epoch': 150,
                          'base_lrs': [2e-4]}.items():
        if state.get(key) != expected:
            raise ValueError(f'B0 scheduler mismatch: {key}')
    logs = {name: curve(directory, name + '_log.pt') for name in ('psnr', 'ssim')}
    if any(len(values) != 150 for values in logs.values()):
        raise ValueError('B0 requires 150 completed full-precision epochs')
    hashes = {}
    net = Baseline(scale=4)
    for epoch in range(16, 21):
        path = directory / 'model' / f'model_{epoch}.pt'
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        hashes[str(epoch)] = sha256(path)
    return config, logs, hashes


def dataset_args(data_root, validation='801-900'):
    return SimpleNamespace(data_range=f'1-800/{validation}', dir_data=str(data_root),
                           scale=[4], ext='img', batch_size=4, test_every=1000,
                           patch_size=256, n_colors=3, rgb_range=255,
                           no_augment=False, data_train=['DIV2K'], test_only=False)


def data_manifest(data_root):
    files = []
    for split, start, stop in [('train', 1, 800), ('valid', 801, 900)]:
        for index in range(start, stop + 1):
            files.extend([data_root / 'DIV2K' / f'DIV2K_{split}_HR' / f'{index:04d}.png',
                          data_root / 'DIV2K' / f'DIV2K_{split}_LR_bicubic' / 'X4' / f'{index:04d}x4.png'])
    records = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append({'path': path.relative_to(data_root).as_posix(), 'sha256': sha256(path)})
    return records


def reevaluate_baseline(baseline, output, data_root, logs):
    dataset = DIV2K(dataset_args(data_root), name='DIV2K', train=False)
    if len(dataset) != 100:
        raise ValueError('validation must contain exactly 100 images')
    proxy = SimpleNamespace(dataset=dataset)
    net = Baseline(scale=4).cuda().eval()
    metrics_dir = output / 'b0_eval' / 'per_image_metrics'
    metrics_dir.mkdir(parents=True)
    # Tail curves are already stored at full precision; only the fixed final
    # checkpoint needs fresh per-image inference (100 rather than 500 images).
    for epoch in (20,):
        net.load_state_dict(torch.load(baseline / 'model' / f'model_{epoch}.pt',
                                       map_location='cpu', weights_only=True), strict=True)
        rows = []
        with torch.inference_mode():
            for lr, hr, filename in dataset:
                sr = utility.quantize(net(lr.unsqueeze(0).cuda()), 255)
                hr = hr.unsqueeze(0).cuda()
                rows.append({'dataset': 'DIV2K', 'scale': 4, 'filename': filename,
                             'psnr': float(utility.calc_psnr(sr, hr, 4, 255, proxy)),
                             'ssim': float(utility.calc_ssim(sr, hr, 4, 255, proxy))})
        for metric, tolerance in [('psnr', 2e-4), ('ssim', 1e-5)]:
            average = float(np.mean([row[metric] for row in rows]))
            delta = average - float(logs[metric][epoch - 1])
            print(f'B0 e{epoch} {metric}: {average:.8f}; replay delta {delta:+.8f}', flush=True)
            if not np.isfinite(average) or abs(delta) > tolerance:
                raise ValueError(f'B0 replay failed for {metric}; M1 training blocked')
        torch.save(rows, metrics_dir / f'epoch_{epoch:04d}.pt')
    del net
    torch.cuda.empty_cache()


def training_command(data_root, output):
    return [sys.executable, 'main.py', '--dir_data', str(data_root),
            '--model', 'LFMNPCSTR', '--data_train', 'DIV2K', '--data_test', 'DIV2K',
            '--data_range', '1-800/801-900', '--scale', '4', '--patch_size', '256',
            '--batch_size', '4', '--n_threads', '8', '--ext', 'img', '--epochs', '20',
            '--test_every', '1000', '--lr', '2e-4', '--scheduler', 'cosine',
            '--scheduler_t_max', '150', '--eta_min', '1e-6', '--loss', '1*L1',
            '--seed', '1', '--save_per_image_metrics', '--save_models',
            '--save', str(output / 'm1')]


def summarize(output, baseline, data_root):
    b = {metric: curve(baseline, metric + '_log.pt')[:20] for metric in ('psnr', 'ssim')}
    m = {metric: curve(output / 'm1', metric + '_log.pt') for metric in ('psnr', 'ssim')}
    if any(len(value) != 20 for value in m.values()):
        raise ValueError('M1 requires exactly 20 completed epochs')
    # Validate identity and protocol before subtracting curves.
    config = read_config(output / 'm1')
    expected = {**EXPECTED, 'model': 'LFMNPCSTR', 'epochs': '20', 'scheduler_t_max': '150'}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'M1 protocol mismatch: {key}')
    for epoch in range(16, 21):
        m_rows = load_per_image(output / 'm1', epoch)
        for metric, column, tolerance in [('psnr', 0, 2e-4), ('ssim', 1, 1e-5)]:
            for rows, values in [(m_rows, m)]:
                average = float(np.mean([row[column] for row in rows.values()]))
                if not np.isfinite(average) or abs(average - values[metric][epoch - 1]) > tolerance:
                    raise ValueError(f'per-image/curve mismatch: epoch {epoch} {metric}')
    for metric, column, tolerance in [('psnr', 0, 2e-4), ('ssim', 1, 1e-5)]:
        rows = load_per_image(output / 'b0_eval', 20)
        if abs(np.mean([row[column] for row in rows.values()]) - b[metric][19]) > tolerance:
            raise ValueError('B0 final replay curve mismatch')
    a = load_per_image(output / 'b0_eval', 20)
    c = load_per_image(output / 'm1', 20)
    keys, delta, ssim = paired_arrays(a, c)
    if not np.isfinite(delta).all() or not np.isfinite(ssim).all():
        raise ValueError('nonfinite paired metrics')
    # Check per-image metrics correspond to the same final curve.
    if abs(np.mean([v[0] for v in c.values()]) - m['psnr'][-1]) > 2e-4:
        raise ValueError('M1 per-image metrics do not match final curve')
    d = m['psnr'] - b['psnr']
    tail, slope = float(d[-5:].mean()), float(np.polyfit(np.arange(5), d[-5:], 1)[0])
    final_ssim = float(m['ssim'][-1] - b['ssim'][-1])
    median = float(np.median(delta))
    decision = decide(tail, slope, median, final_ssim, int((d[-5:] > 0).sum()))
    result = {'decision': decision, 'epoch_deltas': d.tolist(), 'final_delta': float(d[-1]),
              'last5_delta': tail, 'slope_delta': slope, 'final_ssim_delta': final_ssim,
              'paired_mean': float(delta.mean()), 'paired_median': median,
              'win_rate': float((delta > 0).mean()), 'bootstrap95': bootstrap_mean_ci(delta),
              'texture_deltas': {name: float(delta[index].mean())
                                 for name, index in texture_groups(keys, data_root).items()},
              'limitation': 'Accuracy screen only; promotion requires trained routing and server efficiency audit.'}
    diagnosis_path = output / 'diagnosis.json'
    if diagnosis_path.is_file():
        diagnosis = json.loads(diagnosis_path.read_text(encoding='utf-8'))
        result['mechanism_efficiency'] = diagnosis
        if not diagnosis['routing_pass'] or not diagnosis['efficiency_pass']:
            result['decision'] = 'STOP: routing or efficiency gate'
    (output / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return result


def decide(tail, slope, median, ssim, positive_tail):
    if not np.isfinite([tail, slope, median, ssim]).all():
        raise ValueError('nonfinite decision inputs')
    if (tail <= 0 and slope <= 0) or ssim < -0.0001:
        return 'STOP'
    if tail >= 0.01 and median > 0 and slope >= -0.001:
        return 'EXTEND_ELIGIBLE_40: manual mechanism/efficiency audit required'
    if 0 < tail < 0.01 and slope >= 0.001 and positive_tail >= 4:
        return 'GRAY_ELIGIBLE_40_ONCE: manual audit required'
    return 'NO_AUTOMATIC_EXTENSION'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', action='store_true', help='use previously audited output')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    baseline, data_root, output = (p.resolve() for p in (args.baseline, args.data_root, args.output))
    if output == baseline or output in baseline.parents or baseline in output.parents:
        raise ValueError('B0 and new output must be disjoint')
    if args.summarize_only:
        audit_baseline(baseline)
        summarize(output, baseline, data_root)
        return
    if args.prepared and args.prepare_only:
        raise ValueError('--prepared and --prepare-only are mutually exclusive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; use a fresh server with physical GPU 0')
    if (os.cpu_count() or 1) < 16:
        raise RuntimeError('At least 16 CPU cores required to retain baseline n_threads=8')
    if os.environ.get('CUDA_VISIBLE_DEVICES', '0') != '0':
        raise ValueError('Existing main.py assumes physical GPU 0; set CUDA_VISIBLE_DEVICES=0')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError('tracked code is dirty; commit before server execution')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    config, logs, hashes = audit_baseline(baseline)
    artifact_hashes = {name: sha256(baseline / name) for name in
                       ('config.txt', 'scheduler.pt', 'psnr_log.pt', 'ssim_log.pt')}
    if args.prepared:
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if any(manifest[key] != value for key, value in
               [('commit', commit), ('baseline', str(baseline)), ('data_root', str(data_root)),
                ('baseline_checkpoint_sha256', hashes), ('baseline_artifact_sha256', artifact_hashes)]):
            raise ValueError('prepared audit provenance changed')
        if manifest.get('status') != 'PREPARED':
            raise ValueError('output is not a completed preparation')
        print('Rechecking data hashes before training', flush=True)
        if manifest['data_files'] != data_manifest(data_root):
            raise ValueError('data changed since preparation')
        replay = load_per_image(output / 'b0_eval', 20)
        for metric, column, tolerance in [('psnr', 0, 2e-4), ('ssim', 1, 1e-5)]:
            average = float(np.mean([row[column] for row in replay.values()]))
            if not np.isfinite(average) or abs(average - logs[metric][19]) > tolerance:
                raise ValueError('prepared B0 metrics changed or do not match')
    else:
        if output.exists():
            raise FileExistsError(f'output exists: {output}; use a new path')
        output.mkdir(parents=True)
        print('Recording hashes of all 900 HR/LR pairs', flush=True)
        manifest = {'commit': commit, 'baseline': str(baseline), 'data_root': str(data_root),
                    'baseline_config': config, 'baseline_checkpoint_sha256': hashes,
                    'baseline_artifact_sha256': artifact_hashes,
                    'baseline_source_commit': 'UNRECORDED; bf68bfe inferred from timestamp/script',
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0), 'data_files': data_manifest(data_root),
                    'command': training_command(data_root, output), 'status': 'PREPARING'}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        subprocess.run([sys.executable, str(ROOT / 'repro/check_n9_m1.py'),
                        '--data-root', str(data_root), '--baseline', str(baseline),
                        '--output', str(output / 'smoke')], check=True)
        reevaluate_baseline(baseline, output, data_root, logs)
        manifest['status'] = 'PREPARED'
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.prepare_only:
        print('Prepared: no screening training started', flush=True)
        return
    if (output / 'm1').exists():
        raise FileExistsError('M1 output exists; never overwrite or silently resume')
    manifest['status'] = 'TRAINING'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (output / 'm1').mkdir()
    env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': '0'}
    with (output / 'm1' / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(training_command(data_root, output), cwd=ROOT / 'LFMN',
                                   env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    manifest['exit_code'] = code
    manifest['status'] = 'TRAINING_FAILED' if code else 'TRAINING_COMPLETE'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if code:
        raise subprocess.CalledProcessError(code, process.args)
    from diagnose_n9_m1 import diagnose
    diagnose(baseline / 'model/model_20.pt', output / 'm1/model/model_20.pt',
             data_root, output / 'diagnosis.json')
    summarize(output, baseline, data_root)
    manifest['status'] = 'SCREEN_COMPLETE'
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
