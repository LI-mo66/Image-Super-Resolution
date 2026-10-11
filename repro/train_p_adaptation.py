"""Matched P0/P3/P4 adaptation. Actual model/data execution is server-only."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as Baseline
from run_p_diagnostics import measure, masks_for, read_pair, sha
from p_adapt_logging import RunLogger

EXPECTED_CHECKPOINT = '44999471d8cc2d5f7dbf10d354766e08a5d23a84200a1060dfa9a9ed7a7711dd'


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def make_net(scheme, checkpoint, device):
    if sha(checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Official x4 checkpoint SHA mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if scheme == 'P4':
        from model.lfmnp4 import Net
        net = Net(scale=4)
        net.load_baseline(state)
    else:
        net = Baseline(scale=4)
        net.load_state_dict(state, strict=True)
    return net.to(device)


def training_plan(data_root, steps, batch_size, patch, seed):
    """Independent RNG; creation of a candidate never changes the data plan."""
    root = Path(data_root) / 'DIV2K'
    dimensions = {}
    for i in range(1, 801):
        lr = root / 'DIV2K_train_LR_bicubic/X4' / f'{i:04d}x4.png'
        hr = root / 'DIV2K_train_HR' / f'{i:04d}.png'
        if not lr.is_file() or not hr.is_file():
            raise FileNotFoundError(f'Missing training pair {i}: {lr} / {hr}')
        with Image.open(lr) as im:
            w, h = im.size
        with Image.open(hr) as im:
            hw, hh = im.size
        if min(h, w) < patch or hh < h * 4 or hw < w * 4:
            raise ValueError(f'Invalid training pair dimensions for {i}')
        dimensions[i] = (h, w)
    rng = random.Random(seed)
    plan = []
    for _ in range(steps):
        batch = []
        for _ in range(batch_size):
            i = rng.randrange(1, 801)
            h, w = dimensions[i]
            batch.append([i, rng.randrange(h - patch + 1), rng.randrange(w - patch + 1),
                          rng.randrange(2), rng.randrange(2), rng.randrange(2)])
        plan.append(batch)
    plan_hash = hashlib.sha256(json.dumps(plan, separators=(',', ':')).encode()).hexdigest()
    return plan, plan_hash


def batch_from_plan(data_root, items, patch):
    root = Path(data_root) / 'DIV2K'
    batches = [[], []]
    for i, top, left, flip_h, flip_v, transpose in items:
        paths = [root / 'DIV2K_train_LR_bicubic/X4' / f'{i:04d}x4.png',
                 root / 'DIV2K_train_HR' / f'{i:04d}.png']
        for group, (path, scale) in enumerate(zip(paths, (1, 4))):
            with Image.open(path) as im:
                crop = im.convert('RGB').crop((left * scale, top * scale,
                       (left + patch) * scale, (top + patch) * scale))
                image = np.array(crop, copy=True)
            if flip_h:
                image = image[:, ::-1]
            if flip_v:
                image = image[::-1]
            if transpose:
                image = image.transpose(1, 0, 2)
            batches[group].append(torch.from_numpy(image.copy()).permute(2, 0, 1).float())
    lr, hr = (torch.stack(xs) for xs in batches)
    digest = hashlib.sha256(lr.numpy().tobytes() + hr.numpy().tobytes()).hexdigest()
    return lr, hr, digest


def criterion(scheme, sr, hr):
    error = (sr - hr) / 255
    return error.square().mean() if scheme == 'P3' else error.abs().mean()


def evaluate(net, data_root, device, epoch, step, benchmark=False, limit=None):
    net.eval()
    root = Path(data_root)
    pairs = []
    if benchmark:
        for ds in ('Set5', 'Set14', 'B100', 'Urban100', 'manga109'):
            folder = root / 'benchmark' / ds
            if ds == 'manga109' and not folder.exists():
                folder = root / 'benchmark/Manga109'
            files = sorted((folder / 'HR').glob('*.png'))
            expected = {'Set5': 5, 'Set14': 14, 'B100': 100, 'Urban100': 100, 'manga109': 109}[ds]
            if len(files) != expected:
                raise ValueError(f'{ds}: expected {expected} HR files, got {len(files)}')
            pairs.extend((ds, hp.stem, hp, folder / 'LR_bicubic/X4' / (hp.stem + 'x4.png')) for hp in files)
    else:
        pairs = [('DIV2K_valid', f'{i:04d}', root / 'DIV2K/DIV2K_valid_HR' / f'{i:04d}.png',
                  root / 'DIV2K/DIV2K_valid_LR_bicubic/X4' / f'{i:04d}x4.png') for i in range(811, 831)]
    if limit:
        pairs = pairs[:limit]
    rows = []
    with torch.inference_mode():
        for ds, name, hp, lp in pairs:
            lr, hr, alignment = read_pair(hp, lp, 4, 0 if benchmark else 96)
            lr, hr = lr.to(device), hr.to(device)
            sr = net(lr)
            metrics = measure(sr, hr, 4, masks_for(lr))
            rows.append({'epoch': epoch, 'global_step': step, 'dataset': ds, 'image': name,
                         'split': 'benchmark_final' if benchmark else 'validation',
                         'psnr': metrics['psnr_y_quantized'], 'ssim': metrics['ssim_y_quantized'],
                         'lr_hash': sha(lp), 'hr_hash': sha(hp), 'alignment': alignment,
                         'regions': metrics['regions']})
    return rows


def save_checkpoint(path, net, optimizer, step, epoch, config, best):
    state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
             'scheduler': None, 'global_step': step, 'epoch': epoch,
             'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
             'scheme': config['experiment_name'], 'protocol_fingerprint': config['protocol_fingerprint'],
             'batch_plan_sha256': config['batch_plan_sha256'], 'best': best}
    temp = Path(str(path) + '.tmp')
    torch.save(state, temp)
    temp.replace(path)


def configuration(args, plan_hash):
    metric = {'psnr': 'RGB255 clamp-round; Y difference [65.738,129.057,25.064]/256; shave4',
              'ssim': 'BT601 [65.481,128.553,24.966]; shave4; Gaussian11 sigma1.5'}
    fingerprint = {'initial_checkpoint_sha256': sha(args.checkpoint),
                   'model_sha256': sha(ROOT / 'LFMN/model/lfmn.py'),
                   'runner_sha256': sha(__file__), 'metric_source_sha256': sha(ROOT / 'LFMN/utility.py'),
                   'scale': 4, 'steps': args.steps, 'steps_per_epoch': args.steps_per_epoch,
                   'batch_size': args.batch_size, 'patch_lr': args.patch_lr, 'patch_hr': args.patch_lr * 4,
                   'seed': args.seed, 'train_range': [1, 800], 'validation_ids': list(range(811, 831)),
                   'validation_crop_lr': 96, 'optimizer': 'Adam', 'betas': [.9, .999],
                   'learning_rate': args.lr, 'scheduler': 'constant', 'precision': 'FP32',
                   'self_ensemble': False, 'chop': False, 'normalize_overlap': False, 'eval_refine_iters': 0,
                   'metric_protocol': metric, 'batch_plan_sha256': plan_hash,
                   'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(torch.device(args.device)),
                   'augmentation': 'independent plan: hflip/vflip/transpose, paired exact integer crops'}
    return {'scheme': args.scheme, 'experiment_name': args.scheme, 'model': 'LFMN_P4' if args.scheme == 'P4' else 'LFMN',
            'model_source': str(ROOT / ('LFMN/model/lfmnp4.py' if args.scheme == 'P4' else 'LFMN/model/lfmn.py')),
            'command': sys.argv, 'cwd': str(ROOT), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'git_diff_summary': subprocess.check_output(['git', 'diff', '--stat'], cwd=ROOT, text=True).strip(),
            'dataset': str(Path(args.data_root).resolve()), 'train_range': [1, 800], 'validation_range': [811, 830],
            'scale': 4, 'degradation': 'original DIV2K bicubic X4, no regeneration',
            'epochs': args.steps // args.steps_per_epoch, 'steps_per_epoch': args.steps_per_epoch,
            'batch_size': args.batch_size, 'patch_size': args.patch_lr * 4, 'optimizer': 'Adam',
            'learning_rate': args.lr, 'scheduler': 'constant', 'loss': 'RGB MSE/255^2' if args.scheme == 'P3' else 'RGB L1/255',
            'seed': args.seed, 'device': args.device, 'gpu_name': fingerprint['gpu'], 'world_size': 1,
            'ema_enabled': False, 'tab_prototype_ema': True, 'self_ensemble_enabled': False,
            'pretrained_checkpoint': str(Path(args.checkpoint).resolve()), 'baseline_training_config': 'unknown',
            'initial_checkpoint_sha256': sha(args.checkpoint), 'batch_plan_sha256': plan_hash,
            'metric_protocol': metric, 'protocol_fingerprint': fingerprint,
            'candidate_source_sha256': sha(ROOT / 'LFMN/model/lfmnp4.py') if args.scheme == 'P4' else None,
            'resume_checkpoint': str(Path(args.resume).resolve()) if args.resume else None,
            'resume_start_epoch': None, 'resume_start_step': None, 'checks_report': args.checks_report,
            'benchmark_final': args.benchmark_final}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheme', choices=('P0', 'P3', 'P4'), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', default=str(ROOT / 'LFMN/model/scale4_model_939.pt'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--steps-per-epoch', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--patch-lr', type=int, default=48)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--checks-report', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--benchmark-final', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('Actual adaptation is server-only; local code/math checks only')
    if (args.steps, args.steps_per_epoch, args.batch_size, args.patch_lr, args.seed, args.lr) != (2000, 100, 4, 48, 1, 1e-5):
        parser.error('Budget/protocol is preregistered; changes require a new experiment card')
    report = json.loads(Path(args.checks_report).read_text(encoding='utf-8'))
    if report.get('status') != 'verified' or report.get('scheme') != args.scheme:
        raise ValueError('Matching successful server engineering gate is required')
    if report.get('checkpoint_sha256') != sha(args.checkpoint) or report.get('model_sha256') != sha(ROOT / 'LFMN/model/lfmn.py'):
        raise ValueError('Engineering report model/checkpoint mismatch')
    if report.get('runner_sha256') != sha(__file__) or report.get('data_root') != str(Path(args.data_root).resolve()):
        raise ValueError('Engineering report runner/data mismatch')
    if args.scheme == 'P4' and report.get('candidate_source_sha256') != sha(ROOT / 'LFMN/model/lfmnp4.py'):
        raise ValueError('Engineering report candidate source mismatch')
    torch.set_num_threads(4)
    plan, plan_hash = training_plan(args.data_root, args.steps, args.batch_size, args.patch_lr, args.seed)
    config = configuration(args, plan_hash)
    if config['git_dirty']:
        raise ValueError('Refuse training on dirty candidate code')
    net = make_net(args.scheme, args.checkpoint, args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(.9, .999))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    start = 0
    best = {'psnr': None, 'ssim': None, 'epoch': None}
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        if state['scheme'] != args.scheme or state['protocol_fingerprint'] != config['protocol_fingerprint']:
            raise ValueError('Resume scheme/protocol mismatch')
        net.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['global_step'], state['best']
        if start % args.steps_per_epoch or start >= args.steps:
            raise ValueError('Resume only from a completed record epoch below total budget')
        config['resume_start_epoch'], config['resume_start_step'] = state['epoch'] + 1, start
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
    directory = Path(args.output_dir)
    with RunLogger(directory.parent, args.scheme, 4, args.seed, config,
                   resume_dir=directory if args.resume else None,
                   output_directory=None if args.resume else directory) as log:
        directory = log.directory
        checkpoint_dir = directory / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        print(f'SCHEME {args.scheme} commit={config["git_commit"]} run={directory} protocol={config["protocol_fingerprint"]}', flush=True)
        atomic_json(directory / 'batch_plan.json', plan)
        rows = json.loads((directory / 'per_image.json').read_text(encoding='utf-8')) if args.resume else []
        if not args.resume:
            initial = evaluate(net, args.data_root, args.device, 0, 0)
            rows.extend(initial)
            atomic_json(directory / 'per_image.json', rows)
            base_psnr = float(np.mean([r['psnr'] for r in initial]))
            base_ssim = float(np.mean([r['ssim'] for r in initial]))
            log.record({'epoch': 0, 'global_step': 0, 'learning_rate': args.lr, 'train_loss': None,
                        'validation_psnr': base_psnr, 'validation_ssim': base_ssim,
                        'is_best': False, 'elapsed_seconds': 0})
            log.update(initial_metrics={'psnr': base_psnr, 'ssim': base_ssim})
            save_checkpoint(checkpoint_dir / 'last.pt', net, optimizer, 0, 0, config, best)
        stream = hashlib.sha256()
        hash_path = directory / 'batch_hashes.csv'
        batch_rows = []
        if not args.resume:
            with hash_path.open('x', newline='', encoding='utf-8') as empty:
                csv.DictWriter(empty, fieldnames=['global_step', 'sha256']).writeheader()
        if args.resume:
            prior = list(csv.DictReader(hash_path.open(encoding='utf-8')))
            if len(prior) != start or [int(r['global_step']) for r in prior] != list(range(1, start + 1)):
                raise ValueError('Resume data hash timeline inconsistent; retain failed run for audit')
            for r in prior:
                stream.update(r['sha256'].encode())
            batch_rows = prior
        started = time.perf_counter()
        epoch_losses = []
        # Keep partial-session evidence; canonical hashes only describe completed epochs.
        session_hash_path = directory / f'batch_hashes.session{log.session}.csv'
        with session_hash_path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['global_step', 'sha256'])
            writer.writeheader()
            for step in range(start + 1, args.steps + 1):
                lr, hr, batch_hash = batch_from_plan(args.data_root, plan[step - 1], args.patch_lr)
                writer.writerow({'global_step': step, 'sha256': batch_hash})
                handle.flush()
                batch_rows.append({'global_step': step, 'sha256': batch_hash})
                stream.update(batch_hash.encode())
                net.train()
                optimizer.zero_grad(set_to_none=True)
                sr = net(lr.to(args.device))
                loss = criterion(args.scheme, sr, hr.to(args.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise FloatingPointError(f'Nonfinite gradient at step {step}')
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
                if step % 20 == 0:
                    print(f'Step {step}/{args.steps} loss={float(loss):.10f} lr={args.lr:.8g}', flush=True)
                if step % args.steps_per_epoch == 0:
                    epoch = step // args.steps_per_epoch
                    validation = evaluate(net, args.data_root, args.device, epoch, step)
                    rows.extend(validation)
                    atomic_json(directory / 'per_image.json', rows)
                    psnr, ssim = (float(np.mean([r[key] for r in validation])) for key in ('psnr', 'ssim'))
                    is_best = best['psnr'] is None or psnr > best['psnr']
                    if is_best:
                        best = {'psnr': psnr, 'ssim': ssim, 'epoch': epoch}
                    save_checkpoint(checkpoint_dir / 'last.pt', net, optimizer, step, epoch, config, best)
                    temporary_hashes = Path(str(hash_path) + '.tmp')
                    with temporary_hashes.open('w', newline='', encoding='utf-8') as committed:
                        canonical = csv.DictWriter(committed, fieldnames=['global_step', 'sha256'])
                        canonical.writeheader()
                        canonical.writerows(batch_rows)
                    temporary_hashes.replace(hash_path)
                    log.record({'epoch': epoch, 'global_step': step, 'learning_rate': args.lr,
                                'train_loss': float(np.mean(epoch_losses)), 'validation_psnr': psnr,
                                'validation_ssim': ssim, 'is_best': is_best, 'elapsed_seconds': time.perf_counter() - started})
                    log.update(final_metrics={'psnr': psnr, 'ssim': ssim}, best_metrics=best,
                               best_epoch=best['epoch'], last_completed_epoch=epoch,
                               last_completed_step=step, data_stream_sha256=stream.hexdigest())
                    print(f'Epoch {epoch} PSNR={psnr:.9f} SSIM={ssim:.9f} checkpoint=last.pt', flush=True)
                    epoch_losses.clear()
        if args.benchmark_final:
            rows.extend(evaluate(net, args.data_root, args.device, args.steps // args.steps_per_epoch, args.steps, benchmark=True))
            atomic_json(directory / 'per_image.json', rows)
        log.update(final_checkpoint=str(checkpoint_dir / 'last.pt'),
                   final_checkpoint_sha256=sha(checkpoint_dir / 'last.pt'),
                   data_stream_sha256=stream.hexdigest())


if __name__ == '__main__':
    main()
