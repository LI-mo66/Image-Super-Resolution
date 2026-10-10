"""P0 fixed-checkpoint evaluation; optional isolated P1/P2 observation hooks.

No training, no optimizer, no candidate model. Full benchmark scores require
--benchmarks full --crop-lr 0; default crops are explicitly diagnostic only.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net
import utility


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def load_plugin(path):
    if not path:
        return None
    spec = importlib.util.spec_from_file_location('p_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def masks_for(lr):
    """Fixed LR-only strata, not a classifier for periodic textures."""
    gray = (lr.detach().cpu() / 255 * torch.tensor(
        [0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)).sum(1, keepdim=True)
    sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sx[None, None])
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sx.T[None, None])
    smooth = lambda x: F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='reflect'), 5, 1)
    jxx, jyy, jxy = smooth(gx ** 2), smooth(gy ** 2), smooth(gx * gy)
    energy = jxx + jyy
    coherence = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (energy + 1e-12)
    active = energy[0, 0] >= 1e-4
    c = coherence[0, 0]
    return {'flat': ~active, 'edge': active & (c >= .7),
            'mixed': active & (c < .3), 'other': active & (c >= .3) & (c < .7)}


def measure(sr, hr, scale, masks):
    if sr.shape != hr.shape or not torch.isfinite(sr).all():
        raise ValueError('Nonfinite output or SR/HR alignment mismatch')
    q = utility.quantize(sr, 255)
    benchmark = SimpleNamespace(dataset=SimpleNamespace(benchmark=True))
    standard = utility.calc_psnr(q, hr, scale, 255, benchmark)
    if not math.isfinite(standard):
        raise ValueError('Nonfinite PSNR')
    coeff = sr.new_tensor([65.738, 129.057, 25.064]).reshape(1, 3, 1, 1) / 256
    e = ((sr - hr) * coeff).sum(1)[0] ** 2
    eq = ((q - hr) * coeff).sum(1)[0] ** 2
    valid = torch.zeros_like(e, dtype=torch.bool)
    valid[scale:-scale, scale:-scale] = True
    regions = {}
    for name, mask in masks.items():
        up = mask.repeat_interleave(scale, 0).repeat_interleave(scale, 1).to(sr.device)
        selected = up & valid
        n = int(selected.sum())
        regions[name] = {'pixels': n, 'y_mse_continuous': float(e[selected].mean()) if n else None,
                         'y_mse_quantized': float(eq[selected].mean()) if n else None}
    return {'psnr_y_quantized': standard,
            'ssim_y_quantized': float(utility.calc_ssim(q, hr, scale, 255, benchmark)),
            'mse_y_continuous': float(e[valid].mean()),
            'mse_rgb_continuous': float(((sr - hr) ** 2)[..., scale:-scale, scale:-scale].mean()),
            'regions': regions}


def read_pair(hr_path, lr_path, scale, crop):
    def read(path):
        return torch.from_numpy(np.array(Image.open(path).convert('RGB'), copy=True)).permute(2, 0, 1).float()[None]
    lr, hr = read(lr_path), read(hr_path)
    h, w = lr.shape[-2:]
    if hr.shape[-2] < h * scale or hr.shape[-1] < w * scale:
        raise ValueError(f'HR is smaller than LR footprint: {hr_path}')
    original = list(hr.shape[-2:])
    hr = hr[..., :h * scale, :w * scale]
    coords = [0, 0, h, w]
    if crop:
        ch, cw = min(crop, h), min(crop, w)
        top, left = (h - ch) // 2, (w - cw) // 2
        lr = lr[..., top:top + ch, left:left + cw]
        hr = hr[..., top * scale:(top + ch) * scale, left * scale:(left + cw) * scale]
        coords = [top, left, ch, cw]
    if min(lr.shape[-2:]) < 28:
        raise ValueError('LRSA needs sufficiently large input; no silent resize')
    return lr, hr, {'hr_original_hw': original, 'lr_crop_tlhw': coords,
                    'hr_modcrop_hw': [h * scale, w * scale]}


def pairs(args):
    root = Path(args.data_root)
    result = []
    if args.div2k_ids:
        for image_id in args.div2k_ids:
            stem = f'{image_id:04d}'
            result.append(('DIV2K_valid', stem, root / 'DIV2K/DIV2K_valid_HR' / (stem + '.png'),
                           root / f'DIV2K/DIV2K_valid_LR_bicubic/X{args.scale}' / (stem + f'x{args.scale}.png'),
                           'selection' if image_id <= 804 else 'inspection'))
    if args.benchmarks != 'none':
        for dataset in ('Set5', 'Set14', 'B100', 'Urban100', 'manga109'):
            files = sorted((root / 'benchmark' / dataset / 'HR').glob('*.png'))
            if not files:
                raise FileNotFoundError(dataset)
            if args.benchmarks == 'smoke':
                files = files[:1]
            for hr in files:
                lr = root / 'benchmark' / dataset / f'LR_bicubic/X{args.scale}' / (hr.stem + f'x{args.scale}.png')
                result.append((dataset, hr.stem, hr, lr, 'benchmark_smoke' if args.benchmarks == 'smoke' else 'benchmark_full'))
    if not result:
        raise ValueError('No samples selected')
    for _, _, hr, lr, _ in result:
        if not hr.is_file() or not lr.is_file():
            raise FileNotFoundError(f'{hr} / {lr}')
    return result


def state_hash(net):
    digest = hashlib.sha256()
    for key, value in net.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def check(net, plugin, device):
    before = state_hash(net)
    for h, w in ((32, 32), (33, 35), (35, 33)):
        x = torch.linspace(0, 255, 3 * h * w, device=device).reshape(1, 3, h, w)
        with torch.inference_mode():
            plain = net(x)
            if plugin:
                session = plugin.install(net, x, masks_for(x), 'p0')
                try:
                    observed = net(x)
                    if not torch.equal(plain, observed):
                        raise AssertionError('Observation/zero intervention changes output')
                    session.finish(observed, plain, {})
                finally:
                    session.close()
                restored = net(x)
                if not torch.equal(plain, restored):
                    raise AssertionError('Hooks not removed')
            if not torch.isfinite(plain).all() or list(plain.shape[-2:]) != [h * net.scale, w * net.scale]:
                raise AssertionError('Shape/numeric check failed')
    if before != state_hash(net):
        raise AssertionError('Parameters/buffers changed')
    if plugin and hasattr(plugin, 'self_check'):
        plugin.self_check()
    return {'strict_load': True, 'odd_rectangular_shapes': True, 'finite': True,
            'observation_restores_p0': True, 'state_unchanged': True,
            'training': False}


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='E:/fuxian_LFMN_jianghe')
    parser.add_argument('--data-root', default='E:/fuxian_LFMN_jianghe/datasets')
    parser.add_argument('--scale', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--checkpoint')
    parser.add_argument('--plugin')
    parser.add_argument('--scheme', choices=('P0', 'P1', 'P2'), default='P0')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--crop-lr', type=int, default=64)
    parser.add_argument('--benchmarks', choices=('none', 'smoke', 'full'), default='smoke')
    parser.add_argument('--div2k-ids', type=int, nargs='*', default=list(range(801, 809)))
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (args.scheme == 'P0') != (args.plugin is None):
        parser.error('P0 has no plugin; P1/P2 require an explicit isolated plugin')
    if args.crop_lr < 0 or (args.crop_lr and args.crop_lr < 28):
        parser.error('crop must be zero (full image) or >=28')
    torch.set_num_threads(4)
    torch.manual_seed(1)
    project = Path(args.project_root)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = project / 'experiment/all_runs' / f'{args.scheme}_diagnostic_x{args.scale}_seed1_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    original_out, original_err = sys.stdout, sys.stderr
    log = (out / 'diagnostic_log.txt').open('w', encoding='utf-8')
    sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
    config = vars(args).copy()
    config.update({'status': 'running', 'run_id': out.name, 'output_directory': str(out),
                   'seed': 1, 'baseline_training_seed': 'unknown', 'training': False,
                   'baseline_source_commit': '1e51b2068b62a793860e7b7192a43877d9ccc2f1',
                   'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                   'model_source': str(ROOT / 'LFMN/model/lfmn.py'),
                   'model_sha256': sha(ROOT / 'LFMN/model/lfmn.py'),
                   'runner_sha256': sha(__file__), 'metric_source_sha256': sha(ROOT / 'LFMN/utility.py'),
                   'plugin_sha256': sha(args.plugin) if args.plugin else None,
                   'torch': torch.__version__, 'self_ensemble': False, 'chop': False,
                   'normalize_overlap': False, 'eval_refine_iters': 0, 'precision': 'FP32',
                   'metric_protocol': {'psnr': 'RGB255 clamp-round; Y difference [65.738,129.057,25.064]/256; shave scale',
                                       'ssim': 'RGB255 clamp-round; BT601 [65.481,128.553,24.966]; shave scale; Gaussian11 sigma1.5'},
                   'region_protocol': 'LR normalized Sobel/8; 5x5 mean tensor; energy>=1e-4; coherence edge>=.7 mixed<.3; other otherwise; no periodic classifier',
                   'scope': 'full benchmark' if args.benchmarks == 'full' and not args.crop_lr else 'diagnostic crops; not full benchmark scores'})
    started = time.perf_counter()
    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else project / 'LFMN/model' / {2: 'scale2_model_996.pt', 3: 'scale3_model_969.pt', 4: 'scale4_model_939.pt'}[args.scale]
        config['checkpoint'] = str(checkpoint.resolve())
        config['checkpoint_sha256'] = sha(checkpoint)
        config['gpu'] = torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None
        write_json(out / 'config.json', config)
        print(f'Output: {out}', flush=True)
        net = Net(scale=args.scale).to(args.device).eval()
        net.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True), strict=True)
        plugin = load_plugin(args.plugin)
        variants = plugin.VARIANTS if plugin else ['p0']
        config['parameter_count'] = sum(p.numel() for p in net.parameters())
        config['engineering_checks'] = check(net, plugin, args.device)
        write_json(out / 'checks.json', config['engineering_checks'])
        before = state_hash(net)
        rows = []
        if not args.check_only:
            with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['dataset', 'image', 'split', 'variant', 'psnr_y_quantized', 'ssim_y_quantized', 'mse_y_continuous', 'mse_rgb_continuous', 'elapsed_seconds'])
                writer.writeheader()
                for dataset, name, hp, lp, split in pairs(args):
                    lr, hr, alignment = read_pair(hp, lp, args.scale, args.crop_lr)
                    lr, hr = lr.to(args.device), hr.to(args.device)
                    masks = masks_for(lr)
                    for variant in variants:
                        session = plugin.install(net, lr, masks, variant) if plugin else None
                        t = time.perf_counter()
                        try:
                            with torch.inference_mode():
                                sr = net(lr)
                            values = measure(sr, hr, args.scale, masks)
                            probe = session.finish(sr, hr, values) if session else {}
                        finally:
                            if session:
                                session.close()
                        row = {'dataset': dataset, 'image': name, 'split': split, 'variant': variant,
                               'elapsed_seconds': time.perf_counter() - t, **values,
                               'alignment': alignment, 'hr_path': str(hp), 'lr_path': str(lp),
                               'hr_sha256': sha(hp), 'lr_sha256': sha(lp), 'probe': probe}
                        rows.append(row)
                        write_json(out / 'per_image.json', rows)
                        writer.writerow({k: row[k] for k in writer.fieldnames})
                        csv_file.flush()
                        print(f'{dataset}/{name} {variant} PSNR={values["psnr_y_quantized"]:.9f} time={row["elapsed_seconds"]:.2f}s', flush=True)
        if before != state_hash(net):
            raise AssertionError('Diagnostic changed model state')
        config['status'] = 'completed'
        config['sample_variants_completed'] = len(rows)
        config['exit_code'] = 0
    except BaseException:
        config['status'] = 'failed'
        config['exit_code'] = 1
        traceback.print_exc()
        raise
    finally:
        config['elapsed_seconds'] = time.perf_counter() - started
        config['end_time'] = datetime.now().isoformat()
        write_json(out / 'config.json', config)
        sys.stdout, sys.stderr = original_out, original_err
        log.close()


if __name__ == '__main__':
    main()
