#!/usr/bin/env python3
"""Read-only B0/F1 checkpoint diagnostics with a small return archive."""
import argparse
import csv
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from run_logging import launch, write_json
from source_provenance import provenance


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(value):
    import torch
    tensor = value.detach().float()
    if not torch.isfinite(tensor).all():
        raise FloatingPointError('non-finite diagnostic tensor')
    return float(tensor.square().mean().sqrt())


def ratio(a, b):
    return a / b if abs(b) > 1e-12 else None


def cosine(a, b):
    left, right = a.detach().float().flatten(), b.detach().float().flatten()
    dot = float((left * right).sum())
    denominator = float(left.norm() * right.norm())
    return ratio(dot, denominator)


class StageProbe:
    """Observe actual stage states; F1's wrapper returns its original result."""

    def __init__(self, model, label):
        self.model, self.label = model, label
        self.handles = []
        self.reset('')
        self.handles.append(model.first_conv.register_forward_hook(self.first))
        for index, module in enumerate(model.mid_convs):
            self.handles.append(module.register_forward_hook(self.delta_hook(index)))
            self.handles.append(model.esas[index].sigmoid.register_forward_hook(self.mask_hook(index)))
            self.handles.append(model.esas[index].register_forward_hook(self.esa_hook(index)))
        self.original_update = None
        if label == 'F1':
            self.original_update = model.update_stage
            original = self.original_update
            def observed_update(previous, delta, stage):
                result = original(previous, delta, stage)
                self.record(stage, previous, delta, self.esa_inputs[stage], self.esa_outputs[stage], result)
                return result
            model.update_stage = observed_update

    def reset(self, sample):
        self.sample = sample
        self.previous = None
        self.delta_values, self.masks = {}, {}
        self.esa_inputs, self.esa_outputs = {}, {}
        self.states, self.deltas, self.rows = [], [], []

    def first(self, module, inputs, output):
        self.previous = output

    def delta_hook(self, index):
        def hook(module, inputs, output):
            self.delta_values[index] = output
        return hook

    def mask_hook(self, index):
        def hook(module, inputs, output):
            self.masks[index] = output.detach()
        return hook

    def esa_hook(self, index):
        def hook(module, inputs, output):
            self.esa_inputs[index], self.esa_outputs[index] = inputs[0], output
            if self.label == 'B0':
                self.record(index, self.previous, self.delta_values[index], inputs[0], output, output)
        return hook

    def record(self, stage, previous, delta, esa_input, esa_output, state):
        import numpy as np
        mask = self.masks[stage].float()
        sampled = mask.flatten()[::max(1, mask.numel() // 16384)][:16384].cpu().numpy()
        q05, q50, q95 = np.quantile(sampled, [0.05, 0.5, 0.95])
        previous_rms = rms(previous)
        update = state.detach() - previous.detach()
        retention = rms(mask * previous.detach()) if self.label == 'B0' else previous_rms
        projection = float((delta.detach().float() * previous.detach().float()).mean())
        row = dict(model=self.label, sample=self.sample, stage=stage + 1,
                   previous_rms=previous_rms, delta_rms=rms(delta),
                   delta_to_previous=ratio(rms(delta), previous_rms),
                   esa_input_rms=rms(esa_input), esa_output_rms=rms(esa_output),
                   esa_output_to_input=ratio(rms(esa_output), rms(esa_input)),
                   effective_update_rms=rms(update), update_to_previous=ratio(rms(update), previous_rms),
                   state_rms=rms(state), state_to_previous=ratio(rms(state), previous_rms),
                   delta_previous_cosine=cosine(delta, previous),
                   state_previous_cosine=cosine(state, previous),
                   delta_previous_projection=ratio(projection, previous_rms ** 2),
                   direct_previous_retention=ratio(retention, previous_rms),
                   alpha=float(self.model.residual_scales[stage].detach()) if self.label == 'F1' else None,
                   mask_mean=float(mask.mean()), mask_min=float(mask.min()), mask_max=float(mask.max()),
                   mask_q05_sampled=float(q05), mask_q50_sampled=float(q50), mask_q95_sampled=float(q95),
                   mask_lt005=float((mask < 0.05).float().mean()),
                   mask_gt095=float((mask > 0.95).float().mean()),
                   mask_lt001=float((mask < 0.01).float().mean()),
                   mask_gt099=float((mask > 0.99).float().mean()))
        self.rows.append(row)
        self.states.append(state)
        self.deltas.append(delta)
        self.previous = state

    def close(self):
        for handle in self.handles:
            handle.remove()
        if self.original_update is not None:
            del self.model.update_stage
            self.original_update = None
        self.reset('')


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def read_pair(hr_path, lr_path, patch=None):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    def rgb(path):
        value = imageio.imread(path)
        if value.ndim == 2:
            value = np.repeat(value[:, :, None], 3, axis=2)
        if value.ndim != 3 or value.shape[2] != 3:
            raise ValueError('diagnostics require RGB/grayscale images, not RGBA')
        return value
    hr, lr = rgb(hr_path), rgb(lr_path)
    height, width = lr.shape[:2]
    if hr.shape[0] // 4 != height or hr.shape[1] // 4 != width:
        raise ValueError('x4 LR/HR dimensions mismatch: ' + str(hr_path))
    hr = hr[:height * 4, :width * 4]
    if patch:
        if min(height, width) < patch:
            raise ValueError('image smaller than diagnostic patch')
        top, left = (height - patch) // 2, (width - patch) // 2
        lr = lr[top:top + patch, left:left + patch]
        hr = hr[top * 4:(top + patch) * 4, left * 4:(left + patch) * 4]
    lr, hr = [torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float() for value in (lr, hr)]
    if lr.shape[0] != 3 or hr.shape[0] != 3:
        raise ValueError('diagnostics require RGB/grayscale images, not RGBA')
    return lr.unsqueeze(0), hr.unsqueeze(0)


def memory_snapshot():
    result = {}
    for name in ('memory.max', 'memory.current', 'memory.events'):
        path = Path('/sys/fs/cgroup') / name
        try:
            result[str(path)] = path.read_text().strip()
        except OSError:
            pass
    for name in ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.oom_control'):
        path = Path('/sys/fs/cgroup/memory') / name
        try:
            result[str(path)] = path.read_text().strip()
        except OSError:
            pass
    return result


def saved_set5(args, label):
    for source in sorted((args.group / 'comparison_reports').glob('*/set5_per_image_comparison.csv'), reverse=True):
        with source.open(newline='', encoding='utf-8-sig') as stream:
            rows = [row for row in csv.DictReader(stream) if int(row['epoch']) == args.epoch]
        if len(rows) != 5 or len({row['filename'] for row in rows}) != 5:
            continue
        return [dict(model=label, epoch=args.epoch, filename=row['filename'],
                     psnr=float(row[label + '_psnr']), ssim=float(row[label + '_ssim']),
                     metric_origin='saved_report_not_recomputed', metric_source=str(source),
                     metric_source_sha256=sha256(source)) for row in rows]
    print('WARNING: no complete saved Set5 report for ' + label + '; metrics not measured', flush=True)
    return []


def collect_records(args, output):
    destination = output / 'source_records'
    copied = []
    for label in ('B0', 'F1'):
        for name in ('config.json', 'config.txt', 'metrics.csv', 'train_log.txt', 'log.txt'):
            source = args.group / (label + '_x4_seed1') / name
            if source.is_file():
                target = destination / label / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                copied.append(dict(source=str(source), copied=str(target.relative_to(output)), sha256=sha256(source)))
    for name in ('protocol.json', 'resources.json', 'benchmark_comparison.csv', 'summary.txt', 'evaluation_paths.json'):
        source = args.group / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(dict(source=str(source), copied=str(target.relative_to(output)), sha256=sha256(source)))
    reports = args.group / 'comparison_reports'
    if reports.exists():
        for source in sorted(reports.glob('*/set5_per_image_comparison.csv')):
            target = destination / 'set5_history' / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(dict(source=str(source), copied=str(target.relative_to(output)), sha256=sha256(source)))
    write_json(output / 'source_records_manifest.json', copied)
    return copied


def protocol_check(args, output):
    configs = {}
    for label in ('B0', 'F1'):
        path = args.group / (label + '_x4_seed1') / 'config.json'
        configs[label] = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    differences, unknown = {}, []
    keys = ('seed', 'scale', 'patch_size', 'batch_size', 'lr', 'optimizer', 'scheduler',
            'scheduler_t_max', 'eta_min', 'data_range', 'loss', 'workers', 'self_ensemble',
            'pretrained_checkpoint', 'initial_shared_state_sha256', 'metric_protocol')
    for key in keys:
        if any(key not in configs[label] for label in configs):
            unknown.append(key)
        elif configs['B0'][key] != configs['F1'][key]:
            differences[key] = {label: configs[label][key] for label in configs}
    status = 'mismatch' if differences else ('incomplete' if unknown else 'matched_registered_fields')
    result = dict(status=status, differences=differences, unknown_fields=unknown,
                  source_configs={label: str(args.group / (label + '_x4_seed1/config.json')) for label in configs})
    write_json(output / 'protocol_check.json', result)
    return result


def worker(args, output):
    print('PHASE importing torch', flush=True)
    import torch
    from model.lfmn import Net as B0
    from model.lfmnf1 import Net as F1
    if args.cpu:
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
    if args.set5_mode == 'full':
        import utility
    print('PHASE imports complete', flush=True)
    device = torch.device('cpu' if args.cpu else 'cuda')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --cpu explicitly')
    torch.manual_seed(1)
    checkpoint_sources = {}
    for label in ('B0', 'F1'):
        path = args.group / (label + '_x4_seed1/model') / ('model_{}.pt'.format(args.epoch))
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint_sources[label] = dict(path=str(path), sha256=sha256(path))
    records = collect_records(args, output)
    checked = protocol_check(args, output)
    manifest = dict(epoch=args.epoch, seed=1, device=str(device), torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
                    source_group=str(args.group), checkpoints=checkpoint_sources,
                    self_ensemble=False, source=provenance(ROOT),
                    module_paths=dict(B0=inspect.getfile(B0), F1=inspect.getfile(F1)),
                    model_source_sha256={name: sha256(ROOT / name) for name in
                                         ('LFMN/model/lfmn.py', 'LFMN/model/lfmnf1.py', 'LFMN/utility.py')},
                    stage_protocol='fixed DIV2K 0801 onward; center LR{}/HR{}; RGB raw255'.format(args.patch_size, args.patch_size * 4),
                    gradient_samples=args.gradient_samples, set5_mode=args.set5_mode,
                    gradient_protocol='eval-mode raw RGB L1, no optimizer step; not historical training gradients',
                    set5_protocol=('saved original report; no fresh CPU Set5 measurement; audit source protocol separately'
                                   if args.set5_mode == 'saved' else
                                   'full image x4 OFF; existing utility quantize255/Y PSNR crop4/SSIM MATLAB Y crop4'),
                    copied_records=len(records), protocol_check=checked,
                    interpretation='Feature correlation is not functional redundancy; probes do not establish trained gains')
    write_json(output / 'manifest.json', manifest)
    stage_rows, gradient_rows, set5_rows, parity = [], [], [], []
    alpha_rows = []
    for path in sorted((args.group / 'F1_x4_seed1/model').glob('model_*.pt')):
        if not path.stem.removeprefix('model_').isdigit():
            continue
        epoch = int(path.stem.removeprefix('model_'))
        if epoch > args.epoch:
            continue
        state = torch.load(path, map_location='cpu', weights_only=True)
        if 'residual_scales' not in state:
            raise ValueError('F1 checkpoint lacks residual_scales: ' + str(path))
        checkpoint_hash = sha256(path)
        for index, alpha in enumerate(state['residual_scales'].flatten()):
            alpha_rows.append(dict(epoch=epoch, stage=index + 1, alpha=float(alpha),
                                   checkpoint=str(path), checkpoint_sha256=checkpoint_hash))
        del state
    save_csv(output / 'alpha_history.csv', sorted(alpha_rows, key=lambda row: (row['epoch'], row['stage'])))
    benchmark = SimpleNamespace(dataset=SimpleNamespace(benchmark=True))
    for label, cls in (('B0', B0), ('F1', F1)):
        print('PHASE loading ' + label, flush=True)
        net = cls(scale=4).to(device).eval()
        state = torch.load(checkpoint_sources[label]['path'], map_location='cpu', weights_only=True)
        net.load_state_dict(state, strict=True)
        before = state_digest(net)
        probe = StageProbe(net, label)
        try:
            for index in range(args.samples):
                image = '{:04d}'.format(801 + index)
                lr, hr = read_pair(args.data_root / 'DIV2K/DIV2K_valid_HR' / (image + '.png'),
                                   args.data_root / 'DIV2K/DIV2K_valid_LR_bicubic/X4' / (image + 'x4.png'), patch=args.patch_size)
                lr, hr = lr.to(device), hr.to(device)
                probe.reset(image)
                print('PHASE forward {} {}'.format(label, image), flush=True)
                with torch.no_grad():
                    net(lr)
                if len(probe.rows) != 8:
                    raise AssertionError('incomplete eight-stage trace')
                stage_rows.extend(probe.rows)
                if index < args.gradient_samples:
                    print('PHASE gradient {} {}'.format(label, image), flush=True)
                    probe.reset(image + '_gradient')
                    sr = net(lr)
                    loss = (sr - hr).abs().mean()
                    targets = probe.states + probe.deltas
                    if label == 'F1':
                        targets += [net.residual_scales]
                    gradients = torch.autograd.grad(loss, targets)
                    if not all(torch.isfinite(gradient).all() for gradient in gradients):
                        raise FloatingPointError('non-finite eval-mode sensitivity gradients')
                    for stage in range(8):
                        gradient_rows.append(dict(model=label, sample=image, stage=stage + 1,
                                                  raw_rgb_l1=float(loss.detach()),
                                                  state_gradient_rms=rms(gradients[stage]),
                                                  delta_gradient_rms=rms(gradients[8 + stage]),
                                                  alpha_gradient=float(gradients[-1][stage]) if label == 'F1' else None))
                    del sr, loss, gradients, targets
                print('PROBE {} {} complete'.format(label, image), flush=True)
                save_csv(output / 'stage_stats.csv', stage_rows)
                save_csv(output / 'gradient_stats.csv', gradient_rows)
            probe.close()
            if args.set5_mode == 'saved':
                set5_rows.extend(saved_set5(args, label))
            hr_files = sorted(path for path in (args.data_root / 'benchmark/Set5/HR').glob('*') if path.is_file())
            if args.set5_mode == 'full' and len(hr_files) != 5:
                raise ValueError('Set5 must have exactly five HR images')
            for path in hr_files if args.set5_mode == 'full' else []:
                print('PHASE full Set5 {} {}'.format(label, path.stem), flush=True)
                lr, hr = read_pair(path, args.data_root / 'benchmark/Set5/LR_bicubic/X4' / (path.stem + 'x4.png'))
                lr, hr = lr.to(device), hr.to(device)
                with torch.no_grad():
                    sr = utility.quantize(net(lr), 255)
                set5_rows.append(dict(model=label, epoch=args.epoch, filename=path.stem,
                                      psnr=utility.calc_psnr(sr, hr, 4, 255, dataset=benchmark),
                                      ssim=float(utility.calc_ssim(sr, hr, 4, 255, dataset=benchmark))))
            lr, _ = read_pair(args.data_root / 'DIV2K/DIV2K_valid_HR/0801.png',
                              args.data_root / 'DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png', patch=args.patch_size)
            with torch.no_grad():
                plain = net(lr.to(device))
                check_probe = StageProbe(net, label)
                observed = net(lr.to(device))
                check_probe.close()
            torch.testing.assert_close(observed, plain, rtol=1e-5, atol=1e-5)
            maximum = float((observed - plain).abs().max())
            after = state_digest(net)
            if before != after:
                raise AssertionError('diagnostics changed model parameters or buffers')
            if sha256(checkpoint_sources[label]['path']) != checkpoint_sources[label]['sha256']:
                raise AssertionError('source checkpoint file changed during diagnostics')
            parity.append(dict(model=label, monitor_max_abs_difference=maximum,
                               model_state_unchanged=True, source_checkpoint_unchanged=True))
        finally:
            probe.close()
        del net, state
    save_csv(output / 'stage_stats.csv', stage_rows)
    save_csv(output / 'gradient_stats.csv', gradient_rows)
    save_csv(output / ('set5_saved.csv' if args.set5_mode == 'saved' else 'set5_recheck.csv'), set5_rows)
    manifest['set5_rows'] = len(set5_rows)
    write_json(output / 'read_only_checks.json', parity)
    manifest['read_only_checks'] = parity
    manifest['diagnostic_status'] = 'completed'
    write_json(output / 'manifest.json', manifest)
    averages = []
    for label in ('B0', 'F1'):
        for stage in range(1, 9):
            rows = [row for row in stage_rows if row['model'] == label and row['stage'] == stage]
            mean = dict(model=label, stage=stage)
            for key in stage_rows[0]:
                if key in ('model', 'sample', 'stage'):
                    continue
                values = [row[key] for row in rows if row[key] is not None]
                mean[key] = sum(values) / len(values) if values else None
            averages.append(mean)
    save_csv(output / 'stage_summary.csv', averages)
    lines = ['# Read-only B0/F1 diagnostic', '', 'Protocol check: ' + checked['status'],
             'LR patch: {}; gradient samples: {}; Set5 mode: {}.'.format(args.patch_size, args.gradient_samples, args.set5_mode),
             'No training or optimizer step. All source checkpoint and model-state checks passed.', '',
             '| Model | Stage | Delta/previous RMS | Update/previous RMS | State/previous RMS | Mask mean | Alpha |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in averages:
        values = [row[key] for key in ('delta_to_previous', 'update_to_previous', 'state_to_previous', 'mask_mean', 'alpha')]
        lines.append('| {} | {} | {} |'.format(row['model'], row['stage'],
                     ' | '.join('{:.6f}'.format(value) if value is not None else 'NA' for value in values)))
    lines += ['', 'Mask quantiles use a fixed-stride sample of at most 16384 values; means/saturation use all values.',
              'Stage/gradient measurements use fixed independent-validation center patches, not full benchmark metrics.',
              'Gradients are eval-mode sensitivity, not evidence of training numerical instability.',
              'Feature similarity alone does not establish redundant information or performance benefits.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--gradient-samples', type=int)
    parser.add_argument('--patch-size', type=int)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--set5-mode', choices=('saved', 'full'))
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.samples = args.samples if args.samples is not None else (3 if args.cpu else 10)
    args.gradient_samples = args.gradient_samples if args.gradient_samples is not None else (0 if args.cpu else 3)
    args.patch_size = args.patch_size if args.patch_size is not None else (28 if args.cpu else 64)
    args.set5_mode = args.set5_mode or ('saved' if args.cpu else 'full')
    if args.patch_size < 28 or args.cpu_threads < 1:
        raise ValueError('patch-size must be >=28; cpu-threads must be positive')
    args.group, args.data_root = args.group.resolve(), args.data_root.resolve()
    if not 1 <= args.samples <= 100 or not 0 <= args.gradient_samples <= args.samples:
        raise ValueError('samples must be 1..100; gradient-samples must be 0..samples')
    if args.epoch < 1:
        raise ValueError('epoch must be positive')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one diagnostic process, not DDP')
    if os.environ.get('LFMN_MANAGED_RUN'):
        worker(args, Path(os.environ['LFMN_MANAGED_RUN']))
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.output.resolve() if args.output else args.group / 'diagnostics' / ('F1_B0_x4_seed1_' + stamp)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), *sys.argv[1:],
               '--group', str(args.group), '--data-root', str(args.data_root)]
    config = dict(kind='read_only_diagnostic', group=str(args.group), data_root=str(args.data_root),
                  epoch=args.epoch, samples=args.samples, gradient_samples=args.gradient_samples,
                  cpu=args.cpu, cpu_threads=args.cpu_threads, patch_size=args.patch_size, set5_mode=args.set5_mode,
                  self_ensemble=False, optimizer_step=False, **provenance(ROOT))
    archive = output.with_name(output.name + '.tar.gz')
    if output.exists() or archive.exists():
        raise FileExistsError('diagnostic directory/archive already exists: ' + str(output))
    memory_before = memory_snapshot()
    try:
        launch(command, output, config, ROOT)
    finally:
        if output.exists():
            write_json(output / 'memory_events.json', dict(before=memory_before, after=memory_snapshot(),
                       note='Container-wide events; changes are not proof this process caused an OOM. Empty means unavailable.'))
            checksums = {str(path.relative_to(output)): sha256(path)
                         for path in sorted(output.rglob('*')) if path.is_file()}
            write_json(output / 'checksums.json', checksums)
            with tarfile.open(archive, 'x:gz') as package:
                package.add(output, arcname=output.name)
            print('SEND THIS ARCHIVE: ' + str(archive), flush=True)
            print('ARCHIVE SHA256: ' + sha256(archive), flush=True)


if __name__ == '__main__':
    main()
