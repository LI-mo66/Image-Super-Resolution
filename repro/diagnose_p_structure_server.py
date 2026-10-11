"""Fixed-weight post-hoc diagnosis; no training or architecture benefit claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import zipfile


def select_images(rows):
    index = {s: {(r['dataset'], r['image']): r for r in rows[s]} for s in ('Official', 'P3')}
    if any(len(index[s]) != len(rows[s]) for s in index) or index['Official'].keys() != index['P3'].keys():
        raise ValueError('Duplicate or different source image sets')
    chosen = set(k for k in index['Official'] if k[0] == 'Set5')
    for ds in ('Set14', 'B100', 'Urban100', 'manga109'):
        keys = [k for k in index['Official'] if k[0] == ds]
        delta = lambda k: index['P3'][k]['psnr']-index['Official'][k]['psnr']
        ordered = sorted(keys, key=lambda k: (delta(k), k))
        chosen.update(ordered[:2]+ordered[-2:]+sorted(keys, key=lambda k: (abs(delta(k)), k))[:2])
    chosen.update({('Urban100', 'img045'), ('Urban100', 'img048')})
    return sorted(chosen)


def transform(output, kind, factor):
    if kind == 'beta':
        beta, gamma = output
        return 1+factor*(beta-1), gamma
    if kind == 'gamma':
        beta, gamma = output
        return beta, gamma*factor
    return output*factor


def install(net, variant, stats):
    import torch
    handles = []
    kind, factor = variant.rsplit(':', 1) if ':' in variant else ('observe', '1')
    factor = float(factor)
    stage = None
    if kind.startswith('beta@'):
        stage = int(kind.split('@')[1])
        if stage not in range(8):
            raise ValueError('Stage must be 0..7')
        kind = 'beta'
    cache = {}
    def capture_prev(module, inputs, output):
        cache['prev'] = output.detach()
    for name, module in net.named_modules():
        if name == 'first_conv' or name.startswith('esas.') and name.count('.') == 1:
            handles.append(module.register_forward_hook(capture_prev))
    for name, module in net.named_modules():
        cls = type(module).__name__
        if cls not in ('SFML', 'IASA', 'Attention'):
            continue
        def hook(module, inputs, output, name=name, cls=cls):
            tensors = output if isinstance(output, tuple) else (output,)
            record = stats.setdefault(name, {'class': cls, 'calls': 0, 'rms': []})
            record['calls'] += 1
            if not all(torch.isfinite(t).all() for t in tensors):
                raise ValueError(f'Nonfinite module {name}')
            record['rms'].append([float(t.float().square().mean().sqrt()) for t in tensors])
            if cls == 'SFML':
                effective = transform(output, kind, factor) if kind in ('beta', 'gamma') and (stage is None or name == f'sfmls.{stage}') else output
                if 'prev' in cache:
                    beta, gamma = effective
                    prev = cache['prev']
                    if beta.shape != prev.shape or gamma.shape != prev.shape:
                        raise ValueError(f'SFML/prev shape mismatch: {name}')
                    rms = lambda t: float(t.float().square().mean().sqrt())
                    bp = beta*prev
                    record['modulation'] = {'prev_rms': rms(prev), 'beta_prev_rms': rms(bp),
                        'gamma_rms': rms(gamma), 'fm_rms': rms(bp+gamma),
                        'gamma_to_beta_prev_rms': rms(gamma)/max(rms(bp), 1e-12),
                        'beta_mean': float(beta.mean()), 'beta_min': float(beta.min()), 'beta_max': float(beta.max())}
                return effective
            if (cls == 'IASA' and kind == 'iasa') or (cls == 'Attention' and kind == 'lrsa'):
                return transform(output, kind, factor)
            return output
        handles.append(module.register_forward_hook(hook))
    return handles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-eval', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--stage-beta-only', action='store_true', help='Official only; perturb one SFML stage at a time')
    args = parser.parse_args()
    if os.name == 'nt':
        raise RuntimeError('Real model diagnosis is server-only')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    import numpy as np
    from PIL import Image
    from train_p_adaptation import ROOT, Baseline, EXPECTED_CHECKPOINT, atomic_json, sha, measure, masks_for, read_pair
    from p_adapt_logging import RunLogger
    previous = json.loads((args.previous_eval/'config.json').read_text())
    if previous['status'] != 'completed' or previous['official_checkpoint_sha256'] != EXPECTED_CHECKPOINT:
        raise ValueError('Need completed official/P0/P3 evaluation')
    for relative, digest in previous['evaluation_source_sha256'].items():
        if relative != 'repro/eval_p03_final_server.py' and sha(ROOT/relative) != digest:
            raise ValueError(f'Changed evaluation source: {relative}')
    rows = {s: json.loads((args.previous_eval/f'{s}_per_image.json').read_text()) for s in ('Official', 'P3')}
    selected = select_images(rows)
    paths = {'Official': Path(previous['official_checkpoint'])}
    if not args.stage_beta_only:
        paths.update({s: Path(previous['source_runs'][s]['checkpoint']) for s in ('P0', 'P3')})
    states = {}
    for scheme, path in paths.items():
        expected = EXPECTED_CHECKPOINT if scheme == 'Official' else previous['source_runs'][scheme]['config']['final_checkpoint_sha256']
        if sha(path) != expected:
            raise ValueError(f'Checkpoint hash mismatch: {scheme}')
        state = torch.load(path, map_location='cpu', weights_only=scheme == 'Official')
        states[scheme] = state if scheme == 'Official' else state['model']
    config = {'training': False, 'post_hoc_selection': True, 'selected': selected,
              'previous_config': previous, 'previous_eval': str(args.previous_eval.resolve()),
              'device': 'cuda:0', 'gpu_name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'script_sha256': sha(__file__), 'error_visualization': 'mean absolute RGB error times8, clipped0..255',
              'region_scope': 'LR strata; mixed is not periodic texture classifier'}
    variants = ['observe', 'repeat']+([f'beta@{stage}:{factor}' for stage in range(8) for factor in (.98, 1.02)]
                if args.stage_beta_only else [f'{kind}:{factor}' for kind in ('beta', 'gamma', 'iasa', 'lrsa') for factor in (.98, 1.02)])
    config.update(stage_beta_only=args.stage_beta_only, variants=variants,
                  modulation_semantics='prev=first_conv or previous ESA output; effective beta/gamma before TAB')
    reference = {(r['dataset'], r['image']): r for r in rows['Official']}
    with RunLogger(ROOT/'experiment/all_runs', 'P_stage_beta_diagnostic' if args.stage_beta_only else 'P_structure_diagnostic', 4, 1, config) as log:
        records = []
        net = Baseline(scale=4).to('cuda:0').eval()
        for ds, name in selected:
            folder = args.data_root/'benchmark'/ds
            if ds == 'manga109' and not folder.exists():
                folder = args.data_root/'benchmark/Manga109'
            hp, lp = folder/'HR'/f'{name}.png', folder/'LR_bicubic/X4'/f'{name}x4.png'
            if sha(hp) != reference[(ds, name)]['hr_hash'] or sha(lp) != reference[(ds, name)]['lr_hash']:
                raise ValueError(f'Data differs from source: {ds}/{name}')
            lr, hr, alignment = read_pair(hp, lp, 4, 0)
            masks = masks_for(lr)
            lr, hr = lr.to('cuda:0'), hr.to('cuda:0')
            image_dir = log.directory/'images'/ds/name
            image_dir.mkdir(parents=True)
            def png(tensor, target):
                data = tensor.detach().clamp(0, 255).round().byte().cpu().numpy()
                if data.ndim == 4: data = data[0].transpose(1, 2, 0)
                Image.fromarray(data).save(target)
            png(lr, image_dir/'LR.png')
            png(hr, image_dir/'HR.png')
            base = None
            for scheme in paths:
                for variant in variants if scheme == 'Official' else ['observe']:
                    net.load_state_dict(states[scheme], strict=True)
                    torch.manual_seed(1)
                    torch.cuda.manual_seed_all(1)
                    stats = {}
                    handles = install(net, variant, stats)
                    try:
                        with torch.inference_mode(): sr = net(lr)
                    finally:
                        for handle in handles: handle.remove()
                    if {r['class'] for r in stats.values()} != {'SFML', 'IASA', 'Attention'}:
                        raise ValueError('Expected SFML/IASA/LRSA observations not all executed')
                    if args.stage_beta_only and sum('modulation' in r for r in stats.values()) != 8:
                        raise ValueError('Expected eight actual SFML modulation observations')
                    for key, value in net.state_dict().items():
                        if not torch.equal(value.cpu(), states[scheme][key]):
                            raise ValueError(f'Eval changed model state: {key}')
                    metrics = measure(sr, hr, 4, masks)
                    if scheme == 'Official' and variant == 'observe': base = sr.detach().clone()
                    record = {'dataset': ds, 'image': name, 'scheme': scheme, 'variant': variant,
                              'metrics': metrics, 'modules': stats, 'alignment': alignment,
                              'max_abs_vs_official_observe': float((sr-base).abs().max())}
                    records.append(record)
                    if variant == 'observe':
                        png(sr, image_dir/f'{scheme}_SR.png')
                        error = (sr.clamp(0, 255).round()-hr).abs().mean(1)[0]*8
                        png(error, image_dir/f'{scheme}_RGB_error_x8.png')
                    atomic_json(log.directory/'diagnostic.json', records)
                    print(f'{ds}/{name} {scheme} {variant}: PSNR={metrics["psnr_y_quantized"]:.8f}', flush=True)
        if args.stage_beta_only:
            from summarize_p_adaptation import paired_stats
            base_rows = {(r['dataset'], r['image']): r for r in records if r['variant'] == 'observe'}
            summary = {}
            for variant in variants[1:]:
                subset = [r for r in records if r['variant'] == variant]
                summary[variant] = {}
                for ds in ['selected_all']+sorted({r['dataset'] for r in subset}):
                    selected_rows = subset if ds == 'selected_all' else [r for r in subset if r['dataset'] == ds]
                    summary[variant][ds] = {metric: paired_stats([
                        (r['dataset']+'/'+r['image'], r['metrics'][metric]-base_rows[(r['dataset'], r['image'])]['metrics'][metric])
                        for r in selected_rows]) for metric in ('psnr_y_quantized', 'ssim_y_quantized')}
            atomic_json(log.directory/'stage_summary.json', {
                'scope': 'post-hoc selected diagnostic images; not independent gain evidence', 'variants': summary})
        log.update(final_metrics={'images': len(selected), 'records': len(records)})
        output = log.directory
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file(): bundle.write(path, path.relative_to(output.parent))
    print(f'RESULT_ZIP={archive}', flush=True)


if __name__ == '__main__':
    main()
