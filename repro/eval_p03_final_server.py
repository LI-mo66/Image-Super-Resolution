"""Evaluate existing final P0/P3 checkpoints only; never train or select best."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

from summarize_p_adaptation import comparable, paired_stats

COUNTS = {'Set5': 5, 'Set14': 14, 'B100': 100, 'Urban100': 100, 'manga109': 109}


def compare(rows):
    indexed = {s: {(r['dataset'], r['image']): r for r in rows[s]} for s in ('P0', 'P3')}
    if indexed['P0'].keys() != indexed['P3'].keys():
        raise ValueError('Image sets differ')
    result = {}
    for ds, count in COUNTS.items():
        keys = sorted(k for k in indexed['P0'] if k[0] == ds)
        if len(keys) != count:
            raise ValueError(f'{ds}: expected {count}, got {len(keys)}')
        for k in keys:
            if any(not math.isfinite(indexed[s][k][metric]) for s in ('P0', 'P3') for metric in ('psnr', 'ssim')):
                raise ValueError(f'Nonfinite metric: {k}')
            for field in ('lr_hash', 'hr_hash', 'alignment'):
                if indexed['P0'][k][field] != indexed['P3'][k][field]:
                    raise ValueError(f'Data mismatch: {k} {field}')
        entry = {}
        for metric in ('psnr', 'ssim'):
            entry[metric] = {
                s: statistics.mean(indexed[s][k][metric] for k in keys) for s in ('P0', 'P3')}
            entry[metric]['P3_minus_P0'] = paired_stats([
                (k[1], indexed['P3'][k][metric] - indexed['P0'][k][metric]) for k in keys])
        result[ds] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--official-checkpoint', type=Path,
                        help='Also evaluate the original official x4 weights, with strict SHA validation')
    args = parser.parse_args()
    if os.name == 'nt':
        raise RuntimeError('Actual LFMN evaluation is server-only')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import torch
    from train_p_adaptation import Baseline, ROOT, EXPECTED_CHECKPOINT, atomic_json, evaluate, sha
    from p_adapt_logging import RunLogger
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if args.official_checkpoint and sha(args.official_checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError('Original official x4 checkpoint SHA mismatch')
    sources = {}
    for scheme in ('P0', 'P3'):
        matches = sorted(args.run_root.glob(f'{scheme}_x4_seed1_*/config.json'))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {scheme} run: {matches}')
        config = json.loads(matches[0].read_text(encoding='utf-8'))
        checkpoint = matches[0].parent / 'checkpoints/last.pt'
        if config['status'] != 'completed' or config['exit_code'] != 0:
            raise ValueError(f'{scheme}: training incomplete')
        if sha(checkpoint) != config['final_checkpoint_sha256']:
            raise ValueError(f'{scheme}: final checkpoint hash mismatch')
        sources[scheme] = (config, checkpoint)
    a, b = (sources[s][0] for s in ('P0', 'P3'))
    for field in ('batch_plan_sha256', 'data_stream_sha256', 'initial_checkpoint_sha256'):
        if a[field] != b[field]:
            raise ValueError(f'Training evidence mismatch: {field}')
    if comparable(a['protocol_fingerprint']) != comparable(b['protocol_fingerprint']):
        raise ValueError('Training protocol mismatch')
    for field, path in [('model_sha256', ROOT/'LFMN/model/lfmn.py'),
                        ('metric_source_sha256', ROOT/'LFMN/utility.py'),
                        ('runner_sha256', ROOT/'repro/train_p_adaptation.py')]:
        if sha(path) != a['protocol_fingerprint'][field]:
            raise ValueError(f'Evaluation source differs from training: {field}')
    config = {'evaluation_only': True, 'dataset': str(args.data_root.resolve()),
              'metric_protocol': a['metric_protocol'], 'device': 'cuda:0',
              'gpu_name': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_runs': {s: {'config': c, 'checkpoint': str(p.resolve())} for s, (c, p) in sources.items()},
              'selection': 'final epoch20/global_step2000 only; full image; no ensemble/chop',
              'official_checkpoint': str(args.official_checkpoint.resolve()) if args.official_checkpoint else None,
              'official_checkpoint_sha256': EXPECTED_CHECKPOINT if args.official_checkpoint else None,
              'evaluation_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                  (Path(__file__), ROOT/'repro/train_p_adaptation.py', ROOT/'repro/run_p_diagnostics.py', ROOT/'LFMN/model/lfmn.py')}}
    with RunLogger(ROOT/'experiment/all_runs', 'P03_final_benchmark', 4, 1, config) as log:
        rows = {}
        if args.official_checkpoint:
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            net = Baseline(scale=4).to('cuda:0')
            state = torch.load(args.official_checkpoint, map_location='cpu', weights_only=True)
            net.load_state_dict(state, strict=True)
            del state
            print('Official: full five-dataset x4 evaluation starts', flush=True)
            rows['Official'] = evaluate(net, args.data_root, 'cuda:0', 0, 0, benchmark=True)
            atomic_json(log.directory/'Official_per_image.json', rows['Official'])
            print(f"Official: completed {len(rows['Official'])} images", flush=True)
            del net
            torch.cuda.empty_cache()
        for scheme, (source, path) in sources.items():
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state['scheme'] != scheme or state['global_step'] != 2000 or state['epoch'] != 20:
                raise ValueError(f'{scheme}: wrong final checkpoint')
            if state['protocol_fingerprint'] != source['protocol_fingerprint']:
                raise ValueError(f'{scheme}: checkpoint protocol mismatch')
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            net = Baseline(scale=4).to('cuda:0')
            net.load_state_dict(state['model'], strict=True)
            del state
            print(f'{scheme}: full five-dataset x4 evaluation starts; checkpoint={path}', flush=True)
            rows[scheme] = evaluate(net, args.data_root, 'cuda:0', 20, 2000, benchmark=True)
            atomic_json(log.directory/f'{scheme}_per_image.json', rows[scheme])
            print(f'{scheme}: completed {len(rows[scheme])} images', flush=True)
            del net
            torch.cuda.empty_cache()
        summary = compare(rows)
        atomic_json(log.directory/'summary.json', summary)
        if args.official_checkpoint:
            comparisons = {scheme+'_minus_Official': compare({'P0': rows['Official'], 'P3': rows[scheme]})
                           for scheme in ('P0', 'P3')}
            # In each pair's schema P0 denotes reference Official and P3 denotes named candidate.
            atomic_json(log.directory/'official_comparisons.json', {
                'schema': 'P0=Official reference, P3=named candidate; P3_minus_P0=candidate minus Official',
                'comparisons': comparisons})
            official_lines = ['# Final weights versus original official x4 weights', '',
                             '| Dataset | Official | P0 | P3 | P0-Official | P3-Official |',
                             '|---|---:|---:|---:|---:|---:|']
            for ds in COUNTS:
                ref = comparisons['P0_minus_Official'][ds]['psnr']['P0']
                d0 = comparisons['P0_minus_Official'][ds]['psnr']['P3_minus_P0']['mean']
                d3 = comparisons['P3_minus_Official'][ds]['psnr']['P3_minus_P0']['mean']
                official_lines.append(f"| {ds} | {ref:.8f} | {summary[ds]['psnr']['P0']:.8f} | {summary[ds]['psnr']['P3']:.8f} | {d0:+.8f} | {d3:+.8f} |")
            official_report = '\n'.join(official_lines)+'\n'
            (log.directory/'official_summary.md').write_text(official_report, encoding='utf-8')
            print(official_report, flush=True)
        with (log.directory/'paired_images.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['dataset', 'image', 'P0_psnr', 'P3_psnr', 'delta_psnr', 'P0_ssim', 'P3_ssim', 'delta_ssim'])
            p0 = {(r['dataset'], r['image']): r for r in rows['P0']}
            for r in rows['P3']:
                base = p0[(r['dataset'], r['image'])]
                writer.writerow([r['dataset'], r['image'], base['psnr'], r['psnr'], r['psnr']-base['psnr'], base['ssim'], r['ssim'], r['ssim']-base['ssim']])
        lines = ['# P0/P3 final x4 benchmark', '', '| Dataset | P0 PSNR | P3 PSNR | Delta PSNR | Delta SSIM | Wins |', '|---|---:|---:|---:|---:|---:|']
        for ds, entry in summary.items():
            delta = entry['psnr']['P3_minus_P0']
            lines.append(f"| {ds} | {entry['psnr']['P0']:.8f} | {entry['psnr']['P3']:.8f} | {delta['mean']:+.8f} | {entry['ssim']['P3_minus_P0']['mean']:+.8f} | {delta['win_rate']:.3%} |")
        report = '\n'.join(lines)+'\n'
        (log.directory/'summary.md').write_text(report, encoding='utf-8')
        print(report, flush=True)
        log.update(final_metrics=summary)
        output = log.directory
    archive = output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.csv', '.md', '.txt'):
                bundle.write(path, path.relative_to(output.parent))
    print(f'RESULT_ZIP={archive}', flush=True)


if __name__ == '__main__':
    main()
