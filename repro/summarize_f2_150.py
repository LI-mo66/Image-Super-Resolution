#!/usr/bin/env python3
"""Validate and compare completed B0/F2 cosine150 artifacts; never train."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from f2_150_common import ROOT, MODELS, BENCHMARKS, PROTOCOL, read_csv, write_csv, sha256_file, run_directory
sys.path.insert(0, str(ROOT / 'LFMN'))
from run_logging import write_json

EPOCHS = set(range(1, 151))
IDS = {str(i).zfill(4) for i in range(801, 901)}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite metric')
    return result

def off(value):
    return value is False or str(value).lower() in ('false', '0')

def indexed(path):
    rows = read_csv(path)
    require(len(rows) == 150 and {int(r['epoch']) for r in rows} == EPOCHS,
            'Missing or duplicate epoch in ' + str(path))
    return {int(r['epoch']): r for r in rows}

def mean(rows, key):
    return sum(number(r[key]) for r in rows) / len(rows)

def close(a, b, tolerance=1e-10):
    require(abs(number(a) - number(b)) <= tolerance, 'Metric mean mismatch')

def image_rows(rows, dataset, count):
    found = [r for r in rows if r['dataset'] == dataset]
    require(len(found) == count and len({r['filename'] for r in found}) == count,
            'Invalid image identities/count for ' + dataset)
    require(all(int(r['scale']) == 4 for r in found), 'Wrong image scale')
    for row in found:
        number(row['psnr']); number(row['ssim'])
    if dataset == 'DIV2K':
        require({r['filename'] for r in found} == IDS, 'DIV2K identities differ from 801-900')
    return sorted(found, key=lambda r: r['filename'])

def checked_config(config, manifest, evaluation=False):
    require(config.get('status') == 'completed', 'Run is not completed')
    require(config.get('protocol') == PROTOCOL, 'Protocol mismatch')
    for key, value in PROTOCOL.items():
        if key in config:
            require(config[key] == value, 'Top-level protocol mismatch: ' + key)
    require(not config.get('smoke') and not config.get('engineering_only'), 'Engineering-only run')
    for key in ('environment', 'source_sha256', 'commit', 'dataset_fingerprint', 'data_root', 'workers'):
        require(key in manifest and config.get(key) == manifest[key], 'Configuration mismatch: ' + key)
    require(bool(config['environment']) and bool(config['source_sha256']), 'Missing provenance')
    if evaluation:
        require(config.get('test_only') is True, 'Evaluation must be test_only')
    else:
        require(config.get('parameter_count') == 759627, 'Wrong parameter count')
        require(bool(config.get('initial_unchanged_state_sha256')), 'Missing initialization fingerprint')

def validate_run(group, label, manifest):
    directory = run_directory(group, label)
    config = read_json(directory / 'config.json')
    checked_config(config, manifest)
    require(config.get('model') == MODELS[label], 'Wrong model identity')
    metrics = indexed(directory / 'metrics.csv')
    validation = indexed(directory / 'validation_per_checkpoint.csv')
    set5 = indexed(directory / 'set5_per_checkpoint.csv')
    batches = indexed(directory / 'first_batch_hashes.csv')
    details = read_csv(directory / 'set5_per_image.csv')
    require(len(details) == 750, 'Missing Set5 per-image epochs')
    details_by_epoch = {e: [] for e in EPOCHS}
    for row in details:
        require(int(row['epoch']) in EPOCHS, 'Unexpected Set5 epoch')
        details_by_epoch[int(row['epoch'])].append(dict(row, dataset='Set5'))
    raw, hashes = {}, {}
    for epoch in range(1, 151):
        checkpoint = directory / 'model' / ('model_{}.pt'.format(epoch))
        digest = sha256_file(checkpoint)
        hashes[epoch] = digest
        rows = torch.load(directory / 'per_image_metrics' / ('epoch_{:04d}.pt'.format(epoch)),
                          map_location='cpu', weights_only=True)
        require(len(rows) == 105, 'Unexpected raw metric dataset count')
        div = image_rows(rows, 'DIV2K', 100)
        five = image_rows(rows, 'Set5', 5)
        raw[epoch] = {'DIV2K': div, 'Set5': five}
        for record, dataset, prefix, count in ((validation[epoch], div, 'validation', 100),
                                               (set5[epoch], five, 'set5', 5)):
            require(record['checkpoint_sha256'] == digest, 'Checkpoint hash mismatch')
            require(int(record['image_count']) == count and int(record['scale']) == 4
                    and off(record['self_ensemble']), 'Invalid checkpoint evaluation protocol')
            close(record[prefix + '_psnr'], mean(dataset, 'psnr'))
            close(record[prefix + '_ssim'], mean(dataset, 'ssim'))
        csv_five = image_rows(details_by_epoch[epoch], 'Set5', 5)
        for a, b in zip(five, csv_five):
            require(a['filename'] == b['filename'] and b['checkpoint_sha256'] == digest
                    and off(b['self_ensemble']), 'Set5 identity/hash mismatch')
            close(a['psnr'], b['psnr']); close(a['ssim'], b['ssim'])
        m = metrics[epoch]
        for key in ('learning_rate', 'train_loss', 'validation_psnr', 'validation_ssim', 'elapsed_seconds'):
            number(m[key])
        require(number(m['train_loss']) >= 0 and number(m['elapsed_seconds']) >= 0, 'Invalid loss/time')
        lr = PROTOCOL['eta_min'] + (PROTOCOL['lr'] - PROTOCOL['eta_min']) * (1 + math.cos(math.pi * (epoch - 1) / 150)) / 2
        close(m['learning_rate'], lr, 1e-12)
        close(m['validation_psnr'], mean(div, 'psnr'), 2e-5)
        close(m['validation_ssim'], mean(div, 'ssim'), 2e-5)
        batch = batches[epoch]
        require(int(batch['batch_size']) == 4 and batch['lr_shape'] == '(4, 3, 64, 64)'
                and batch['hr_shape'] == '(4, 3, 256, 256)', 'Invalid first batch shape')
        require(len(batch['lr_sha256']) == 64 and len(batch['hr_sha256']) == 64, 'Missing first batch hash')
    best = max(range(1, 151), key=lambda e: mean(raw[e]['DIV2K'], 'psnr'))
    return dict(config=config, metrics=metrics, validation=validation, set5=set5,
                batches=batches, raw=raw, hashes=hashes, best=best)

def validate_evaluation(group, label, selection, entry, run, manifest, final_entry=None):
    epoch = 150 if selection == 'final150' else run['best']
    require(int(entry['epoch']) == epoch and entry['checkpoint_sha256'] == run['hashes'][epoch],
            'Evaluation selection/epoch/checkpoint mismatch')
    reused = selection == 'best_div2k' and epoch == 150 and entry == final_entry
    recorded_selection = 'final150' if reused else selection
    directory = (group / entry['directory']).resolve()
    require(directory.is_relative_to(group.resolve()), 'Evaluation path outside group')
    config = read_json(directory / 'config.json')
    checked_config(config, manifest, evaluation=True)
    require(config.get('model') == MODELS[label] and config.get('epoch') == epoch
            and config.get('selection') == recorded_selection and config.get('checkpoint_sha256') == run['hashes'][epoch],
            'Evaluation identity mismatch')
    rows = read_csv(directory / 'per_image_metrics.csv')
    require(len(rows) == sum(BENCHMARKS.values()), 'Benchmark image count mismatch')
    summaries = read_csv(directory / 'benchmark_means.csv')
    require(len(summaries) == 5 and {r['dataset'] for r in summaries} == set(BENCHMARKS), 'Benchmark summary mismatch')
    result = {}
    for dataset, count in BENCHMARKS.items():
        images = image_rows(rows, dataset, count)
        record = next(r for r in summaries if r['dataset'] == dataset)
        require(int(record['image_count']) == count and int(record['epoch']) == epoch
                and record['selection'] == recorded_selection and record['checkpoint_sha256'] == run['hashes'][epoch],
                'Mixed benchmark checkpoint/selection')
        require(int(record.get('scale', 4)) == 4 and off(record.get('self_ensemble', False)), 'Invalid benchmark evaluation protocol')
        close(record['psnr'], mean(images, 'psnr')); close(record['ssim'], mean(images, 'ssim'))
        result[dataset] = images
    return result

def paired(a, b, dataset, selection, output):
    require([r['filename'] for r in a] == [r['filename'] for r in b], 'Paired image identities differ')
    deltas = np.array([number(y['psnr']) - number(x['psnr']) for x, y in zip(a, b)], dtype=np.float64)
    rng = np.random.default_rng(1)
    boot = deltas[rng.integers(0, len(deltas), size=(2000, len(deltas)))].mean(axis=1)
    for x, y, delta in zip(a, b, deltas):
        output.append(dict(selection=selection, dataset=dataset, filename=x['filename'],
                           B0_psnr=number(x['psnr']), F2_psnr=number(y['psnr']), delta_psnr=float(delta),
                           B0_ssim=number(x['ssim']), F2_ssim=number(y['ssim']),
                           delta_ssim=number(y['ssim']) - number(x['ssim'])))
    return dict(selection=selection, dataset=dataset, image_count=len(a), B0_psnr=mean(a, 'psnr'),
                F2_psnr=mean(b, 'psnr'), delta_psnr=float(deltas.mean()),
                B0_ssim=mean(a, 'ssim'), F2_ssim=mean(b, 'ssim'), delta_ssim=mean(b, 'ssim') - mean(a, 'ssim'),
                median_delta_psnr=float(np.median(deltas)), win_rate=float((deltas > 0).mean()),
                bootstrap95_low=float(np.quantile(boot, .025)), bootstrap95_high=float(np.quantile(boot, .975)))

def summarize(group):
    group = Path(group).resolve()
    manifest = read_json(group / 'protocol.json')
    require(manifest.get('protocol') == PROTOCOL and manifest.get('engineering_only') is False, 'Unregistered/engineering group')
    # Reject paired provenance and inputs before hashing large checkpoints.
    early_configs = {label: read_json(run_directory(group, label) / 'config.json') for label in MODELS}
    for config in early_configs.values():
        checked_config(config, manifest)
    require(early_configs['B0']['initial_unchanged_state_sha256'] == early_configs['F2']['initial_unchanged_state_sha256'], 'Paired scratch initialization differs')
    early_batches = {label: indexed(run_directory(group, label) / 'first_batch_hashes.csv') for label in MODELS}
    require(early_batches['B0'] == early_batches['F2'], 'Paired first batches differ')
    runs = {label: validate_run(group, label, manifest) for label in MODELS}
    require(runs['B0']['config']['initial_unchanged_state_sha256'] == runs['F2']['config']['initial_unchanged_state_sha256'],
            'Paired scratch initialization differs')
    require(runs['B0']['batches'] == runs['F2']['batches'], 'Paired first batches differ')
    paths = read_json(group / 'evaluation_paths.json')
    evaluations = {selection: {label: validate_evaluation(group, label, selection, paths[label][selection], runs[label], manifest, paths[label]['final150'])
                              for label in MODELS} for selection in ('final150', 'best_div2k')}
    comparisons = []
    for epoch in range(1, 151):
        row = {'epoch': epoch}
        for label, run in runs.items():
            row.update({label + '_div2k_psnr': mean(run['raw'][epoch]['DIV2K'], 'psnr'),
                        label + '_div2k_ssim': mean(run['raw'][epoch]['DIV2K'], 'ssim'),
                        label + '_set5_psnr': mean(run['raw'][epoch]['Set5'], 'psnr'),
                        label + '_set5_ssim': mean(run['raw'][epoch]['Set5'], 'ssim'),
                        label + '_train_loss': number(run['metrics'][epoch]['train_loss']),
                        label + '_learning_rate': number(run['metrics'][epoch]['learning_rate'])})
        for metric in ('div2k_psnr', 'div2k_ssim', 'set5_psnr', 'set5_ssim'):
            row['delta_' + metric] = row['F2_' + metric] - row['B0_' + metric]
        comparisons.append(row)
    aggregates = {}
    for name, count in (('final150', 1), ('last5', 5), ('last10', 10)):
        aggregates[name] = {key: sum(r[key] for r in comparisons[-count:]) / count
                            for key in comparisons[-1] if key != 'epoch'}
    final_delta = aggregates['final150']['delta_div2k_psnr']
    last5 = aggregates['last5']['delta_div2k_psnr']
    gate = 'NO_GAIN' if final_delta <= 0 else 'REPEAT_VERIFY' if final_delta < .03 else 'RETAIN' if final_delta < .05 else 'PRIORITY_MULTISEED'
    details, benchmark = [], []
    independent = paired(runs['B0']['raw'][150]['DIV2K'], runs['F2']['raw'][150]['DIV2K'], 'DIV2K', 'final150', details)
    for selection, models in evaluations.items():
        for dataset in BENCHMARKS:
            benchmark.append(paired(models['B0'][dataset], models['F2'][dataset], dataset, selection, details))
    identity = [dict(label=label, model=MODELS[label], run=manifest['runs'][label],
                     best_div2k_epoch=runs[label]['best'], final_checkpoint_sha256=runs[label]['hashes'][150],
                     best_checkpoint_sha256=runs[label]['hashes'][runs[label]['best']]) for label in MODELS]
    report = dict(status='validated_complete150', protocol=PROTOCOL, provenance=manifest, identities=identity,
                  aggregates=aggregates, independent_div2k_final=independent, benchmarks=benchmark, secondary_reused_final={label: runs[label]['best'] == 150 and paths[label]['best_div2k'] == paths[label]['final150'] for label in MODELS},
                  phase150_gate=gate, temporal_consistency=dict(last5_positive=last5 > 0,
                  last10_positive=aggregates['last10']['delta_div2k_psnr'] > 0,
                  note='Fixed150 delta defines the development bucket; last5/10 must still be reviewed.'), bootstrap=dict(seed=1, iterations=2000, unit='paired image',
                  limitation='Image confidence intervals do not establish generalization across training seeds.'),
                  resources_manifest=str(group / 'resources_latest.json') if (group / 'resources_latest.json').exists() else None)
    directory = group / 'comparison_reports' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / 'comparison_metrics.csv', comparisons)
    write_csv(directory / 'benchmark_comparisons.csv', benchmark)
    write_csv(directory / 'per_image_deltas.csv', details)
    write_csv(directory / 'label_identity.csv', identity)
    write_json(directory / 'summary.json', report)
    lines = ['# B0/F2 validated independent cosine150 comparison', '',
             'All 150 checkpoints, independent DIV2K image means, paired batches and benchmark provenance passed.',
             'Primary: fixed epoch150. Secondary: best DIV2K mean, first exact tie. Set5 never selects checkpoints.', '',
             'Phase150 development gate: **' + gate + '**. This is a development threshold, not statistical proof.',
             'Paired image bootstrap: seed1, 2000 replicates. Image intervals do not establish cross-training-seed gains.', '',
             '| Window | DIV2K PSNR delta | DIV2K SSIM delta | Set5 PSNR delta |', '|---|---:|---:|---:|']
    for name, row in aggregates.items():
        lines.append('| {} | {:.8f} | {:.8f} | {:.8f} |'.format(name, row['delta_div2k_psnr'], row['delta_div2k_ssim'], row['delta_set5_psnr']))
    lines.extend(['', '| Selection | Dataset | B0 | F2 | Delta | Median delta | Win rate | Image bootstrap 95% |', '|---|---|---:|---:|---:|---:|---:|---|'])
    for row in [independent] + benchmark:
        lines.append('| {selection} | {dataset} | {B0_psnr:.8f} | {F2_psnr:.8f} | {delta_psnr:.8f} | {median_delta_psnr:.8f} | {win_rate:.4f} | [{bootstrap95_low:.8f}, {bootstrap95_high:.8f}] |'.format(**row))
    lines.extend(['', 'No cross-dataset macro average. No automatic continuation or further training.',
                  'Full epoch table: [comparison_metrics.csv](comparison_metrics.csv).',
                  'Image pairs: [per_image_deltas.csv](per_image_deltas.csv).',
                  'Provenance and identities: [summary.json](summary.json).'])
    (directory / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(group / 'comparison_latest.json', dict(directory=str(directory.relative_to(group)), status=report['status']))
    return report, directory

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    args = parser.parse_args()
    report, directory = summarize(args.group)
    print('VALIDATED_COMPLETE150 {} {}'.format(report['phase150_gate'], directory), flush=True)

if __name__ == '__main__':
    main()
