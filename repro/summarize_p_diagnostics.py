"""Strict paired fixed-checkpoint diagnostic reports; no training conclusions."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

METRICS = ('psnr_y_quantized', 'ssim_y_quantized', 'mse_y_continuous', 'mse_rgb_continuous')
COMMON = ('checkpoint_sha256', 'model_sha256', 'runner_sha256', 'metric_source_sha256',
          'metric_protocol', 'region_protocol', 'scale', 'crop_lr', 'scope', 'div2k_ids',
          'benchmarks', 'data_root', 'seed', 'device', 'gpu', 'torch', 'precision',
          'self_ensemble', 'chop', 'normalize_overlap', 'eval_refine_iters', 'baseline_source_commit')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None

def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

def stats(values, bootstrap=False):
    if not values:
        return {'images': 0}
    result = {'images': len(values), 'mean': mean(values), 'median': statistics.median(values)}
    if bootstrap:
        rng = random.Random(20261010)
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(5000))
        result.update(win_rate=sum(v > 0 for v in values) / len(values),
                      zero_rate=sum(v == 0 for v in values) / len(values),
                      bootstrap95_ci=[samples[124], samples[4874]],
                      bootstrap_seed=20261010, bootstrap_replicates=5000,
                      statistical_unit='image; conditional on this fixed checkpoint')
    return result

def identity(row):
    return (row['dataset'], row['split'], row['image'], json.dumps(row['alignment'], sort_keys=True))

def validate_baseline(row, reference):
    for key in (*METRICS, 'hr_sha256', 'lr_sha256', 'alignment', 'regions'):
        if key not in row or key not in reference or row[key] != reference[key]:
            raise ValueError(f'P0 mismatch for {identity(row)}: {key}')

def flatten_numbers(value, prefix=''):
    result = {}
    if numeric(value):
        result[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in ('query', 'lr_y', 'lr_x', 'stage'):
                result.update(flatten_numbers(child, f'{prefix}.{key}' if prefix else key))
    elif isinstance(value, list) and all(v is None or numeric(v) for v in value):
        result[prefix + '.head_mean'] = mean(value)
    return result

def stage_report(rows):
    stages = defaultdict(list)
    for row in rows:
        probe = row.get('probe', {})
        for stage, data in probe.get('stages', {}).items():
            per_query = defaultdict(list)
            for query in data.get('queries', []):
                for key, value in flatten_numbers(query).items():
                    if value is not None:
                        per_query[key].append(value)
            image = {key: mean(values) for key, values in per_query.items()}
            image['eligible_query_count'] = data.get('eligible_query_count')
            image['pearson_incompatible_mass_vs_final_error'] = data.get('pearson_incompatible_mass_vs_final_error')
            stages[stage].append(image)
    return {stage: {'images': len(images), 'image_equal_weight_statistics': {
        key: stats([image[key] for image in images if numeric(image.get(key))])
        for key in sorted(set().union(*(image.keys() for image in images)))},
        'interpretation': 'Candidate slots and attention masses are observations; correlation is not PSNR gain. Queries/heads are not independent images.'}
        for stage, images in sorted(stages.items())}

def summarize(run_paths):
    partitions = defaultdict(list)
    for path in run_paths:
        path = Path(path)
        config = load(path / 'config.json')
        if config.get('status') != 'completed' or config.get('check_only'):
            raise ValueError(f'Only completed actual-data runs can be summarized: {path}')
        missing = [key for key in COMMON if key not in config]
        if missing:
            raise ValueError(f'Missing protocol fields in {path}: {missing}')
        rows = load(path / 'per_image.json')
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'Empty/invalid rows in {path}')
        partition = (config['scale'], config['crop_lr'], config['scope'])
        partitions[partition].append((str(path), config, rows))
    output = {'evidence_scope': 'Fixed-checkpoint diagnosis only; no trained structural gain or GO claim.', 'groups': []}
    for partition, runs in partitions.items():
        reference_config = runs[0][1]
        for path, config, rows in runs:
            for key in COMMON:
                if config[key] != reference_config[key]:
                    raise ValueError(f'Protocol mismatch {key}: {path}')
            seen = set()
            for row in rows:
                k = (identity(row), row['variant'])
                if k in seen:
                    raise ValueError(f'Duplicate row in {path}: {k}')
                seen.add(k)
                if any(not numeric(row.get(key)) for key in METRICS):
                    raise ValueError(f'Nonfinite/missing metric: {path} {k}')
        independent = {}
        for path, config, rows in runs:
            if config['scheme'] == 'P0':
                for row in rows:
                    if row['variant'] != 'p0':
                        raise ValueError('Independent P0 contains interventions')
                    key = identity(row)
                    if key in independent:
                        validate_baseline(row, independent[key])
                    independent[key] = row
        if not independent:
            raise ValueError('An independent P0 run is required')
        group = {'protocol': {key: reference_config[key] for key in COMMON}, 'runs': [r[0] for r in runs],
                 'p0': [], 'p1': [], 'p2': [], 'p1_sfml_observations': [],
                 'device_note': 'GPU architecture is matched; this does not assert bitwise determinism.'}
        strata = defaultdict(list)
        for row in independent.values():
            strata[(row['dataset'], row['split'])].append(row)
        for (dataset, split), rows in sorted(strata.items()):
            group['p0'].append({'dataset': dataset, 'split': split,
                'scope_label': 'coverage smoke; not complete benchmark scores' if split == 'benchmark_smoke' else reference_config['scope'],
                'metrics': {key: stats([r[key] for r in rows]) for key in METRICS}})
        for path, config, rows in runs:
            if config['scheme'] == 'P0':
                continue
            bases = {identity(row): row for row in rows if row['variant'] == 'p0'}
            for row in rows:
                key = identity(row)
                if key not in bases or key not in independent:
                    raise ValueError(f'Missing paired independent P0: {path} {key}')
                validate_baseline(bases[key], independent[key])
                for field in ('hr_sha256', 'lr_sha256', 'alignment'):
                    if row[field] != bases[key][field]:
                        raise ValueError(f'Intervention input mismatch: {field}')
            variant_keys = defaultdict(set)
            for row in rows:
                variant_keys[row['variant']].add(identity(row))
            expected_variants = {'P1': {'p0', 'beta_minus', 'beta_plus', 'gamma_minus', 'gamma_plus'}, 'P2': {'p0'}}
            if set(variant_keys) != expected_variants.get(config['scheme'], set()):
                raise ValueError(f'Missing/unexpected complete variants: {path}')
            if set(bases) != set(independent):
                raise ValueError(f'Run does not cover the full independent P0 image set: {path}')
            if any(keys != set(bases) for keys in variant_keys.values()):
                raise ValueError(f'Variants do not cover identical image sets: {path}')
            paired = defaultdict(list)
            for row in rows:
                paired[(row['dataset'], row['split'], row['variant'])].append(row)
            if config['scheme'] == 'P1':
                for (dataset, split, variant), subset in sorted(paired.items()):
                    by_stage = defaultdict(list)
                    for row in subset:
                        stages = row.get('probe', {}).get('stages', [])
                        if not isinstance(stages, list) or not stages:
                            raise ValueError(f'Missing P1 SFML stage observations: {path}')
                        if len({stage['stage'] for stage in stages}) != len(stages):
                            raise ValueError('Duplicate P1 stage')
                        for stage in stages:
                            by_stage[stage['stage']].append(flatten_numbers(stage))
                    for stage, images in sorted(by_stage.items()):
                        if len(images) != len(subset):
                            raise ValueError('Incomplete P1 stages across images')
                        keys = sorted(set().union(*(image.keys() for image in images)))
                        group['p1_sfml_observations'].append({
                            'run': path, 'dataset': dataset, 'split': split, 'variant': variant,
                            'stage': stage, 'images': len(images),
                            'image_equal_weight_statistics': {key: stats([image[key] for image in images if numeric(image.get(key))]) for key in keys},
                            'interpretation': 'Per-image stage/region descriptors averaged equally. Averaged p05/p50/p95 are approximate descriptor summaries, not pooled quantiles or significance evidence. Sorting changes refer to permutation slots, not cluster membership.'})
                    if variant == 'p0':
                        continue
                    deltas = [r['psnr_y_quantized'] - bases[identity(r)]['psnr_y_quantized'] for r in subset]
                    biggest = max(range(len(deltas)), key=lambda i: abs(deltas[i]))
                    result = {'run': path, 'dataset': dataset, 'split': split, 'variant': variant,
                        'psnr_delta_db': stats(deltas, True),
                        'ssim_delta': stats([r['ssim_y_quantized'] - bases[identity(r)]['ssim_y_quantized'] for r in subset], True),
                        'largest_influence_image': {'image': subset[biggest]['image'], 'delta_db': deltas[biggest],
                            'leave_one_out_mean_db': mean([v for i,v in enumerate(deltas) if i != biggest])}, 'regions': {}}
                    for region in sorted(subset[0]['regions']):
                        valid, coverage = [], []
                        for row in subset:
                            a, b = row['regions'][region], bases[identity(row)]['regions'][region]
                            if a['pixels'] != b['pixels']:
                                raise ValueError('Region coverage mismatch')
                            if a['pixels']:
                                valid.append((a,b)); coverage.append(a['pixels'])
                        result['regions'][region] = {'covered_images': len(valid), 'total_pixels': sum(coverage),
                            'pixels_per_image': stats(coverage), 'equal_image_weight_mse_delta': {
                                key: stats([a[key]-b[key] for a,b in valid], True)
                                for key in ('y_mse_continuous', 'y_mse_quantized')},
                            'mse_improvement_rate': {key: sum(a[key] < b[key] for a,b in valid) / len(valid) if valid else None for key in ('y_mse_continuous', 'y_mse_quantized')},
                            'note': 'MSE delta <0 means improvement; reported win_rate counts positive deltas, not improvements.'}
                    group['p1'].append(result)
            elif config['scheme'] == 'P2':
                for (dataset, split, variant), subset in sorted(paired.items()):
                    group['p2'].append({'run': path, 'dataset': dataset, 'split': split,
                        'variant': variant, 'stages': stage_report(subset)})
            else:
                raise ValueError('Unknown diagnostic scheme')
        output['groups'].append(group)
    return output

def markdown(report):
    lines = ['# P0/P1/P2 固定权重诊断', '', report['evidence_scope'], '',
             '选择集、检验集、benchmark smoke 分开报告；bootstrap 按图片抽样，不能证明跨训练 seed 稳定性。', '']
    for group in report['groups']:
        p = group['protocol']
        lines += [f"## ×{p['scale']} / LR crop {p['crop_lr']} / {p['scope']}", '', '### P0', '',
                  '| 数据集 | split | 图片数 | PSNR均值 | SSIM均值 |', '|---|---|---:|---:|---:|']
        for row in group['p0']:
            m = row['metrics']
            lines.append(f"| {row['dataset']} | {row['split']} | {m[METRICS[0]]['images']} | {m[METRICS[0]]['mean']:.9f} | {m[METRICS[1]]['mean']:.9f} |")
        lines += ['', 'benchmark_smoke 仅为覆盖烟雾，不是完整 benchmark 成绩。', '', '### P1', '',
                  '| 数据集 / split | variant | ΔPSNR均值 / 中位数 | 胜率 | 图片bootstrap95%CI | 最大影响图 |', '|---|---|---:|---:|---|---|']
        for row in group['p1']:
            d = row['psnr_delta_db']; b = row['largest_influence_image']
            lines.append(f"| {row['dataset']} / {row['split']} | {row['variant']} | {d['mean']:.9f} / {d['median']:.9f} | {d['win_rate']:.3f} | {d['bootstrap95_ci']} | {b['image']} ({b['delta_db']:+.9f}) |")
        lines += ['', '区域配对 MSE、覆盖图片数和像素数见 JSON；均值按图片等权。', '',
                  '### P1 SFML observations', '',
                  '| 数据集 / split | variant | stage | multiplicative / prev RMS (all) | gamma / prev RMS (all) | 排序槽位变化率 | beta越界率 |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for row in group['p1_sfml_observations']:
            values = row['image_equal_weight_statistics']
            def field(key):
                value = values.get(key, {}).get('mean')
                return f'{value:.9f}' if value is not None else '—'
            lines.append(f"| {row['dataset']} / {row['split']} | {row['variant']} | {row['stage']} | {field('regions.all.multiplicative_to_prev_rms')} | {field('regions.all.gamma_to_prev_rms')} | {field('sorting_permutation_changed_fraction')} | {field('beta_outside_01_fraction')} |")
        lines += ['', 'SFML 分位数为逐图描述分位数的等权均值，不是合并像素分位数或显著性证据。排序变化指排列槽位，不等于聚类成员变化。相同 GPU 架构不代表位级确定性。', '', '### P2', '']
        for row in group['p2']:
            lines += [f"{row['dataset']} / {row['split']}", '']
            for stage, data in row['stages'].items():
                lines += [f"- stage {stage}: {data['images']} 张图片；", '']
                for key, value in data['image_equal_weight_statistics'].items():
                    lines.append(f"  {key}: mean={value.get('mean')} (有效图片数={value['images']})")
                lines.append('')
        lines += ['P2 候选组成、attention质量及其与最终误差的相关性是观察量，不是 PSNR 收益，也不能据此确认瓶颈。', '']
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True, help='Ignored JSON output path; sibling .md is generated')
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.suffix != '.json':
        parser.error('--output must end in .json')
    for path in (target, target.with_suffix('.md')):
        ignored = subprocess.run(['git', 'check-ignore', '--quiet', str(path)], cwd=Path(__file__).resolve().parents[1])
        if ignored.returncode != 0:
            parser.error('Reports must be written to a git-ignored output path')
    report = summarize(args.runs)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    target.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
    print(target)

if __name__ == '__main__':
    main()
