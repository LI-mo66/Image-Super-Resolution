#!/usr/bin/env python3
"""Zero-training SVD audit for stage-specific N12 gate compression."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
sys.path.insert(0, str(ROOT / 'repro'))

import utility
from audit_srpr_sharing import (
    AuditableSRPRv2, make_dataset, paired_stats, reference_check,
)


FIXED_RANKS = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48)
ENERGY_TARGETS = (0.95, 0.99, 0.995, 0.999)
BASELINE_PARAMETERS = 759627
N12_PARAMETERS = 841563
ORIGINAL_GATE_PARAMETERS = 4800


def factorized_gate_parameters(rank):
    return 147 * int(rank) + 48


def candidate_parameters(ranks):
    return (
        N12_PARAMETERS - 8 * ORIGINAL_GATE_PARAMETERS
        + sum(factorized_gate_parameters(rank) for rank in ranks)
    )


def build_svd_bank(model, ranks=tuple(range(1, 49))):
    originals, approximations, spectra = [], [], []
    for stage, prox in enumerate(model.proximal):
        weight = prox.gate.weight.detach().reshape(48, 99)
        matrix = weight.cpu().double()
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        energy = singular.square()
        cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
        stage_approximations = {}
        for rank in ranks:
            reconstructed = (
                u[:, :rank] * singular[:rank].unsqueeze(0)
            ) @ vh[:rank]
            stage_approximations[int(rank)] = reconstructed.to(
                device=weight.device, dtype=weight.dtype
            ).reshape_as(prox.gate.weight)
        relative_errors = {}
        norm = matrix.norm().clamp_min(1e-30)
        for rank in ranks:
            approximate = stage_approximations[int(rank)].detach().cpu().double()
            relative_errors[str(rank)] = float(
                (approximate.reshape_as(matrix) - matrix).norm() / norm
            )
        originals.append(prox.gate.weight.detach().clone())
        approximations.append(stage_approximations)
        spectra.append({
            'stage': stage + 1,
            'singular_values': [float(value) for value in singular],
            'cumulative_energy': [float(value) for value in cumulative],
            'relative_weight_error_by_rank': relative_errors,
        })
    return originals, approximations, spectra


def ranks_for_energy(spectra, target):
    ranks = []
    for stage in spectra:
        cumulative = stage['cumulative_energy']
        rank = next(
            index + 1 for index, value in enumerate(cumulative)
            if value >= target
        )
        ranks.append(rank)
    return ranks


def apply_rank_plan(model, originals, approximations, ranks):
    if len(ranks) != 8:
        raise ValueError('rank plan must contain eight entries')
    with torch.no_grad():
        for stage, rank in enumerate(ranks):
            source = originals[stage] if rank is None else approximations[stage][int(rank)]
            model.proximal[stage].gate.weight.copy_(source)


def evaluate_rank_plans(
        model, loader, plans, originals, approximations, max_images,
        device, with_ssim):
    rows = {plan['label']: [] for plan in plans}
    started = time.time()
    with torch.inference_mode():
        for image_index, (lr, hr, filename) in enumerate(loader):
            if image_index >= max_images:
                break
            lr, hr = lr.to(device), hr.to(device)
            for plan in plans:
                apply_rank_plan(
                    model, originals, approximations, plan['ranks']
                )
                sr = utility.quantize(model(lr), 255)
                row = {
                    'filename': filename[0],
                    'psnr': float(utility.calc_psnr(
                        sr, hr, 4, 255, dataset=loader
                    )),
                }
                if with_ssim:
                    row['ssim'] = float(utility.calc_ssim(
                        sr, hr, 4, 255, dataset=loader
                    ))
                rows[plan['label']].append(row)
            print(
                f'[{image_index + 1:03d}/{max_images:03d}] '
                f'{len(plans)} rank plans, '
                f'elapsed={(time.time() - started) / 60:.1f} min',
                flush=True,
            )
    apply_rank_plan(model, originals, approximations, [None] * 8)
    if any(len(values) != max_images for values in rows.values()):
        raise RuntimeError('the requested validation images were not evaluated')
    return rows


def compression_metadata(ranks):
    parameters = candidate_parameters(ranks)
    original_gate_total = 8 * ORIGINAL_GATE_PARAMETERS
    compressed_gate_total = sum(
        factorized_gate_parameters(rank) for rank in ranks
    )
    return {
        'ranks': [int(rank) for rank in ranks],
        'candidate_parameters': int(parameters),
        'parameter_overhead_vs_b0': float(
            parameters / BASELINE_PARAMETERS - 1
        ),
        'gate_parameter_reduction': float(
            1 - compressed_gate_total / original_gate_total
        ),
    }


def classify(stats, metadata):
    ci_low = stats['bootstrap_95ci'][0]
    ssim_delta = stats['mean_ssim_delta']
    enough_compression = metadata['gate_parameter_reduction'] >= .30
    if not enough_compression:
        return 'REFERENCE_NO_MEANINGFUL_COMPRESSION'
    if (
        stats['mean_psnr_delta'] >= -.003
        and ci_low >= -.006
        and ssim_delta >= -.00005
    ):
        return 'PROMISING'
    if stats['mean_psnr_delta'] >= -.010:
        return 'MARGINAL'
    return 'REJECT'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference-metrics', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--screen-images', type=int, default=20)
    parser.add_argument('--confirm-images', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.screen_images <= args.confirm_images <= 100:
        raise ValueError('require 1 <= screen <= confirm <= 100')
    for path in (args.checkpoint, args.reference_metrics):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('the full low-rank audit requires CUDA')
    model = AuditableSRPRv2().to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.set_audit_plan()
    originals, approximations, spectra = build_svd_bank(model)

    dataset = make_dataset(args.data_root.resolve())
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    full_rows = evaluate_rank_plans(
        model, loader, [{'label': 'full', 'ranks': [None] * 8}],
        originals, approximations, args.confirm_images, device,
        with_ssim=True,
    )['full']
    reproduction_error = reference_check(
        full_rows, args.reference_metrics.resolve()
    )

    global_plans = [
        {'label': f'fixed_rank_{rank}', 'ranks': [rank] * 8}
        for rank in FIXED_RANKS
    ]
    for target in ENERGY_TARGETS:
        global_plans.append({
            'label': f'energy_{target:.3f}',
            'ranks': ranks_for_energy(spectra, target),
        })
    global_rows = evaluate_rank_plans(
        model, loader, global_plans, originals, approximations,
        args.confirm_images, device, with_ssim=True,
    )
    global_results = {}
    for plan in global_plans:
        label = plan['label']
        stats = paired_stats(global_rows[label], full_rows)
        metadata = compression_metadata(plan['ranks'])
        global_results[label] = {
            **metadata, **stats,
            'decision': classify(stats, metadata),
        }

    stage_plans = []
    for rank in (4, 8, 12):
        for stage in range(8):
            ranks = [None] * 8
            ranks[stage] = rank
            stage_plans.append({
                'label': f'stage_{stage + 1}_rank_{rank}',
                'ranks': ranks,
            })
    stage_rows = evaluate_rank_plans(
        model, loader, stage_plans, originals, approximations,
        args.screen_images, device, with_ssim=False,
    )
    screen_reference = full_rows[:args.screen_images]
    stage_sensitivity = {
        plan['label']: paired_stats(
            stage_rows[plan['label']], screen_reference
        ) for plan in stage_plans
    }

    promising = [
        (label, result) for label, result in global_results.items()
        if result['decision'] == 'PROMISING'
    ]
    selected = None
    if promising:
        label, result = min(
            promising,
            key=lambda item: item[1]['candidate_parameters'],
        )
        selected = {
            'label': label,
            'ranks': result['ranks'],
            'candidate_parameters': result['candidate_parameters'],
            'mean_psnr_delta': result['mean_psnr_delta'],
            'mean_ssim_delta': result['mean_ssim_delta'],
            'bootstrap_95ci': result['bootstrap_95ci'],
        }
    overall_decision = (
        'IMPLEMENT_MINIMAL_LOWRANK_CANDIDATE'
        if selected is not None else
        ('MARGINAL_DIAGNOSE_BEFORE_IMPLEMENTATION'
         if any(
             result['decision'] == 'MARGINAL'
             and result['gate_parameter_reduction'] >= .30
             for result in global_results.values()
         )
         else 'REJECT_LOWRANK_GATE')
    )

    payload = {
        'audit': 'stage-specific SRPR gate low-rank SVD',
        'checkpoint': str(args.checkpoint.resolve()),
        'reference_metrics': str(args.reference_metrics.resolve()),
        'device': str(device),
        'screen_images': args.screen_images,
        'confirm_images': args.confirm_images,
        'reference_max_abs_metric_error': reproduction_error,
        'full_n12': {
            'mean_psnr': float(np.mean([row['psnr'] for row in full_rows])),
            'mean_ssim': float(np.mean([row['ssim'] for row in full_rows])),
        },
        'spectra': spectra,
        'global_results': global_results,
        'stage_sensitivity_20_images': stage_sensitivity,
        'selected_plan': selected,
        'overall_decision': overall_decision,
        'interpretation_warning': (
            'Post-training SVD tests compressibility of learned gates; it does '
            'not guarantee the same outcome for factorized training from scratch.'
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    lines = [
        'SRPR stage-specific gate low-rank audit',
        f'checkpoint={args.checkpoint.resolve()}',
        f'full_n12_psnr={payload["full_n12"]["mean_psnr"]:.6f}',
        f'full_n12_ssim={payload["full_n12"]["mean_ssim"]:.7f}',
        f'reference_max_abs_error={reproduction_error:.9g}',
        '', 'Global rank results:',
    ]
    for label, result in global_results.items():
        lines.append(
            f'{label}: {result["decision"]}; ranks={result["ranks"]}; '
            f'delta={result["mean_psnr_delta"]:+.6f} dB; '
            f'CI=[{result["bootstrap_95ci"][0]:+.6f},'
            f'{result["bootstrap_95ci"][1]:+.6f}]; '
            f'SSIM={result["mean_ssim_delta"]:+.7f}; '
            f'params={result["candidate_parameters"]}; '
            f'gate_reduction={result["gate_parameter_reduction"]:.1%}'
        )
    lines.extend(['', 'Per-stage 20-image sensitivity:'])
    for rank in (4, 8, 12):
        values = [
            stage_sensitivity[f'stage_{stage + 1}_rank_{rank}'][
                'mean_psnr_delta'
            ] for stage in range(8)
        ]
        lines.append(
            f'rank{rank}: ' + ', '.join(
                f'stage{stage + 1}={value:+.6f}'
                for stage, value in enumerate(values)
            )
        )
    lines.extend([
        '', f'overall_decision={overall_decision}',
        f'selected_plan={json.dumps(selected, ensure_ascii=False)}',
    ])
    summary = args.output.with_suffix('.txt')
    summary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'json={args.output}')
    print(f'summary={summary}')


if __name__ == '__main__':
    main()
