#!/usr/bin/env python3
"""Zero-training functional audit of the N12/SRPRv2 sharing boundary."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

import utility
from data.div2k import DIV2K
from model.lfmn import Net as BaselineNet
from model.lfmnsrprss import BicubicAdjoint


COMPONENTS = ('candidate', 'gate', 'writeback', 'q_update')


class LRProximalWriteback(nn.Module):
    """Exact module/key layout used by the trained N12 checkpoint."""

    def __init__(self, channels=48, hidden=16):
        super().__init__()
        inputs = channels * 2 + 3
        self.candidate = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.LeakyReLU(0.1, inplace=True), nn.Conv2d(hidden, channels, 1),
        )
        self.gate = nn.Conv2d(inputs, channels, 1)
        self.writeback = nn.Sequential(
            nn.Conv2d(inputs, hidden, 1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        self.q_update = nn.Conv2d(channels + 3, 3, 1)
        nn.init.zeros_(self.writeback[-1].weight)
        nn.init.zeros_(self.writeback[-1].bias)
        nn.init.zeros_(self.q_update.weight)
        nn.init.zeros_(self.q_update.bias)


class AuditableSRPRv2(BaselineNet):
    """N12 forward with read-only component routing interventions."""

    def __init__(self, scale=4, n_feats=48, side_c=32, n_stage=8):
        super().__init__(scale, n_feats, side_c, n_stage, False)
        if int(scale) != 4 or int(n_stage) != 8:
            raise ValueError('the audit requires eight-stage x4 N12')
        self.observation = BicubicAdjoint(scale)
        self.state_channels = n_feats
        self.proximal = nn.ModuleList([
            LRProximalWriteback(n_feats, hidden=16) for _ in range(n_stage)
        ])
        self.stage_probe = nn.Conv2d(n_feats, 3 * scale * scale, 1)
        self.set_audit_plan()

    def set_audit_plan(self, routes=None, ablations=None):
        identity = list(range(8))
        self.audit_routes = {
            name: list((routes or {}).get(name, identity))
            for name in COMPONENTS
        }
        if any(len(route) != 8 for route in self.audit_routes.values()):
            raise ValueError('every component route must contain eight donors')
        self.audit_ablations = {
            name: set((ablations or {}).get(name, ()))
            for name in COMPONENTS
        }

    def _final_decode(self, x, x0, feat, q):
        u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
        u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        base = F.interpolate(
            x, scale_factor=4, mode='bilinear', align_corners=False
        )
        q_hr = F.interpolate(
            q, scale_factor=4, mode='bilinear', align_corners=False
        )
        return self.last_conv(u) + base + q_hr

    def _observation_estimate(self, x, feat, q):
        base = F.interpolate(
            x + q, scale_factor=4, mode='bilinear', align_corners=False
        )
        return base + F.pixel_shuffle(self.stage_probe(feat), 4)

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        state = torch.zeros_like(feat)
        q = torch.zeros(
            x.shape[0], 3, x.shape[2], x.shape[3],
            device=x.device, dtype=x.dtype,
        )
        for stage in range(8):
            prev = feat
            beta, gamma = self.sfmls[stage](fs)
            fm = beta * prev + gamma
            g_attn, l_attn = self.blocks[stage]
            t = g_attn(fm)
            s = l_attn(t, self.patch_size[stage])
            feat = self.esas[stage](prev + self.mid_convs[stage](s))

            x_hat = self._observation_estimate(x, feat, q)
            residual = x - self.observation.down(x_hat)
            back = self.observation.adjoint(residual, x_hat.shape[-2:])
            observation = F.interpolate(
                back, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
            joined = torch.cat((feat, state, observation), dim=1)

            if stage in self.audit_ablations['candidate']:
                candidate = state
            else:
                donor = self.audit_routes['candidate'][stage]
                candidate = self.proximal[donor].candidate(joined)

            if stage in self.audit_ablations['gate']:
                gate = torch.full_like(state, .5)
            else:
                donor = self.audit_routes['gate'][stage]
                gate = torch.sigmoid(self.proximal[donor].gate(joined))
            state = state + gate * (candidate - state)

            correction_input = torch.cat(
                (feat, state, observation), dim=1
            )
            if stage in self.audit_ablations['writeback']:
                delta_feat = torch.zeros_like(feat)
            else:
                donor = self.audit_routes['writeback'][stage]
                delta_feat = self.proximal[donor].writeback(correction_input)

            if stage in self.audit_ablations['q_update']:
                delta_q = torch.zeros_like(q)
            else:
                donor = self.audit_routes['q_update'][stage]
                delta_q = self.proximal[donor].q_update(
                    torch.cat((state, observation), dim=1)
                )
            feat = feat + delta_feat
            q = q + delta_q
        return self._final_decode(x, x0, feat, q)


def make_dataset(data_root):
    args = SimpleNamespace(
        data_range='1-800/801-900', dir_data=str(data_root), scale=[4],
        ext='img', batch_size=1, test_every=1000, patch_size=256,
        n_colors=3, rgb_range=255, no_augment=True,
        data_train=['DIV2K'], test_only=False,
    )
    return DIV2K(args, name='DIV2K', train=False)


def full_plan(label='full'):
    return {'label': label, 'routes': {}, 'ablations': {}}


def donor_plan(component, donor):
    return {
        'label': f'{component}:donor{donor + 1}',
        'routes': {component: [donor] * 8}, 'ablations': {},
    }


def swap_plan(component, name, route):
    return {
        'label': f'{component}:{name}',
        'routes': {component: route}, 'ablations': {},
    }


def ablation_plan(component, stage):
    return {
        'label': f'{component}:ablate_stage{stage + 1}',
        'routes': {}, 'ablations': {component: [stage]},
    }


def combined_plan(label, donor_by_component, components):
    return {
        'label': label,
        'routes': {
            component: [donor_by_component[component]] * 8
            for component in components
        },
        'ablations': {},
    }


def evaluate(model, loader, plans, max_images, device, with_ssim=False):
    rows = {plan['label']: [] for plan in plans}
    started = time.time()
    with torch.inference_mode():
        for image_index, (lr, hr, filename) in enumerate(loader):
            if image_index >= max_images:
                break
            lr, hr = lr.to(device), hr.to(device)
            for plan in plans:
                model.set_audit_plan(plan['routes'], plan['ablations'])
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
            elapsed = time.time() - started
            print(
                f'[{image_index + 1:03d}/{max_images:03d}] '
                f'{len(plans)} plans, elapsed={elapsed / 60:.1f} min',
                flush=True,
            )
    if any(len(value) != max_images for value in rows.values()):
        raise RuntimeError('the requested number of validation images was not evaluated')
    model.set_audit_plan()
    return rows


def paired_stats(candidate, reference, bootstrap_samples=10000):
    reference_by_name = {row['filename']: row for row in reference}
    if {row['filename'] for row in candidate} != set(reference_by_name):
        raise ValueError('paired filenames differ')
    psnr_delta = np.asarray([
        row['psnr'] - reference_by_name[row['filename']]['psnr']
        for row in candidate
    ], dtype=np.float64)
    rng = np.random.default_rng(1)
    draws = rng.choice(
        psnr_delta, (bootstrap_samples, len(psnr_delta)), replace=True
    ).mean(axis=1)
    result = {
        'mean_psnr_delta': float(psnr_delta.mean()),
        'median_psnr_delta': float(np.median(psnr_delta)),
        'win_rate': float((psnr_delta > 0).mean()),
        'bootstrap_95ci': [
            float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975)),
        ],
        'per_image_psnr_delta': {
            row['filename']: float(
                row['psnr'] - reference_by_name[row['filename']]['psnr']
            ) for row in candidate
        },
    }
    if 'ssim' in candidate[0]:
        ssim_delta = np.asarray([
            row['ssim'] - reference_by_name[row['filename']]['ssim']
            for row in candidate
        ])
        result['mean_ssim_delta'] = float(ssim_delta.mean())
        result['ssim_win_rate'] = float((ssim_delta > 0).mean())
    return result


def reference_check(full_rows, reference_path, tolerance=2e-5):
    saved = torch.load(reference_path, map_location='cpu', weights_only=True)
    saved_by_name = {row['filename']: row for row in saved}
    errors = []
    for row in full_rows:
        if row['filename'] not in saved_by_name:
            raise ValueError(f'missing reference image {row["filename"]}')
        errors.append(abs(row['psnr'] - saved_by_name[row['filename']]['psnr']))
        errors.append(abs(row['ssim'] - saved_by_name[row['filename']]['ssim']))
    maximum = max(errors)
    if maximum > tolerance:
        raise ValueError(
            f'N12 reproduction differs from stored metrics: {maximum} > {tolerance}'
        )
    return maximum


def recommendation(delta):
    if delta >= -.003:
        return 'CONSIDER_SHARING'
    if delta >= -.010:
        return 'UNCERTAIN_KEEP_INDEPENDENT_FOR_NOW'
    return 'KEEP_STAGE_SPECIFIC'


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
        raise RuntimeError('the full sharing audit requires CUDA')
    model = AuditableSRPRv2().to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    dataset = make_dataset(args.data_root.resolve())
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )

    # Reproduce the untouched N12 curve before any expensive interventions.
    # A code/data/metric mismatch therefore stops the audit immediately.
    reference_rows = evaluate(
        model, loader, [full_plan()], args.confirm_images, device,
        with_ssim=True,
    )['full']
    reproduction_error = reference_check(
        reference_rows, args.reference_metrics.resolve()
    )

    screen_plans = [full_plan()]
    for component in COMPONENTS:
        for stage in range(8):
            screen_plans.append(ablation_plan(component, stage))
        screen_plans.append(swap_plan(
            component, 'cyclic_next', list(range(1, 8)) + [0]
        ))
        screen_plans.append(swap_plan(
            component, 'repeat_spec_swap', [4, 5, 6, 7, 0, 1, 2, 3]
        ))
    screen = evaluate(
        model, loader, screen_plans, args.screen_images, device,
        with_ssim=False,
    )
    screen_stats = {
        label: paired_stats(rows, screen['full'])
        for label, rows in screen.items() if label != 'full'
    }

    donor_plans = []
    for component in COMPONENTS:
        donor_plans.extend(donor_plan(component, donor) for donor in range(8))
    donors = evaluate(
        model, loader, donor_plans, args.confirm_images, device,
        with_ssim=True,
    )
    donor_stats = {
        label: paired_stats(rows, reference_rows)
        for label, rows in donors.items()
    }
    best_donors = {}
    component_recommendations = {}
    for component in COMPONENTS:
        candidates = [
            (donor, donor_stats[f'{component}:donor{donor + 1}'])
            for donor in range(8)
        ]
        donor, stats = max(candidates, key=lambda item: item[1]['mean_psnr_delta'])
        best_donors[component] = donor
        component_recommendations[component] = {
            'best_donor_stage': donor + 1,
            'best_donor_stats': stats,
            'recommendation': recommendation(stats['mean_psnr_delta']),
            'all_donor_mean_deltas': [
                donor_stats[f'{component}:donor{i + 1}']['mean_psnr_delta']
                for i in range(8)
            ],
        }

    partition_plans = [
        combined_plan(
            'partition:shared_state_candidate_gate', best_donors,
            ('candidate', 'gate'),
        ),
        combined_plan(
            'partition:shared_output_writeback_q', best_donors,
            ('writeback', 'q_update'),
        ),
        combined_plan(
            'partition:all_components_shared', best_donors, COMPONENTS,
        ),
    ]
    partitions = evaluate(
        model, loader, partition_plans, args.confirm_images, device,
        with_ssim=True,
    )
    partition_stats = {
        label: paired_stats(rows, reference_rows)
        for label, rows in partitions.items()
    }

    stage_ablation = {
        component: [
            screen_stats[f'{component}:ablate_stage{stage + 1}'][
                'mean_psnr_delta'
            ] for stage in range(8)
        ] for component in COMPONENTS
    }
    swap_stats = {
        component: {
            mode: screen_stats[f'{component}:{mode}']
            for mode in ('cyclic_next', 'repeat_spec_swap')
        } for component in COMPONENTS
    }
    payload = {
        'audit': 'SRPR sharing boundary',
        'checkpoint': str(args.checkpoint.resolve()),
        'reference_metrics': str(args.reference_metrics.resolve()),
        'device': str(device),
        'screen_images': args.screen_images,
        'confirm_images': args.confirm_images,
        'reference_max_abs_metric_error': reproduction_error,
        'full_n12': {
            'mean_psnr': float(np.mean([row['psnr'] for row in reference_rows])),
            'mean_ssim': float(np.mean([row['ssim'] for row in reference_rows])),
        },
        'component_recommendations': component_recommendations,
        'partition_stats': partition_stats,
        'stage_ablation_screen_mean_psnr_delta': stage_ablation,
        'swap_screen_stats': swap_stats,
        'all_donor_stats': donor_stats,
        'interpretation_warning': (
            'Donor sharing tests trained functional interchangeability; it is '
            'not an estimate of optimally retrained shared weights.'
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

    lines = [
        'SRPR sharing-boundary audit',
        f'checkpoint={args.checkpoint.resolve()}',
        f'full_n12_psnr={payload["full_n12"]["mean_psnr"]:.6f}',
        f'full_n12_ssim={payload["full_n12"]["mean_ssim"]:.7f}',
        f'reference_max_abs_error={reproduction_error:.9g}',
        '', 'Component recommendations:',
    ]
    for component in COMPONENTS:
        item = component_recommendations[component]
        stats = item['best_donor_stats']
        lines.append(
            f'{component}: {item["recommendation"]}; '
            f'best_donor=stage{item["best_donor_stage"]}; '
            f'delta={stats["mean_psnr_delta"]:+.6f} dB; '
            f'CI=[{stats["bootstrap_95ci"][0]:+.6f},'
            f'{stats["bootstrap_95ci"][1]:+.6f}]; '
            f'SSIM={stats["mean_ssim_delta"]:+.7f}'
        )
    lines.extend(['', 'Partition confirmation:'])
    for label, stats in partition_stats.items():
        lines.append(
            f'{label}: delta={stats["mean_psnr_delta"]:+.6f} dB; '
            f'CI=[{stats["bootstrap_95ci"][0]:+.6f},'
            f'{stats["bootstrap_95ci"][1]:+.6f}]; '
            f'SSIM={stats["mean_ssim_delta"]:+.7f}'
        )
    lines.extend(['', 'Stage-ablation screen deltas:'])
    for component, values in stage_ablation.items():
        lines.append(
            f'{component}: ' + ', '.join(
                f'stage{i + 1}={value:+.6f}'
                for i, value in enumerate(values)
            )
        )
    summary_path = args.output.with_suffix('.txt')
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'json={args.output}')
    print(f'summary={summary_path}')


if __name__ == '__main__':
    main()
