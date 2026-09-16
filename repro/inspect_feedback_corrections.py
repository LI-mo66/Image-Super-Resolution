#!/usr/bin/env python3
"""Measure learned feedback correction magnitudes on real LR images."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = PROJECT_ROOT / 'LFMN'
sys.path.insert(0, str(LFMN_ROOT))

from model.lfmnfeedback import Net as FeedbackNet  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--image_dir',
        type=Path,
        default=PROJECT_ROOT / 'datasets' / 'DIV2K' / 'DIV2K_valid_LR_bicubic' / 'X4',
    )
    parser.add_argument('--start', type=int, default=801)
    parser.add_argument('--end', type=int, default=810)
    parser.add_argument('--feedback_stages', type=str, default='3-5-7')
    parser.add_argument('--feedback_mid', type=int, default=8)
    parser.add_argument('--feedback_scale', type=float, default=0.1)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def parse_stages(value):
    return tuple(int(item) for item in value.replace(',', '-').split('-') if item)


def load_rgb_tensor(path, device):
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32)
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)


def find_image(image_dir, number):
    candidates = [
        image_dir / '{:04d}x4.png'.format(number),
        image_dir / '{:04d}.png'.format(number),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(image_dir.glob('{:04d}*'.format(number)))
    if matches:
        return matches[0]
    raise FileNotFoundError('no LR image found for {:04d} under {}'.format(number, image_dir))


def main():
    args = parse_args()
    if args.start > args.end:
        raise ValueError('--start must not exceed --end')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    stages = parse_stages(args.feedback_stages)
    model = FeedbackNet(
        scale=4,
        feedback_stages=stages,
        feedback_mid=args.feedback_mid,
        feedback_scale=args.feedback_scale,
    )
    try:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    stats = defaultdict(lambda: {
        'beta_abs_sum': 0.0,
        'gamma_abs_sum': 0.0,
        'beta_sq_sum': 0.0,
        'gamma_sq_sum': 0.0,
        'beta_max': 0.0,
        'gamma_max': 0.0,
        'count': 0,
    })
    handles = []

    def make_hook(stage):
        def hook(_module, _inputs, outputs):
            delta_beta, delta_gamma = outputs
            item = stats[stage]
            item['beta_abs_sum'] += delta_beta.abs().sum().item()
            item['gamma_abs_sum'] += delta_gamma.abs().sum().item()
            item['beta_sq_sum'] += delta_beta.square().sum().item()
            item['gamma_sq_sum'] += delta_gamma.square().sum().item()
            item['beta_max'] = max(item['beta_max'], delta_beta.abs().max().item())
            item['gamma_max'] = max(item['gamma_max'], delta_gamma.abs().max().item())
            item['count'] += delta_beta.numel()
        return hook

    for stage in stages:
        handles.append(model.feedback[str(stage)].register_forward_hook(make_hook(stage)))

    image_paths = [find_image(args.image_dir, number) for number in range(args.start, args.end + 1)]
    with torch.inference_mode():
        for index, path in enumerate(image_paths, start=1):
            model(load_rgb_tensor(path, device))
            print('[{}/{}] {}'.format(index, len(image_paths), path.name))

    for handle in handles:
        handle.remove()

    print('\ncheckpoint: {}'.format(args.checkpoint))
    print('device: {}; images: {}-{}'.format(device, args.start, args.end))
    print('stage,weight_abs_mean,weight_abs_max,delta_beta_mae,delta_beta_rms,'
          'delta_beta_max,delta_gamma_mae,delta_gamma_rms,delta_gamma_max')
    for stage in stages:
        branch = model.feedback[str(stage)]
        weight = branch.expand.weight.detach()
        item = stats[stage]
        count = item['count']
        print(
            '{},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}'.format(
                stage,
                weight.abs().mean().item(),
                weight.abs().max().item(),
                item['beta_abs_sum'] / count,
                (item['beta_sq_sum'] / count) ** 0.5,
                item['beta_max'],
                item['gamma_abs_sum'] / count,
                (item['gamma_sq_sum'] / count) ** 0.5,
                item['gamma_max'],
            )
        )


if __name__ == '__main__':
    main()
