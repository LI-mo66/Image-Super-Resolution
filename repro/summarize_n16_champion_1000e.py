#!/usr/bin/env python3
"""Integrity and trajectory summary for the frozen N16 1000-epoch run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (20, 40, 150, 200, 400, 600, 800, 1000)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def curve(path):
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.ndim != 3 or tuple(value.shape[1:]) != (1, 1):
        raise ValueError('{} has unexpected shape {}'.format(path, value.shape))
    result = value[:, 0, 0].double().numpy()
    if len(result) != 1000 or not np.isfinite(result).all():
        raise ValueError('{} is not a finite 1000-epoch curve'.format(path))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    required = (
        run / 'long_run_manifest.json', run / 'model/model_1000.pt',
        run / 'optimizer.pt', run / 'scheduler.pt', run / 'rgcrd_log.pt',
        run / 'psnr_log.pt', run / 'ssim_log.pt',
        run / 'per_image_metrics/epoch_1000.pt',
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    psnr, ssim = curve(run / 'psnr_log.pt'), curve(run / 'ssim_log.pt')
    scheduler = torch.load(
        run / 'scheduler.pt', map_location='cpu', weights_only=True
    )
    if int(scheduler.get('last_epoch', -1)) != 1000:
        raise ValueError('scheduler is not at epoch 1000')
    if set(scheduler.get('milestones')) != {200, 400, 600, 800}:
        raise ValueError('scheduler milestones differ')
    if float(scheduler.get('gamma', -1)) != 0.5:
        raise ValueError('scheduler gamma differs')
    rgcrd = torch.load(
        run / 'rgcrd_log.pt', map_location='cpu', weights_only=True
    )
    rows = rgcrd.get('rows', [])
    if len(rows) != 1000 or int(rows[-1][0]) != 1000:
        raise ValueError('RGCRD log is not complete through epoch 1000')
    image_rows = torch.load(
        run / 'per_image_metrics/epoch_1000.pt',
        map_location='cpu', weights_only=True,
    )
    if len(image_rows) != 100:
        raise ValueError('epoch-1000 DIV2K metrics are not 100 images')
    sys.path.insert(0, str(ROOT / 'LFMN'))
    from model.lfmnsrprv2 import Net as SRPRv2Net
    checkpoint = run / 'model/model_1000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = SRPRv2Net(scale=4)
    model.load_state_dict(state, strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    best_epoch = int(np.argmax(psnr)) + 1
    tail_x = np.arange(901, 1001, dtype=np.float64)
    tail_slope = float(np.polyfit(tail_x, psnr[-100:], 1)[0])
    payload = {
        'candidate': 'N16 frozen champion: SRPRv2+C1',
        'epochs': 1000,
        'checkpoint_selection': 'fixed epoch 1000 endpoint',
        'parameters': parameters,
        'trajectory': {
            str(epoch): {'psnr': float(psnr[epoch - 1]),
                         'ssim': float(ssim[epoch - 1])}
            for epoch in CHECKPOINTS
        },
        'final_psnr': float(psnr[-1]),
        'final_ssim': float(ssim[-1]),
        'best_psnr': float(psnr.max()),
        'best_epoch': best_epoch,
        'last10_psnr_mean': float(psnr[-10:].mean()),
        'last20_psnr_mean': float(psnr[-20:].mean()),
        'last100_psnr_slope_per_epoch': tail_slope,
        'checkpoint_sha256': sha256(checkpoint),
        'strict_checkpoint_load': True,
        'scheduler_integrity': True,
        'rgcrd_rows': len(rows),
        'per_image_count': len(image_rows),
        'decision': 'READY_FOR_FIXED_EPOCH1000_BENCHMARKS',
    }
    (run / 'summary_1000e.json').write_text(
        json.dumps(payload, indent=2) + '\n', encoding='utf-8'
    )
    print('N16 frozen champion 1000-epoch result')
    print('checkpoint_selection=fixed epoch 1000 endpoint')
    for epoch in CHECKPOINTS:
        print('epoch_{:04d}: psnr={:.6f}; ssim={:.7f}'.format(
            epoch, psnr[epoch - 1], ssim[epoch - 1]
        ))
    print('final_psnr={:.6f}'.format(psnr[-1]))
    print('final_ssim={:.7f}'.format(ssim[-1]))
    print('best_psnr={:.6f}@epoch{}'.format(psnr.max(), best_epoch))
    print('last10_psnr_mean={:.6f}'.format(psnr[-10:].mean()))
    print('last20_psnr_mean={:.6f}'.format(psnr[-20:].mean()))
    print('last100_psnr_slope_per_epoch={:+.8f}'.format(tail_slope))
    print('parameters={}'.format(parameters))
    print('checkpoint_sha256={}'.format(payload['checkpoint_sha256']))
    print('decision={}'.format(payload['decision']))


if __name__ == '__main__':
    main()
