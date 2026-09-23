#!/usr/bin/env python3
"""Persist fixed-crop mechanism diagnostics for a trained N13 checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnsrprv3 import Net  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, default=20)
    args = parser.parse_args()
    checkpoint = args.run / 'model' / f'model_{args.epoch}.pt'
    lr_path = args.data_root / 'DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png'
    if not checkpoint.is_file() or not lr_path.is_file():
        raise FileNotFoundError(f'{checkpoint} or {lr_path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = imageio.imread(lr_path)
    sample = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
    sample = sample.unsqueeze(0)[..., :64, :64].to(device)
    model = Net(scale=4).to(device).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )
    with torch.no_grad():
        output_on = model(sample)
        diagnostics = model.diagnostics_snapshot()
        model.feedback_enabled = False
        output_off = model(sample)
    result = {
        'epoch': args.epoch,
        'checkpoint': str(checkpoint.resolve()),
        'input': str(lr_path.resolve()),
        'feedback_on_off_mean_abs': float((output_on - output_off).abs().mean()),
        'feedback_on_off_max_abs': float((output_on - output_off).abs().max()),
        **{key: value.tolist() for key, value in diagnostics.items()},
    }
    output = args.run / f'mechanism_epoch_{args.epoch:04d}.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
