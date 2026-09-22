#!/usr/bin/env python3
"""N12 real DIV2K one-pair forward/backward/save-reload smoke check."""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnsrprv2 import Net


def load_pair(root, image_id='0001'):
    hr_path = root / 'DIV2K_train_HR' / f'{image_id}.png'
    lr_path = root / 'DIV2K_train_LR_bicubic' / 'X4' / f'{image_id}x4.png'
    if not hr_path.is_file() or not lr_path.is_file():
        raise FileNotFoundError(f'missing pair: {hr_path} / {lr_path}')
    hr = np.asarray(Image.open(hr_path).convert('RGB'), dtype=np.float32)
    lr = np.asarray(Image.open(lr_path).convert('RGB'), dtype=np.float32)
    height = min(lr.shape[0], 64)
    width = min(lr.shape[1], 64)
    height -= height % 4
    width -= width % 4
    if height < 8 or width < 8:
        raise ValueError(f'LR pair is too small: {lr.shape}')
    lr = torch.from_numpy(lr[:height, :width]).permute(2, 0, 1).unsqueeze(0)
    hr = torch.from_numpy(hr[:height * 4, :width * 4]).permute(2, 0, 1).unsqueeze(0)
    return lr / 255.0, hr / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True,
                        help='DIV2K directory containing train HR/LR folders')
    parser.add_argument('--image-id', default='0001')
    args = parser.parse_args()
    torch.manual_seed(1)
    lr, hr = load_pair(args.root.resolve(), args.image_id)
    model = Net(scale=4).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    optimizer.zero_grad(set_to_none=True)
    sr = model(lr)
    loss = F.l1_loss(sr, hr)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(sr).all() and torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    diagnostics = model.diagnostics_snapshot()
    assert diagnostics and all(torch.isfinite(v).all() for v in diagnostics.values())

    model.eval()
    with torch.no_grad():
        reference = model(lr)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model.pt'
        torch.save(model.state_dict(), path)
        reloaded = Net(scale=4).eval()
        reloaded.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        with torch.no_grad():
            restored = reloaded(lr)
        assert torch.equal(reference, restored), 'reload output mismatch'

    print(json.dumps({
        'candidate': 'N12/SRPRv2',
        'image_id': args.image_id,
        'lr_shape': list(lr.shape),
        'hr_shape': list(hr.shape),
        'sr_shape': list(sr.shape),
        'loss': float(loss.detach()),
        'finite_backward': True,
        'diagnostic_stages': len(next(iter(diagnostics.values()))),
        'save_reload': 'passed',
    }, indent=2))


if __name__ == '__main__':
    main()
