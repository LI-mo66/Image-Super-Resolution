#!/usr/bin/env python3
"""Real DIV2K train-step and checkpoint smoke test for N13/SRPRv3."""
import argparse
import io
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnsrprv3 import Net  # noqa: E402


def image_tensor(path):
    array = imageio.imread(path)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    args = parser.parse_args()
    lr_path = args.data_root / 'DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png'
    hr_path = args.data_root / 'DIV2K/DIV2K_train_HR/0001.png'
    if not lr_path.is_file() or not hr_path.is_file():
        raise FileNotFoundError(f'missing DIV2K pair: {lr_path}, {hr_path}')
    lr = image_tensor(lr_path)[..., :32, :32]
    hr = image_tensor(hr_path)[..., :128, :128]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr, hr = lr.to(device), hr.to(device)
    torch.manual_seed(1)
    model = Net(scale=4).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    prediction = model(lr)
    loss = F.l1_loss(prediction, hr)
    optimizer.zero_grad()
    loss.backward()
    for name, parameter in model.named_parameters():
        if name.startswith(('reconstruction.', 'feedback.')) or name in (
                'update_logits', 'feedback_logits'):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    assert torch.isfinite(loss)
    diagnostics = model.diagnostics_snapshot()
    assert diagnostics['update_l2'].min() > 0
    assert diagnostics['feedback_l2'][:3].min() > 0

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = Net(scale=4).to(device).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval()
    with torch.no_grad():
        expected = model(lr)
        actual = restored(lr)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print({
        'status': 'passed', 'device': str(device),
        'loss': float(loss.detach()), 'output_shape': list(prediction.shape),
        'update_l2': diagnostics['update_l2'].tolist(),
        'feedback_l2': diagnostics['feedback_l2'].tolist(),
    })


if __name__ == '__main__':
    main()
