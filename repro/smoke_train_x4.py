"""Run one real DIV2K x4 optimizer step on the reconstructed LFMN model."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = ROOT / "LFMN"
sys.path.insert(0, str(LFMN_ROOT))

from data.div2k import DIV2K  # noqa: E402
from model.lfmn import Net  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hr-patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU smoke test")
    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    data_args = SimpleNamespace(
        data_range="1-800/801-900",
        dir_data=str(cli.data_root.resolve()),
        scale=[4],
        ext="img",
        batch_size=cli.batch_size,
        test_every=1000,
        patch_size=cli.hr_patch_size,
        n_colors=3,
        rgb_range=255,
        no_augment=False,
        data_train=["DIV2K"],
    )
    dataset = DIV2K(data_args, name="DIV2K", train=True)
    loader = DataLoader(
        dataset,
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    lr, hr, _ = next(iter(loader))
    device = torch.device("cuda")
    model = Net(scale=4).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    lr = lr.to(device, non_blocking=True)
    hr = hr.to(device, non_blocking=True)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    sr = model(lr)
    loss = torch.nn.functional.l1_loss(sr, hr)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"LR/HR/SR: {tuple(lr.shape)} / {tuple(hr.shape)} / {tuple(sr.shape)}")
    print(f"L1 loss: {loss.item():.6f}")
    print(f"one optimizer step: {elapsed:.3f}s; peak allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
