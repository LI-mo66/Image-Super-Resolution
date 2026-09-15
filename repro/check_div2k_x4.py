"""Validate the local DIV2K bicubic x4 train/validation pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "DIV2K",
    )
    return parser.parse_args()


def check_split(root: Path, split: str, first: int, last: int) -> None:
    hr_dir = root / f"DIV2K_{split}_HR"
    lr_dir = root / f"DIV2K_{split}_LR_bicubic" / "X4"
    expected = last - first + 1
    hr_files = sorted(hr_dir.glob("*.png"))
    lr_files = sorted(lr_dir.glob("*.png"))
    if len(hr_files) != expected or len(lr_files) != expected:
        raise RuntimeError(
            f"{split}: expected {expected} HR/LR files, got "
            f"{len(hr_files)}/{len(lr_files)}"
        )

    min_lr_w = min_lr_h = 10**9
    for image_id in range(first, last + 1):
        stem = f"{image_id:04d}"
        hr_path = hr_dir / f"{stem}.png"
        lr_path = lr_dir / f"{stem}x4.png"
        if not hr_path.is_file() or not lr_path.is_file():
            raise FileNotFoundError(f"missing pair: {hr_path} / {lr_path}")
        with Image.open(hr_path) as hr_image, Image.open(lr_path) as lr_image:
            hr_image.verify()
            lr_image.verify()
        with Image.open(hr_path) as hr_image, Image.open(lr_path) as lr_image:
            hr_w, hr_h = hr_image.size
            lr_w, lr_h = lr_image.size
        if (hr_w, hr_h) != (lr_w * 4, lr_h * 4):
            raise ValueError(
                f"size mismatch for {stem}: HR={hr_w}x{hr_h}, LR={lr_w}x{lr_h}"
            )
        min_lr_w = min(min_lr_w, lr_w)
        min_lr_h = min(min_lr_h, lr_h)

    print(
        f"{split}: {expected} pairs OK; "
        f"smallest LR dimensions are at least {min_lr_w}x{min_lr_h}"
    )


def main() -> None:
    root = parse_args().root.resolve()
    check_split(root, "train", 1, 800)
    check_split(root, "valid", 801, 900)
    print(f"DIV2K bicubic x4 validation passed: {root}")


if __name__ == "__main__":
    main()
