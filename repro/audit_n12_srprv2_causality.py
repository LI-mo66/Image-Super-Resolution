#!/usr/bin/env python3
"""Paired, no-training causal interventions for trained N12/SRPRv2 checkpoints."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LFMN"))

from model.lfmn import Net as BaselineNet  # noqa: E402
from model.lfmnsrprv2 import Net as SRPRv2  # noqa: E402
import utility  # noqa: E402


VARIANTS = (
    "full",
    "no_final_q",
    "no_q",
    "zero_observation",
    "reset_state",
    "no_writeback",
    "state_writeback_only",
    "stateless_writeback_only",
)


class _BenchmarkFlag:
    benchmark = True


class _BenchmarkWrapper:
    dataset = _BenchmarkFlag()


BENCHMARK_WRAPPER = _BenchmarkWrapper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srpr-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="DIV2K")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    require(path.is_file(), f"checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and next(iter(state)).startswith("model."):
        state = {key[6:]: value for key, value in state.items()}
    return state


def load_models(
    srpr_checkpoint: Path,
    baseline_checkpoint: Path,
    device: torch.device,
) -> tuple[SRPRv2, BaselineNet]:
    srpr = SRPRv2(scale=4)
    srpr.load_state_dict(checkpoint_state(srpr_checkpoint), strict=True)
    baseline = BaselineNet(scale=4)
    baseline.load_state_dict(checkpoint_state(baseline_checkpoint), strict=True)
    srpr.to(device).eval().requires_grad_(False)
    baseline.to(device).eval().requires_grad_(False)
    return srpr, baseline


def paired_paths(data_root: Path, dataset: str, limit: int) -> list[tuple[Path, Path]]:
    if dataset.lower() == "div2k":
        hr_dir = data_root / "DIV2K" / "DIV2K_valid_HR"
        lr_dir = data_root / "DIV2K" / "DIV2K_valid_LR_bicubic" / "X4"
        hr_paths = sorted(hr_dir.glob("*.png"))
        pairs = [(lr_dir / f"{path.stem}x4.png", path) for path in hr_paths]
    else:
        aliases = {"bsd100": "B100", "manga109": "manga109"}
        folder = aliases.get(dataset.lower(), dataset)
        base = data_root / "benchmark" / folder
        hr_paths = sorted((base / "HR").glob("*"))
        lr_dir = base / "LR_bicubic" / "X4"
        pairs = []
        for hr_path in hr_paths:
            candidates = sorted(lr_dir.glob(f"{hr_path.stem}x4.*"))
            require(len(candidates) == 1, f"missing or ambiguous LR pair for {hr_path}")
            pairs.append((candidates[0], hr_path))
    pairs = pairs[:limit]
    require(len(pairs) == limit, f"requested {limit} {dataset} pairs, found {len(pairs)}")
    require(all(lr.is_file() and hr.is_file() for lr, hr in pairs), "a paired image is missing")
    return pairs


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def decode(model: SRPRv2, x: torch.Tensor, x0: torch.Tensor, feat: torch.Tensor,
           q: torch.Tensor, include_q: bool) -> torch.Tensor:
    u = model.lrelu(model.pixel_shuffle(model.upconv1(x0 + feat)))
    u = model.lrelu(model.pixel_shuffle(model.upconv2(u)))
    base = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
    output = model.last_conv(u) + base
    if include_q:
        output = output + F.interpolate(q, scale_factor=4, mode="bilinear", align_corners=False)
    return output


def forward_variant(model: SRPRv2, x: torch.Tensor, variant: str) -> torch.Tensor:
    require(variant in VARIANTS, f"unknown intervention: {variant}")
    x0 = model.first_conv(x)
    shallow = model.fea(x)
    feat = x0
    state = torch.zeros_like(feat)
    q = torch.zeros_like(x)
    zero_observation = variant in {"zero_observation", "state_writeback_only",
                                   "stateless_writeback_only"}
    disable_q = variant in {"no_q", "state_writeback_only", "stateless_writeback_only"}
    reset_state = variant in {"reset_state", "stateless_writeback_only"}
    disable_writeback = variant == "no_writeback"

    for index, prox in enumerate(model.proximal):
        previous = feat
        beta, gamma = model.sfmls[index](shallow)
        mixed = beta * previous + gamma
        global_mixer, local_mixer = model.blocks[index]
        transformed = local_mixer(global_mixer(mixed), model.patch_size[index])
        feat = model.esas[index](previous + model.mid_convs[index](transformed))

        if zero_observation:
            observation = torch.zeros_like(x)
        else:
            estimate = model._observation_estimate(x, feat, q)
            residual = x - model.observation.down(estimate)
            backprojection = model.observation.adjoint(residual, estimate.shape[-2:])
            observation = F.interpolate(
                backprojection, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        if reset_state:
            state = torch.zeros_like(state)
        state, delta_feat, delta_q, _ = prox(feat, state, observation)
        if not disable_writeback:
            feat = feat + delta_feat
        if not disable_q:
            q = q + delta_q

    include_q = variant not in {"no_final_q", "no_q", "state_writeback_only",
                                "stateless_writeback_only"}
    return decode(model, x, x0, feat, q, include_q)


def bootstrap_ci(values: list[float], draws: int, seed: int = 20260923) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_rows(rows: list[dict], reference: str, draws: int) -> dict:
    psnr = [row["psnr"] for row in rows]
    ssim = [row["ssim"] for row in rows]
    output = {
        "psnr_mean": float(np.mean(psnr)),
        "ssim_mean": float(np.mean(ssim)),
    }
    if reference:
        psnr_delta = [row["psnr"] - row[f"{reference}_psnr"] for row in rows]
        ssim_delta = [row["ssim"] - row[f"{reference}_ssim"] for row in rows]
        output.update({
            f"psnr_delta_vs_{reference}_mean": float(np.mean(psnr_delta)),
            f"psnr_delta_vs_{reference}_ci95": bootstrap_ci(psnr_delta, draws),
            f"psnr_win_rate_vs_{reference}": float(np.mean(np.asarray(psnr_delta) > 0)),
            f"ssim_delta_vs_{reference}_mean": float(np.mean(ssim_delta)),
        })
    return output


def metric_pair(sr: torch.Tensor, hr: torch.Tensor, benchmark: bool) -> tuple[float, float]:
    sr = utility.quantize(sr, 255)
    wrapper = BENCHMARK_WRAPPER if benchmark else None
    return (
        float(utility.calc_psnr(sr, hr, 4, 255, dataset=wrapper)),
        float(utility.calc_ssim(sr, hr, 4, 255, dataset=wrapper)),
    )


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    srpr, baseline = load_models(args.srpr_checkpoint, args.baseline_checkpoint, device)
    pairs = paired_paths(args.data_root, args.dataset, args.limit)
    benchmark = args.dataset.lower() != "div2k"
    all_rows: dict[str, list[dict]] = {"baseline": []}
    all_rows.update({variant: [] for variant in VARIANTS})
    full_equivalence_max_abs = 0.0

    with torch.inference_mode():
        for index, (lr_path, hr_path) in enumerate(pairs, start=1):
            lr = image_tensor(lr_path).to(device)
            hr = image_tensor(hr_path).to(device)
            hr = hr[..., :lr.shape[-2] * 4, :lr.shape[-1] * 4]
            baseline_output = baseline(lr)
            baseline_psnr, baseline_ssim = metric_pair(baseline_output, hr, benchmark)
            all_rows["baseline"].append({
                "filename": hr_path.stem, "psnr": baseline_psnr, "ssim": baseline_ssim,
            })
            full_output = None
            for variant in VARIANTS:
                output = forward_variant(srpr, lr, variant)
                psnr, ssim = metric_pair(output, hr, benchmark)
                if variant == "full":
                    full_output = output
                    if index == 1:
                        direct_output = srpr(lr)
                        full_equivalence_max_abs = float(
                            (direct_output - full_output).abs().max().item()
                        )
                        require(
                            full_equivalence_max_abs <= 1e-5,
                            "manual full path does not reproduce SRPRv2.forward",
                        )
                require(full_output is not None, "full intervention must run first")
                all_rows[variant].append({
                    "filename": hr_path.stem,
                    "psnr": psnr,
                    "ssim": ssim,
                    "baseline_psnr": baseline_psnr,
                    "baseline_ssim": baseline_ssim,
                    "output_mae_vs_full": float((output - full_output).abs().mean().item()),
                })
            print(f"[{index:03d}/{len(pairs):03d}] {hr_path.stem}", flush=True)

    full_by_name = {row["filename"]: row for row in all_rows["full"]}
    summaries = {"baseline": summarize_rows(all_rows["baseline"], "", args.bootstrap_draws)}
    for variant in VARIANTS:
        rows = all_rows[variant]
        for row in rows:
            full = full_by_name[row["filename"]]
            row["full_psnr"] = full["psnr"]
            row["full_ssim"] = full["ssim"]
        summary = summarize_rows(rows, "baseline", args.bootstrap_draws)
        if variant != "full":
            full_delta = [row["psnr"] - row["full_psnr"] for row in rows]
            summary.update({
                "psnr_delta_vs_full_mean": float(np.mean(full_delta)),
                "psnr_delta_vs_full_ci95": bootstrap_ci(full_delta, args.bootstrap_draws),
                "output_mae_vs_full_mean": float(np.mean([
                    row["output_mae_vs_full"] for row in rows
                ])),
            })
        summaries[variant] = summary

    payload = {
        "mode": "N12_SRPRv2_causal_audit",
        "dataset": args.dataset,
        "images": len(pairs),
        "device": str(device),
        "srpr_checkpoint": str(args.srpr_checkpoint.resolve()),
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "full_equivalence_max_abs": full_equivalence_max_abs,
        "variants": list(VARIANTS),
        "summaries": summaries,
        "per_image": all_rows,
        "notes": [
            "All interventions use the same trained SRPRv2 checkpoint and no retraining.",
            "Ablation-at-inference creates distribution shift and diagnoses necessity, not retrained sufficiency.",
            "DIV2K metrics follow the repository validation protocol; benchmark metrics use Y-channel shave=4.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def print_summary(payload: dict) -> None:
    print(f"dataset={payload['dataset']} images={payload['images']}")
    for name, values in payload["summaries"].items():
        line = f"{name}: PSNR={values['psnr_mean']:.6f} SSIM={values['ssim_mean']:.6f}"
        if name != "baseline":
            line += f" delta_vs_B0={values['psnr_delta_vs_baseline_mean']:+.6f}"
        if name not in {"baseline", "full"}:
            line += f" delta_vs_full={values['psnr_delta_vs_full_mean']:+.6f}"
        print(line)


def main() -> None:
    args = parse_args()
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    payload = run(args)
    print_summary(payload)


if __name__ == "__main__":
    main()
