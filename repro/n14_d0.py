#!/usr/bin/env python3
"""N14/CPES D0 diagnostics for a frozen LFMN baseline.

This script never changes the baseline parameters.  It provides three modes:

* smoke: checkpoint, trace, gradient isolation, and tiny real-pair checks;
* operators: compare TABs through responses in the common LR feature space;
* probes: train equal-budget frozen-backbone probes for D1 and R3.

Outputs are diagnostic evidence, not performance claims for CPES-SR.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LFMN"))

from model.lfmn import Net as LFMN  # noqa: E402


PAIR_INDICES = ((0, 1), (2, 3), (4, 5), (6, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "operators", "probes"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch", type=int, default=32, help="LR crop size")
    parser.add_argument("--operator-images", type=int, default=20)
    parser.add_argument("--train-images", type=int, default=800)
    parser.add_argument("--val-images", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-crops", type=int, default=2)
    parser.add_argument("--probe-lr", type=float, default=2e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-3)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--ranks", default="8,16,24,32")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)


def load_model(checkpoint: Path, device: torch.device) -> LFMN:
    require(checkpoint.is_file(), f"checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and next(iter(state)).startswith("model."):
        state = {key[6:]: value for key, value in state.items()}
    model = LFMN(scale=4)
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model


def trace_backbone(model: LFMN, image: torch.Tensor) -> dict[str, object]:
    x0 = model.first_conv(image)
    shallow = model.fea(image)
    feature = x0
    stages = []
    tab_inputs = []
    tab_outputs = []
    tab_residuals = []
    for index, block in enumerate(model.blocks):
        previous = feature
        beta, gamma = model.sfmls[index](shallow)
        modulated = beta * previous + gamma
        global_mixer, local_mixer = block
        tab_output = global_mixer(modulated)
        transformed = local_mixer(tab_output, model.patch_size[index])
        feature = model.esas[index](previous + model.mid_convs[index](transformed))
        stages.append(feature)
        tab_inputs.append(modulated)
        tab_outputs.append(tab_output)
        tab_residuals.append(tab_output - modulated)
    return {
        "x0": x0,
        "shallow": shallow,
        "stages": stages,
        "tab_inputs": tab_inputs,
        "tab_outputs": tab_outputs,
        "tab_residuals": tab_residuals,
    }


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def paired_paths(data_root: Path, split: str, limit: int) -> list[tuple[Path, Path]]:
    base = data_root / "DIV2K"
    hr_dir = base / f"DIV2K_{split}_HR"
    lr_dir = base / f"DIV2K_{split}_LR_bicubic" / "X4"
    hr_paths = sorted(hr_dir.glob("*.png"))[:limit]
    pairs = []
    for hr_path in hr_paths:
        lr_path = lr_dir / f"{hr_path.stem}x4.png"
        require(lr_path.is_file(), f"missing LR pair: {lr_path}")
        pairs.append((lr_path, hr_path))
    require(len(pairs) == limit, f"requested {limit} {split} pairs, found {len(pairs)}")
    return pairs


def crop_pair(
    lr: torch.Tensor,
    hr: torch.Tensor,
    patch: int,
    rng: random.Random,
    deterministic_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = lr.shape
    require(height >= patch and width >= patch, "LR image is smaller than requested crop")
    if deterministic_index is None:
        top = rng.randrange(height - patch + 1)
        left = rng.randrange(width - patch + 1)
    else:
        positions = (
            ((height - patch) // 2, (width - patch) // 2),
            (0, 0),
            (height - patch, width - patch),
            (0, width - patch),
            (height - patch, 0),
        )
        top, left = positions[deterministic_index % len(positions)]
    lr_crop = lr[:, top:top + patch, left:left + patch]
    hr_crop = hr[:, 4 * top:4 * (top + patch), 4 * left:4 * (left + patch)]
    return lr_crop, hr_crop


def sample_batch(
    pairs: list[tuple[Path, Path]],
    patch: int,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    lr_crops, hr_crops = [], []
    for _ in range(batch_size):
        lr_path, hr_path = pairs[rng.randrange(len(pairs))]
        lr, hr = image_tensor(lr_path), image_tensor(hr_path)
        lr_crop, hr_crop = crop_pair(lr, hr, patch, rng)
        if rng.random() < 0.5:
            lr_crop, hr_crop = lr_crop.flip(-1), hr_crop.flip(-1)
        if rng.random() < 0.5:
            lr_crop, hr_crop = lr_crop.flip(-2), hr_crop.flip(-2)
        lr_crops.append(lr_crop)
        hr_crops.append(hr_crop)
    return torch.stack(lr_crops), torch.stack(hr_crops)


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.flatten(2).transpose(1, 2).reshape(-1, left.shape[1]).float()
    right = right.flatten(2).transpose(1, 2).reshape(-1, right.shape[1]).float()
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    cross = left.T @ right
    numerator = cross.square().sum()
    denominator = torch.sqrt((left.T @ left).square().sum() * (right.T @ right).square().sum())
    return float((numerator / denominator.clamp_min(1e-12)).item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten().float(), right.flatten().float(), dim=0).item())


def topk_jaccard(left: torch.Tensor, right: torch.Tensor, fraction: float = 0.1) -> float:
    left = left.square().mean(1).flatten(1)
    right = right.square().mean(1).flatten(1)
    count = max(1, int(left.shape[1] * fraction))
    scores = []
    for first, second in zip(left, right):
        first_set = set(first.topk(count).indices.cpu().tolist())
        second_set = set(second.topk(count).indices.cpu().tolist())
        scores.append(len(first_set & second_set) / max(1, len(first_set | second_set)))
    return float(np.mean(scores))


def standardized(feature: torch.Tensor) -> torch.Tensor:
    mean = feature.mean(dim=(-2, -1), keepdim=True)
    std = feature.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (feature - mean) / std


def low_rank_errors(
    first: torch.Tensor,
    second: torch.Tensor,
    ranks: Iterable[int],
) -> dict[str, dict[str, float]]:
    results: dict[str, list[tuple[float, float]]] = {str(rank): [] for rank in ranks}
    for sample_first, sample_second in zip(first, second):
        matrices = [item.flatten(1).T.float() for item in (sample_first, sample_second)]
        shared_matrix = torch.cat(matrices, dim=1)
        shared_u = torch.linalg.svd(shared_matrix, full_matrices=False).U
        for rank in ranks:
            effective_rank = min(rank, shared_u.shape[1], matrices[0].shape[0])
            shared_basis = shared_u[:, :effective_rank]
            independent_errors = []
            shared_errors = []
            for matrix in matrices:
                norm = torch.linalg.vector_norm(matrix).clamp_min(1e-12)
                singular = torch.linalg.svdvals(matrix)
                tail = singular[effective_rank:]
                independent_errors.append(float(torch.sqrt(tail.square().sum()).div(norm).item()))
                reconstructed = shared_basis @ (shared_basis.T @ matrix)
                shared_errors.append(float(torch.linalg.vector_norm(matrix - reconstructed).div(norm).item()))
            results[str(rank)].append((float(np.mean(independent_errors)), float(np.mean(shared_errors))))
    output = {}
    for rank, values in results.items():
        independent = float(np.mean([value[0] for value in values]))
        shared = float(np.mean([value[1] for value in values]))
        output[rank] = {
            "independent_error": independent,
            "shared_error": shared,
            "shared_over_independent": shared / max(independent, 1e-12),
        }
    return output


class LinearResidualProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Conv2d(48, 3 * 16, 3, padding=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.project(feature), 4)


class ResponsePredictor(nn.Module):
    """Equal-capacity predictor used for both static and dynamic inputs."""

    def __init__(self, input_channels: int = 40, hidden: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.body(feature)


def highpass(image: torch.Tensor) -> torch.Tensor:
    kernel = image.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(image.shape[1], 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=image.shape[1])


def bootstrap_ci(values: list[float], seed: int = 20260923, draws: int = 2000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_smoke(args: argparse.Namespace, model: LFMN, device: torch.device) -> dict:
    train_pair = paired_paths(args.data_root, "train", 1)[0]
    lr, hr = image_tensor(train_pair[0]), image_tensor(train_pair[1])
    lr, hr = crop_pair(lr, hr, args.patch, random.Random(1), deterministic_index=0)
    lr, hr = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
    with torch.no_grad():
        trace = trace_backbone(model, lr)
        output = model(lr)
    require(output.shape == hr.shape, "baseline output/HR shape mismatch")
    require(len(trace["stages"]) == 8, "expected eight stage features")
    for collection in ("stages", "tab_inputs", "tab_outputs", "tab_residuals"):
        require(all(item.shape == (1, 48, args.patch, args.patch) for item in trace[collection]),
                f"unexpected trace shape in {collection}")
        require(all(torch.isfinite(item).all() for item in trace[collection]),
                f"non-finite trace in {collection}")

    probes = nn.ModuleList([LinearResidualProbe() for _ in range(8)]).to(device)
    target = hr - F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
    prediction = probes[0](trace["stages"][0].detach())
    F.l1_loss(prediction, target).backward()
    require(probes[0].project.weight.grad is not None, "probe gradient is missing")
    require(all(parameter.grad is None for parameter in model.parameters()),
            "frozen baseline unexpectedly received gradients")
    payload = {
        "mode": "smoke",
        "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_shape": list(lr.shape),
        "output_shape": list(output.shape),
        "baseline_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "probe_parameters_each": sum(parameter.numel() for parameter in probes[0].parameters()),
        "probe_gradient_sum": float(probes[0].project.weight.grad.abs().sum().item()),
        "baseline_gradients_present": False,
        "status": "passed",
    }
    save_json(args.output, payload)
    return payload


def run_operators(args: argparse.Namespace, model: LFMN, device: torch.device) -> dict:
    pairs = paired_paths(args.data_root, "valid", args.operator_images)
    ranks = tuple(int(value) for value in args.ranks.split(","))
    per_pair: dict[str, dict[str, list]] = {
        str(index + 1): {
            "native_cka": [], "native_topk_jaccard": [], "common_cka": [],
            "common_cosine": [], "jacobian_noise_cosine": [],
            "jacobian_highpass_cosine": [], "basis": [],
        }
        for index in range(4)
    }
    generator = torch.Generator(device=device).manual_seed(20260923)
    rng = random.Random(20260923)
    with torch.inference_mode():
        for image_index, (lr_path, hr_path) in enumerate(pairs):
            lr, hr = image_tensor(lr_path), image_tensor(hr_path)
            lr, _ = crop_pair(lr, hr, args.patch, rng, deterministic_index=image_index)
            lr = lr.unsqueeze(0).to(device)
            trace = trace_backbone(model, lr)
            for pair_index, (source_index, receiver_index) in enumerate(PAIR_INDICES):
                source_residual = trace["tab_residuals"][source_index]
                receiver_residual = trace["tab_residuals"][receiver_index]
                metrics = per_pair[str(pair_index + 1)]
                metrics["native_cka"].append(linear_cka(source_residual, receiver_residual))
                metrics["native_topk_jaccard"].append(topk_jaccard(source_residual, receiver_residual))

                source_input = standardized(trace["tab_inputs"][source_index])
                receiver_input = standardized(trace["tab_inputs"][receiver_index])
                common = 0.5 * (source_input + receiver_input)
                source_tab = model.blocks[source_index][0]
                receiver_tab = model.blocks[receiver_index][0]
                source_common = source_tab(common) - common
                receiver_common = receiver_tab(common) - common
                metrics["common_cka"].append(linear_cka(source_common, receiver_common))
                metrics["common_cosine"].append(cosine(source_common, receiver_common))
                metrics["basis"].append(low_rank_errors(source_common, receiver_common, ranks))

                noise = torch.randn(common.shape, generator=generator, device=device, dtype=common.dtype)
                noise = noise / noise.square().mean().sqrt().clamp_min(1e-6)
                spatial = highpass(common)
                spatial = spatial / spatial.square().mean().sqrt().clamp_min(1e-6)
                epsilon = 1e-2
                for name, perturbation in (("noise", noise), ("highpass", spatial)):
                    source_response = source_tab(common + epsilon * perturbation) - source_tab(common)
                    receiver_response = receiver_tab(common + epsilon * perturbation) - receiver_tab(common)
                    metrics[f"jacobian_{name}_cosine"].append(
                        cosine(source_response, receiver_response)
                    )

    summary = {}
    for pair, values in per_pair.items():
        pair_summary = {}
        for key, items in values.items():
            if key != "basis":
                pair_summary[key] = {
                    "mean": float(np.mean(items)),
                    "ci95": bootstrap_ci(items),
                }
        pair_summary["basis"] = {}
        for rank in ranks:
            rank_key = str(rank)
            independent = [item[rank_key]["independent_error"] for item in values["basis"]]
            shared = [item[rank_key]["shared_error"] for item in values["basis"]]
            ratios = [item[rank_key]["shared_over_independent"] for item in values["basis"]]
            pair_summary["basis"][rank_key] = {
                "independent_error_mean": float(np.mean(independent)),
                "shared_error_mean": float(np.mean(shared)),
                "shared_over_independent_mean": float(np.mean(ratios)),
                "shared_over_independent_ci95": bootstrap_ci(ratios),
            }
        summary[pair] = pair_summary
    payload = {
        "mode": "operators",
        "checkpoint": str(args.checkpoint.resolve()),
        "images": args.operator_images,
        "lr_patch": args.patch,
        "ranks": list(ranks),
        "pairs": summary,
        "notes": [
            "All comparisons are made in the common Bx48xHxW LR space.",
            "Low-rank diagnostics factor TAB response fields, not internal token IDs.",
            "These are surrogates for R1/R2 and do not by themselves prove CPES will improve PSNR.",
        ],
    }
    save_json(args.output, payload)
    return payload


def normalized_response(response: torch.Tensor) -> torch.Tensor:
    magnitude = response.square().mean(dim=1, keepdim=True).sqrt()
    return standardized(magnitude)


def stage_code(batch: int, stage: int, height: int, width: int, tensor: torch.Tensor) -> torch.Tensor:
    code = tensor.new_zeros(batch, 8, height, width)
    code[:, stage] = 1
    return code


def evaluate_probes(
    model: LFMN,
    probes: nn.ModuleList,
    static_predictor: ResponsePredictor,
    dynamic_predictor: ResponsePredictor,
    projection: torch.Tensor,
    pairs: list[tuple[Path, Path]],
    patch: int,
    eval_crops: int,
    device: torch.device,
) -> dict:
    stage_l1 = [[] for _ in range(8)]
    stage_hf_l1 = [[] for _ in range(8)]
    static_errors: list[float] = []
    dynamic_errors: list[float] = []
    rng = random.Random(0)
    with torch.inference_mode():
        for image_index, (lr_path, hr_path) in enumerate(pairs):
            lr_full, hr_full = image_tensor(lr_path), image_tensor(hr_path)
            for crop_index in range(eval_crops):
                lr, hr = crop_pair(
                    lr_full, hr_full, patch, rng,
                    deterministic_index=image_index * eval_crops + crop_index,
                )
                lr, hr = lr.unsqueeze(0).to(device), hr.unsqueeze(0).to(device)
                trace = trace_backbone(model, lr)
                target = hr - F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
                target_hf = highpass(target)
                shallow = standardized(trace["shallow"])
                for stage in range(8):
                    prediction = probes[stage](trace["stages"][stage])
                    stage_l1[stage].append(float(F.l1_loss(prediction, target).item()))
                    stage_hf_l1[stage].append(float(F.l1_loss(highpass(prediction), target_hf).item()))
                    response_target = normalized_response(trace["tab_residuals"][stage])
                    code = stage_code(1, stage, patch, patch, shallow)
                    dynamic_code = F.conv2d(standardized(trace["tab_inputs"][stage]), projection)
                    static_output = static_predictor(torch.cat((shallow, code), dim=1))
                    dynamic_output = dynamic_predictor(torch.cat((shallow, dynamic_code), dim=1))
                    static_errors.append(float(F.l1_loss(static_output, response_target).item()))
                    dynamic_errors.append(float(F.l1_loss(dynamic_output, response_target).item()))
    improvements = [
        (static - dynamic) / max(static, 1e-12)
        for static, dynamic in zip(static_errors, dynamic_errors)
    ]
    return {
        "stage_residual_l1": [float(np.mean(values)) for values in stage_l1],
        "stage_highpass_l1": [float(np.mean(values)) for values in stage_hf_l1],
        "static_response_l1": float(np.mean(static_errors)),
        "dynamic_response_l1": float(np.mean(dynamic_errors)),
        "dynamic_relative_improvement": float(np.mean(improvements)),
        "dynamic_relative_improvement_ci95": bootstrap_ci(improvements),
        "validation_samples": len(static_errors),
    }


def run_probes(args: argparse.Namespace, model: LFMN, device: torch.device) -> dict:
    train_pairs = paired_paths(args.data_root, "train", args.train_images)
    valid_pairs = paired_paths(args.data_root, "valid", args.val_images)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    seed_results = []
    for seed in seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        rng = random.Random(seed)
        probe_template = LinearResidualProbe()
        probes = nn.ModuleList([copy.deepcopy(probe_template) for _ in range(8)]).to(device)
        static_predictor = ResponsePredictor().to(device)
        dynamic_predictor = copy.deepcopy(static_predictor).to(device)
        projection = torch.randn(8, 48, 1, 1, device=device)
        projection = projection / projection.square().sum(1, keepdim=True).sqrt().clamp_min(1e-6)
        optimizer = torch.optim.Adam(
            [
                {"params": probes.parameters(), "lr": args.probe_lr},
                {
                    "params": list(static_predictor.parameters()) + list(dynamic_predictor.parameters()),
                    "lr": args.predictor_lr,
                },
            ],
            betas=(0.9, 0.999),
        )
        for step in range(args.steps):
            lr, hr = sample_batch(train_pairs, args.patch, args.batch_size, rng)
            lr, hr = lr.to(device), hr.to(device)
            with torch.no_grad():
                trace = trace_backbone(model, lr)
                target = hr - F.interpolate(lr, scale_factor=4, mode="bicubic", align_corners=False)
                shallow = standardized(trace["shallow"])
            probe_loss = target.new_zeros(())
            static_loss = target.new_zeros(())
            dynamic_loss = target.new_zeros(())
            for stage in range(8):
                probe_loss = probe_loss + F.l1_loss(probes[stage](trace["stages"][stage]), target)
                response_target = normalized_response(trace["tab_residuals"][stage])
                code = stage_code(args.batch_size, stage, args.patch, args.patch, shallow)
                dynamic_code = F.conv2d(standardized(trace["tab_inputs"][stage]), projection)
                static_loss = static_loss + F.l1_loss(
                    static_predictor(torch.cat((shallow, code), dim=1)), response_target
                )
                dynamic_loss = dynamic_loss + F.l1_loss(
                    dynamic_predictor(torch.cat((shallow, dynamic_code), dim=1)), response_target
                )
            total = (probe_loss + static_loss + dynamic_loss) / 8
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            if (step + 1) % max(1, args.steps // 10) == 0:
                print(
                    f"seed={seed} step={step + 1}/{args.steps} "
                    f"probe={probe_loss.item() / 8:.6f} "
                    f"static={static_loss.item() / 8:.6f} "
                    f"dynamic={dynamic_loss.item() / 8:.6f}",
                    flush=True,
                )
        result = evaluate_probes(
            model, probes, static_predictor, dynamic_predictor, projection,
            valid_pairs, args.patch, args.eval_crops, device,
        )
        result["seed"] = seed
        seed_results.append(result)

    aggregate = {
        "stage_residual_l1_mean": np.mean(
            [result["stage_residual_l1"] for result in seed_results], axis=0
        ).tolist(),
        "stage_highpass_l1_mean": np.mean(
            [result["stage_highpass_l1"] for result in seed_results], axis=0
        ).tolist(),
        "dynamic_relative_improvement_mean": float(np.mean([
            result["dynamic_relative_improvement"] for result in seed_results
        ])),
    }
    payload = {
        "mode": "probes",
        "checkpoint": str(args.checkpoint.resolve()),
        "frozen_baseline": True,
        "train_images": args.train_images,
        "validation_images": args.val_images,
        "steps_per_seed": args.steps,
        "probe_lr": args.probe_lr,
        "predictor_lr": args.predictor_lr,
        "batch_size": args.batch_size,
        "lr_patch": args.patch,
        "seeds": list(seeds),
        "probe_parameters_each": sum(parameter.numel() for parameter in LinearResidualProbe().parameters()),
        "static_predictor_parameters": sum(parameter.numel() for parameter in ResponsePredictor().parameters()),
        "dynamic_predictor_parameters": sum(parameter.numel() for parameter in ResponsePredictor().parameters()),
        "seed_results": seed_results,
        "aggregate": aggregate,
        "notes": [
            "Stage probes predict the same HR bicubic residual with identical linear heads.",
            "The static/dynamic test is an equal-capacity surrogate for R3.",
            "Dynamic input uses a fixed random eight-channel projection of current TAB input.",
            "Probe results diagnose accessibility; they do not constitute CPES performance.",
        ],
    }
    save_json(args.output, payload)
    return payload


def main() -> None:
    args = parse_args()
    require(args.patch >= 16, "LR patch must be at least 16 for the baseline LRSA schedule")
    device = torch.device(args.device)
    model = load_model(args.checkpoint.resolve(), device)
    if args.mode == "smoke":
        payload = run_smoke(args, model, device)
    elif args.mode == "operators":
        payload = run_operators(args, model, device)
    else:
        payload = run_probes(args, model, device)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
