"""Measure final B0/F2 network-only FP32 CUDA resources (no training or quality scoring)."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_network_types(repo):
    # Force both models to the same explicit Python baseline even when a .so exists.
    package_name = "_f2_resource_models"
    model_dir = repo / "LFMN" / "model"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    sys.modules[package_name] = package
    loaded = []
    for name in ("lfmn", "lfmnf2"):
        path = model_dir / (name + ".py")
        spec = importlib.util.spec_from_file_location(package_name + "." + name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Python model source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded.append(module.Net)
    return loaded


def percentile(values, percent):
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    low = math.floor(index)
    high = math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def safe_run(group, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("protocol.runs entries must be relative directory strings")
    result = (group / relative).resolve()
    if result == group or group not in result.parents:
        raise ValueError("Run directory escapes the supplied group")
    return result


def environment(torch):
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_index": device,
        "device_name": props.name,
        "device_total_memory_bytes": props.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "visible_cuda_device_count": torch.cuda.device_count(),
        "precision": "FP32",
        "input_dtype": "torch.float32",
        "model_dtype": "torch.float32",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "settings_changed_by_profiler": False,
    }


def profile(args):
    import torch

    group = args.group.resolve()
    protocol_path = group / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol", {}).get("stop_epoch") != 150:
        raise ValueError("Resource comparison requires protocol.stop_epoch == 150")
    if protocol.get("engineering_only") is not False:
        raise ValueError("Resource comparison requires engineering_only == false")
    runs = protocol.get("runs", {})
    if not all(name in runs for name in ("B0", "F2")):
        raise ValueError("protocol.runs must contain B0 and F2")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU timings are not a valid substitute")
    # One current CUDA device, no DataParallel or cross-device speed comparison.
    device = torch.device("cuda", torch.cuda.current_device())
    env = environment(torch)
    expected = protocol.get("environment", {})
    for key, actual in (("gpu_name", env["device_name"]), ("torch_version", env["torch_version"]),
                        ("cuda_version", env["cuda_version"]), ("cudnn_version", env["cudnn_version"])):
        if expected.get(key) != actual:
            raise ValueError("Resource environment differs from registered paired training: " + key)
    repo = Path(__file__).resolve().parents[1]
    original, f2 = load_network_types(repo)
    paths = {name: safe_run(group, runs[name]) / "model" / "model_150.pt"
             for name in ("B0", "F2")}
    checkpoints = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[name] = {"path": str(path), "sha256": sha256(path),
                             "epoch": 150, "strict_load": True}
    generator = torch.Generator(device="cpu").manual_seed(123)
    inputs = {size: torch.rand((1, 3, size, size), generator=generator,
                               dtype=torch.float32) * 255
              for size in args.sizes}
    rows, samples = [], []
    with torch.inference_mode():
        for name, network_type in (("B0", original), ("F2", f2)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.manual_seed(1)
            model = (network_type(scale=4) if name == "B0"
                     else network_type(scale=4, init_seed=1))
            state = torch.load(paths[name], map_location="cpu", weights_only=True)
            # Training saves the bare state_dict; wrappers/prefix rewriting are forbidden.
            model.load_state_dict(state, strict=True)
            del state
            model = model.to(device=device, dtype=torch.float32).eval()
            params = sum(parameter.numel() for parameter in model.parameters())
            if params != 759627:
                raise RuntimeError(f"{name} parameter count {params} != 759627")
            for size in args.sizes:
                x = inputs[size].to(device)
                torch.cuda.synchronize(device)
                output = model(x)
                expected_shape = (1, 3, size * 4, size * 4)
                if tuple(output.shape) != expected_shape or not bool(torch.isfinite(output).all()):
                    raise RuntimeError(f"{name} invalid output for input {size}")
                del output
                for _ in range(args.warmup):
                    output = model(x)
                    torch.cuda.synchronize(device)
                    del output
                torch.cuda.synchronize(device)
                # Release unused warmup blocks; retain weights and input.
                torch.cuda.empty_cache()
                case = []
                for repeat in range(args.repeats):
                    torch.cuda.synchronize(device)
                    before_allocated = torch.cuda.memory_allocated(device)
                    before_reserved = torch.cuda.memory_reserved(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    output = model(x)
                    torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    peak_allocated = torch.cuda.max_memory_allocated(device)
                    peak_reserved = torch.cuda.max_memory_reserved(device)
                    sample = {
                        "model": name, "input_height": size, "input_width": size,
                        "repeat": repeat, "wall_ms": elapsed_ms,
                        "before_allocated_bytes": before_allocated,
                        "before_reserved_bytes": before_reserved,
                        "peak_allocated_bytes": peak_allocated,
                        "peak_reserved_bytes": peak_reserved,
                        "incremental_peak_allocated_bytes": max(0, peak_allocated - before_allocated),
                        "incremental_peak_reserved_bytes": max(0, peak_reserved - before_reserved),
                        "checkpoint_sha256": checkpoints[name]["sha256"],
                        "environment_key": "environment",
                    }
                    samples.append(sample)
                    case.append(sample)
                    del output
                times = [sample["wall_ms"] for sample in case]
                row = {
                    "model": name, "epoch": 150, "scale": 4, "batch": 1,
                    "input_height": size, "input_width": size,
                    "gpu": env["device_name"], "precision": "FP32",
                    "ensemble": "OFF", "chop": False,
                    "params": params, "ffn_only_macs": 185856 * size * size,
                    "full_flops": None,
                    "warmup": args.warmup, "repeats": args.repeats,
                    "median_ms": statistics.median(times),
                    "p10_ms": percentile(times, 10), "p90_ms": percentile(times, 90),
                    "min_ms": min(times), "max_ms": max(times),
                    "std_ms": statistics.pstdev(times),
                    "peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in case),
                    "peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in case),
                    "incremental_peak_allocated_bytes": max(s["incremental_peak_allocated_bytes"] for s in case),
                    "incremental_peak_reserved_bytes": max(s["incremental_peak_reserved_bytes"] for s in case),
                    "checkpoint_sha256": checkpoints[name]["sha256"],
                }
                rows.append(row)
                print(f"{name} {size}x{size}: median={row['median_ms']:.3f} ms, "
                      f"peak allocated={row['peak_allocated_bytes']} bytes", flush=True)
                del x
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()

    output_dir = group / ("resource_OFF_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output_dir.mkdir(exist_ok=False)
    csv_path = output_dir / "resource_comparison.csv"
    json_path = output_dir / "resource_samples.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "environment": env,
        "checkpoints": checkpoints,
        "source_sha256": {name: sha256(repo / "LFMN" / "model" / (name + ".py"))
                          for name in ("lfmn", "lfmnf2")},
        "input": {"seed": 123, "generator": "CPU", "distribution": "uniform [0,255)",
                  "batch": 1, "same_cpu_tensor_reused_for_both_models": True,
                  "sizes": args.sizes},
        "measurement": {
            "warmup": args.warmup, "repeats": args.repeats, "ensemble": "OFF",
            "chop": False, "inference_mode": True, "scale": 4,
            "timing": "perf_counter wall time with CUDA synchronization before and after each forward",
            "order": "sequential models B0 then F2; sizes in CLI order; no interleaving",
            "scope": "single whole-image network forward; excludes checkpoint/input I/O, quantization and metrics",
            "memory": "per-forward allocator peaks; increments subtract weights/input and cached baseline before forward; reserved increments can be zero when cache is reused",
            "finite_and_shape_check": "once before warmup, outside timed repeats",
            "std_definition": "population standard deviation",
            "percentile_definition": "linear interpolation of sorted wall times",
            "full_flops": None,
            "ffn_only_macs_formula": "185856 * input_height * input_width",
            "ffn_only_macs_coverage": "FFN convolution/linear operations only; does not represent complete model FLOPs",
            "quality_scores": "not evaluated by this efficiency-only script",
            "speed_delta": "not computed; rows share one GPU and process, external mismatched-GPU rows must not be compared directly",
        },
        "summaries": rows,
        "samples": samples,
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = {"directory": output_dir.name, "resource_comparison_csv": str(csv_path.relative_to(group)),
              "resource_samples_json": str(json_path.relative_to(group))}
    (group / "resources_latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Resource results: {output_dir}", flush=True)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", type=Path, help="150-epoch experiment group containing protocol.json")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sizes", default="64,128", help="comma-separated square LR sizes")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats must be >= 1")
    try:
        args.sizes = [int(value.strip()) for value in args.sizes.split(",")]
    except ValueError:
        parser.error("sizes must contain comma-separated integers")
    if not args.sizes or any(size < 32 for size in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique integers >= 32")
    profile(args)


if __name__ == "__main__":
    main()
