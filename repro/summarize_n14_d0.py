#!/usr/bin/env python3
"""Summarize N14 D0 JSON diagnostics without overstating the evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    smoke = load(args.run_dir / "smoke.json")
    operators = load(args.run_dir / "operators.json")
    probes = load(args.run_dir / "probes.json")

    print("N14/CPES-SR D0 diagnostic summary")
    print("checkpoint:", smoke["checkpoint"])
    print("smoke:", smoke["status"])
    print("\nR1/R2 operator-response evidence by proposed pair")
    for pair, metrics in operators["pairs"].items():
        print(
            "pair {}: native_CKA={:.4f}, common_CKA={:.4f}, "
            "noise_Jcos={:.4f}, highpass_Jcos={:.4f}".format(
                pair,
                metrics["native_cka"]["mean"],
                metrics["common_cka"]["mean"],
                metrics["jacobian_noise_cosine"]["mean"],
                metrics["jacobian_highpass_cosine"]["mean"],
            )
        )
        for rank, result in metrics["basis"].items():
            print(
                "  rank {} shared/independent error={:.4f} CI={}".format(
                    rank,
                    result["shared_over_independent_mean"],
                    result["shared_over_independent_ci95"],
                )
            )

    aggregate = probes["aggregate"]
    print("\nD1 frozen-backbone linear probe")
    print("residual L1 by stage:", aggregate["stage_residual_l1_mean"])
    print("high-pass L1 by stage:", aggregate["stage_highpass_l1_mean"])
    print("\nR3 equal-capacity response-prediction surrogate")
    print(
        "dynamic relative improvement: {:.2%}".format(
            aggregate["dynamic_relative_improvement_mean"]
        )
    )
    for result in probes["seed_results"]:
        print(
            "seed {}: {:.2%}, CI={}".format(
                result["seed"],
                result["dynamic_relative_improvement"],
                result["dynamic_relative_improvement_ci95"],
            )
        )

    print("\nDecision must be made from R1, R2, and R3 together.")
    print("D1 is supportive only. These diagnostics are not CPES performance results.")


if __name__ == "__main__":
    main()
