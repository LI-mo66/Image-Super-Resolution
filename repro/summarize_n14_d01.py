#!/usr/bin/env python3
"""Summarize the cycle-aligned N14 D0.1 diagnostic and apply fixed gates."""
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
    result = load(args.run_dir / "homologous_operators.json")

    if result.get("pairing") != "homologous":
        raise RuntimeError("expected a homologous-pair operator result")

    print("N14/CPES-SR D0.1 cycle-aligned diagnostic")
    print("checkpoint:", smoke["checkpoint"])
    print("smoke:", smoke["status"])
    print("pairing: (1,5), (2,6), (3,7), (4,8)")
    print()

    supported = 0
    gray = 0
    for pair, metrics in result["pairs"].items():
        rank8 = metrics["basis"]["8"]
        ratio = rank8["shared_over_independent_mean"]
        transfer = metrics["prototype_transfer_relative_error"]["mean"]
        transfer_ci = metrics["prototype_transfer_relative_error"]["ci95"]
        proto_cos = metrics["prototype_set_cosine"]["mean"]
        transfer_cos = metrics["prototype_transfer_cosine"]["mean"]
        if ratio <= 1.05 and transfer <= 0.05:
            decision = "SUPPORT"
            supported += 1
        elif ratio <= 1.10 and transfer <= 0.10:
            decision = "GRAY"
            gray += 1
        else:
            decision = "REJECT"
        print(
            f"pair {pair}: {decision}; native_CKA={metrics['native_cka']['mean']:.4f}, "
            f"common_CKA={metrics['common_cka']['mean']:.4f}, "
            f"prototype_set_cos={proto_cos:.4f}"
        )
        print(
            f"  prototype transfer: relative_error={transfer:.4f} CI={transfer_ci}, "
            f"cosine={transfer_cos:.4f}"
        )
        for rank, value in metrics["basis"].items():
            print(
                f"  rank {rank} shared/independent="
                f"{value['shared_over_independent_mean']:.4f} "
                f"CI={value['shared_over_independent_ci95']}"
            )

    print()
    if supported >= 3:
        overall = "D0.1 SUPPORT: eligible for a separately reviewed M1 design."
    elif supported + gray >= 3:
        overall = "D0.1 GRAY: only a targeted supplemental diagnostic is allowed."
    else:
        overall = "D0.1 REJECT: stop the relation-prototype-sharing branch of CPES."
    print(overall)
    print(f"supported_pairs={supported}/4 gray_pairs={gray}/4")
    print("This is a mechanism diagnostic, not a CPES performance result.")


if __name__ == "__main__":
    main()
