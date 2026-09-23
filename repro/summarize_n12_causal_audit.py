#!/usr/bin/env python3
"""Compare epoch-20/40 SRPRv2 intervention JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print("N12/SRPRv2 no-training causal audit")
    for epoch in (20, 40):
        result = load(args.run_dir / f"epoch_{epoch:04d}.json")
        summaries = result["summaries"]
        print(f"\nepoch {epoch}; dataset={result['dataset']}; images={result['images']}")
        for name in result["variants"]:
            values = summaries[name]
            line = (
                f"{name}: vs_B0={values['psnr_delta_vs_baseline_mean']:+.6f} "
                f"CI={values['psnr_delta_vs_baseline_ci95']}"
            )
            if name != "full":
                line += (
                    f"; vs_full={values['psnr_delta_vs_full_mean']:+.6f} "
                    f"CI={values['psnr_delta_vs_full_ci95']}"
                )
            print(line)

    print("\nInterpretation rules")
    print("- full > reset_state supports cross-stage persistence necessity.")
    print("- full > no_writeback supports feature-writeback necessity.")
    print("- no_q/no_final_q >= full argues against retaining the RGB q path.")
    print("- zero_observation >= full argues against retaining D/D^T observation feedback.")
    print("- state_writeback_only near or above full supports an SRPR-Lite retraining candidate.")
    print("Inference interventions have distribution shift; use them to reject branches, not claim retrained gains.")


if __name__ == "__main__":
    main()
