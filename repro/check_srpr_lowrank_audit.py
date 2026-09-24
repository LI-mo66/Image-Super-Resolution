#!/usr/bin/env python3
"""Structural and numerical checks for the SRPR gate SVD audit."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro'))
from audit_srpr_lowrank import (
    AuditableSRPRv2, build_svd_bank, candidate_parameters,
    compression_metadata, ranks_for_energy,
)


def main():
    torch.manual_seed(1)
    model = AuditableSRPRv2().eval()
    assert sum(p.numel() for p in model.parameters()) == 841563
    # Fresh N12 correction heads are zero-initialized; make the downstream
    # state path observable before testing whether gate truncation changes it.
    with torch.no_grad():
        for index, prox in enumerate(model.proximal):
            prox.writeback[-1].weight.fill_((index + 1) * 1e-4)
            prox.q_update.weight.fill_((index + 1) * 1e-4)
    originals, approximations, spectra = build_svd_bank(model)
    assert len(originals) == len(approximations) == len(spectra) == 8
    maximum_full_rank_error = max(
        stage['relative_weight_error_by_rank']['48'] for stage in spectra
    )
    assert maximum_full_rank_error < 1e-5, maximum_full_rank_error
    for target in (0.95, 0.99, 0.995, 0.999):
        ranks = ranks_for_energy(spectra, target)
        assert len(ranks) == 8 and all(1 <= rank <= 48 for rank in ranks)
    expected = {4: 808251, 8: 812955, 12: 817659, 16: 822363}
    for rank, parameters in expected.items():
        assert candidate_parameters([rank] * 8) == parameters
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        reference = model(x)
        for stage in range(8):
            model.proximal[stage].gate.weight.copy_(approximations[stage][4])
        compressed = model(x)
    mean_change = float((compressed - reference).abs().mean())
    assert mean_change > 0
    metadata = compression_metadata([8] * 8)
    assert metadata['gate_parameter_reduction'] > .70
    print(json.dumps({
        'audit': 'stage-specific SRPR gate low-rank SVD',
        'n12_parameters': 841563,
        'maximum_rank48_relative_weight_error': maximum_full_rank_error,
        'rank8_candidate_parameters': metadata['candidate_parameters'],
        'rank8_gate_parameter_reduction': metadata['gate_parameter_reduction'],
        'rank4_output_mean_abs_change': mean_change,
        'svd_shapes_and_energy': 'passed',
        'parameter_formulas': 'passed',
    }, indent=2))


if __name__ == '__main__':
    main()
