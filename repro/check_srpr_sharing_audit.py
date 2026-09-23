#!/usr/bin/env python3
"""Local structural checks for the zero-training SRPR sharing audit."""
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'repro'))
from audit_srpr_sharing import AuditableSRPRv2, COMPONENTS


def main():
    torch.manual_seed(1)
    model = AuditableSRPRv2().eval()
    parameters = sum(p.numel() for p in model.parameters())
    assert parameters == 841563, parameters
    # Make the zero-initialized correction heads observable for interventions.
    with torch.no_grad():
        for index, prox in enumerate(model.proximal):
            prox.writeback[-1].weight.fill_((index + 1) * 1e-4)
            prox.q_update.weight.fill_((index + 1) * 1e-4)
    x = torch.rand(1, 3, 16, 16)
    with torch.no_grad():
        model.set_audit_plan()
        reference = model(x)
        assert torch.equal(reference, model(x))
        changed = {}
        for component in COMPONENTS:
            model.set_audit_plan(routes={component: [0] * 8})
            output = model(x)
            changed[component] = float((output - reference).abs().mean())
            assert changed[component] > 0, component
        for component in COMPONENTS:
            model.set_audit_plan(ablations={component: [3]})
            assert not torch.equal(model(x), reference), component
    model.set_audit_plan()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = AuditableSRPRv2().eval()
    restored.load_state_dict(
        torch.load(buffer, map_location='cpu', weights_only=True), strict=True
    )
    with torch.no_grad():
        assert torch.equal(reference, restored(x))
    print(json.dumps({
        'audit': 'SRPR sharing boundary',
        'parameters': parameters,
        'expected_n12_parameters': 841563,
        'component_intervention_mean_abs_change': changed,
        'stage_ablation_paths': 'passed',
        'strict_reload': 'passed',
    }, indent=2))


if __name__ == '__main__':
    main()
