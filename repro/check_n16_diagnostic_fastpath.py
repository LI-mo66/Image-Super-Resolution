#!/usr/bin/env python3
"""Verify that the N16 inference diagnostic fast path is output-exact."""
import io
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LFMN_ROOT = os.path.join(ROOT, 'LFMN')
if LFMN_ROOT not in sys.path:
    sys.path.insert(0, LFMN_ROOT)

from model.lfmnsrprv2 import Net as SRPRv2Net


def main():
    torch.manual_seed(19)
    model = SRPRv2Net(scale=4).eval()
    sample = torch.rand(1, 3, 28, 28) * 255
    state_keys = tuple(model.state_dict())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    with torch.inference_mode():
        reference = model.enable_eval_diagnostics(True)(sample)
        diagnostics = model.diagnostics_snapshot()
        fast = model.enable_eval_diagnostics(False)(sample)
    torch.testing.assert_close(fast, reference, rtol=0, atol=0)
    assert diagnostics
    assert all(value.shape == (8,) for value in diagnostics.values())
    assert model.last_diagnostics == {}

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SRPRv2Net(scale=4).eval()
    restored.load_state_dict(
        torch.load(buffer, map_location='cpu', weights_only=True), strict=True
    )
    assert tuple(restored.state_dict()) == state_keys
    assert not restored.collect_eval_diagnostics
    with torch.inference_mode():
        restored_output = restored(sample)
    torch.testing.assert_close(restored_output, fast, rtol=0, atol=0)

    training_model = SRPRv2Net(scale=4).train()
    training_model.load_state_dict(model.state_dict(), strict=True)
    fast_training_model = SRPRv2Net(scale=4).train()
    fast_training_model.load_state_dict(model.state_dict(), strict=True)
    diagnostic_input = sample.clone().requires_grad_(True)
    fast_input = sample.clone().requires_grad_(True)
    diagnostic_output = training_model.enable_eval_diagnostics(True)(
        diagnostic_input
    )
    fast_training_output = fast_training_model(fast_input)
    torch.testing.assert_close(
        fast_training_output, diagnostic_output, rtol=0, atol=0
    )
    diagnostic_output.mean().backward()
    fast_training_output.mean().backward()
    assert training_model.last_diagnostics
    assert fast_training_model.last_diagnostics == {}
    torch.testing.assert_close(fast_input.grad, diagnostic_input.grad,
                               rtol=0, atol=0)
    for diagnostic_parameter, fast_parameter in zip(
        training_model.parameters(), fast_training_model.parameters()
    ):
        if diagnostic_parameter.grad is None or fast_parameter.grad is None:
            assert diagnostic_parameter.grad is fast_parameter.grad
        else:
            torch.testing.assert_close(
                fast_parameter.grad, diagnostic_parameter.grad,
                rtol=0, atol=0,
            )

    print(json.dumps({
        'audit': 'N16 diagnostic-free inference fast path',
        'parameters': parameter_count,
        'state_dict_keys_unchanged': True,
        'eval_output_max_abs_error': float((fast - reference).abs().max()),
        'eval_diagnostics_opt_in': 'passed',
        'default_eval_diagnostics_skipped': 'passed',
        'training_diagnostics_opt_in': 'passed',
        'training_output_max_abs_error': float(
            (fast_training_output - diagnostic_output).detach().abs().max()
        ),
        'training_input_gradient_max_abs_error': float(
            (fast_input.grad - diagnostic_input.grad).abs().max()
        ),
        'training_parameter_gradients_exact': 'passed',
        'strict_save_reload': 'passed',
        'finite_training_gradient': 'passed',
    }, indent=2))


if __name__ == '__main__':
    main()
