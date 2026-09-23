#!/usr/bin/env python3
"""Structural, causal, gradient and reload checks for N13/SRPRv3."""
import io
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet  # noqa: E402
from model.lfmnsrprv3 import Net as SRPRv3Net  # noqa: E402


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def assert_new_gradients(model):
    prefixes = ('reconstruction.', 'feedback.', 'update_logits', 'feedback_logits')
    checked = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            assert parameter.grad is not None, f'missing gradient: {name}'
            assert torch.isfinite(parameter.grad).all(), f'nonfinite gradient: {name}'
            assert parameter.grad.abs().sum() > 0, f'zero gradient: {name}'
            checked.append(name)
    assert checked


def main():
    torch.manual_seed(1)
    baseline = BaselineNet(scale=4).eval()
    model = SRPRv3Net(scale=4).eval()
    baseline_parameters = count_parameters(baseline)
    candidate_parameters = count_parameters(model)
    assert candidate_parameters <= baseline_parameters
    assert not any(hasattr(model, name) for name in (
        'upconv1', 'upconv2', 'pixel_shuffle', 'last_conv'
    ))
    assert model.update_stages == (1, 3, 5, 7)

    audit_model = SRPRv3Net(scale=4)
    incompatible = audit_model.load_state_dict(baseline.state_dict(), strict=False)
    expected_missing = {
        name for name in audit_model.state_dict()
        if name.startswith(('reconstruction.', 'feedback.'))
        or name in ('update_logits', 'feedback_logits')
    }
    expected_unexpected = {
        name for name in baseline.state_dict()
        if name.startswith(('upconv1.', 'upconv2.', 'last_conv.'))
    }
    assert set(incompatible.missing_keys) == expected_missing
    assert set(incompatible.unexpected_keys) == expected_unexpected

    x = torch.rand(1, 3, 32, 32) * 255
    with torch.no_grad():
        output_on = model(x)
        diagnostics = model.diagnostics_snapshot()
        model.feedback_enabled = False
        output_off = model(x)
        model.feedback_enabled = True
    assert output_on.shape == (1, 3, 128, 128)
    assert torch.isfinite(output_on).all()
    assert (output_on - output_off).abs().max() > 1e-8
    assert set(diagnostics) == {
        'update_l2', 'residual_l2', 'consistency_ratio', 'feedback_l2',
        'update_scale', 'feedback_scale',
    }
    assert diagnostics['update_l2'].shape == (4,)
    assert diagnostics['residual_l2'].shape == (4,)
    assert diagnostics['consistency_ratio'].shape == (4,)
    assert diagnostics['feedback_l2'].shape == (4,)
    assert diagnostics['update_scale'].shape == (4,)
    assert diagnostics['feedback_scale'].shape == (3,)
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert diagnostics['update_l2'].min() > 0
    assert diagnostics['feedback_l2'][:3].min() > 0
    assert diagnostics['feedback_l2'][3] == 0
    with torch.no_grad():
        rectangular = model(torch.rand(1, 3, 36, 40) * 255)
    assert rectangular.shape == (1, 3, 144, 160)
    assert torch.isfinite(rectangular).all()

    train_model = SRPRv3Net(scale=4).train()
    train_input = torch.rand(1, 3, 32, 32) * 255
    target = torch.rand(1, 3, 128, 128) * 255
    loss = torch.nn.functional.l1_loss(train_model(train_input), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert_new_gradients(train_model)

    buffer = io.BytesIO()
    torch.save(train_model.state_dict(), buffer)
    buffer.seek(0)
    reloaded = SRPRv3Net(scale=4).eval()
    reloaded.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    train_model.eval()
    with torch.no_grad():
        reference = train_model(train_input)
        restored = reloaded(train_input)
    torch.testing.assert_close(reference, restored, rtol=0, atol=0)

    amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.float16 if amp_device == 'cuda' else torch.bfloat16
    amp_model = SRPRv3Net(scale=4).to(amp_device).eval()
    amp_input = x.to(amp_device)
    with torch.no_grad(), torch.autocast(device_type=amp_device, dtype=amp_dtype):
        amp_output = amp_model(amp_input)
    assert torch.isfinite(amp_output).all()

    result = {
        'candidate': 'N13/SRPRv3',
        'baseline_parameters': baseline_parameters,
        'candidate_parameters': candidate_parameters,
        'parameter_delta': candidate_parameters - baseline_parameters,
        'output_shape': list(output_on.shape),
        'feedback_on_off_max_abs': float((output_on - output_off).abs().max()),
        'diagnostics': {key: value.tolist() for key, value in diagnostics.items()},
        'amp_device': amp_device,
        'status': 'passed',
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
