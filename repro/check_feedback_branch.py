#!/usr/bin/env python3
"""Check feedback initialization, parameter budget, and gradient flow."""
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LFMN_ROOT = PROJECT_ROOT / 'LFMN'
sys.path.insert(0, str(LFMN_ROOT))

from model.lfmn import Net as BaselineNet  # noqa: E402
from model.lfmnfeedback import Net as FeedbackNet  # noqa: E402


def main():
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseline = BaselineNet(scale=4)
    feedback = FeedbackNet(scale=4, feedback_stages=(3, 5, 7), feedback_mid=8)

    checkpoint = LFMN_ROOT / 'model' / 'scale4_model_939.pt'
    state = torch.load(checkpoint, map_location='cpu') if checkpoint.is_file() else baseline.state_dict()
    baseline.load_state_dict(state, strict=True)
    incompatible = feedback.load_state_dict(state, strict=False)
    expected_missing = {key for key in feedback.state_dict() if key.startswith('feedback.')}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            'unexpected checkpoint mismatch: missing={}, unexpected={}'.format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )

    baseline = baseline.to(device).eval()
    feedback = feedback.to(device).eval()
    sample = torch.rand(1, 3, 32, 32, device=device)
    with torch.inference_mode():
        baseline_output = baseline(sample)
        feedback_output = feedback(sample)
    max_abs_difference = (baseline_output - feedback_output).abs().max().item()
    if max_abs_difference != 0.0:
        raise RuntimeError(
            'zero-initialized feedback changed baseline output: {}'.format(
                max_abs_difference
            )
        )

    baseline_params = sum(parameter.numel() for parameter in baseline.parameters())
    feedback_params = sum(parameter.numel() for parameter in feedback.parameters())
    added_params = feedback_params - baseline_params
    added_percent = 100.0 * added_params / baseline_params
    if added_percent > 1.0:
        raise RuntimeError('feedback branch exceeds 1% parameter budget')

    branch = feedback.feedback['3']
    branch.train()
    delta_beta, delta_gamma = branch(torch.rand(2, 48, 16, 16, device=device))
    (delta_beta.sum() + delta_gamma.sum()).backward()
    gradient_sum = branch.expand.weight.grad.abs().sum().item()
    if gradient_sum == 0.0:
        raise RuntimeError('feedback output projection receives no gradient')

    print('device: {}'.format(device))
    print('baseline parameters: {:,}'.format(baseline_params))
    print('feedback parameters: {:,}'.format(feedback_params))
    print('added parameters: {:,} ({:.3f}%)'.format(added_params, added_percent))
    print('initial max output difference: {:.1f}'.format(max_abs_difference))
    print('feedback output gradient sum: {:.6f}'.format(gradient_sum))
    print('feedback branch validation passed')


if __name__ == '__main__':
    main()
