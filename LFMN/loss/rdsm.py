"""Training-only supervision for residual-demand modulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_auxiliary(output):
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError('RDSM supervision requires (sr, auxiliary) model output')
    _, auxiliary = output
    required = {
        'base', 'scale', 'demand_predictions', 'residual_predictions'
    }
    missing = required.difference(auxiliary)
    if missing:
        raise ValueError('RDSM auxiliary output is missing {}'.format(sorted(missing)))
    return auxiliary


def target_residual(auxiliary, hr):
    base = auxiliary['base']
    scale = int(auxiliary['scale'])
    if base.shape != hr.shape:
        raise ValueError(
            'RDSM base/HR shape mismatch: {} vs {}'.format(base.shape, hr.shape)
        )
    return F.pixel_unshuffle(hr - base, downscale_factor=scale)


class ResidualProbeLoss(nn.Module):
    """Deeply supervise the shared stage probe in LR residual space."""

    uses_auxiliary_output = True

    def forward(self, output, hr):
        auxiliary = unpack_auxiliary(output)
        target = target_residual(auxiliary, hr)
        predictions = auxiliary['residual_predictions']
        if not predictions:
            raise ValueError('RDSM residual predictions are empty')
        return torch.stack([
            F.l1_loss(prediction, target) for prediction in predictions
        ]).mean()


class DemandDistillationLoss(nn.Module):
    """Teach the inference router where residual reconstruction is unfinished."""

    uses_auxiliary_output = True

    def forward(self, output, hr):
        auxiliary = unpack_auxiliary(output)
        target = target_residual(auxiliary, hr)
        demands = auxiliary['demand_predictions']
        residuals = auxiliary['residual_predictions']
        if len(demands) != len(residuals) or not demands:
            raise ValueError('RDSM demand/residual stage counts do not match')

        losses = []
        for demand, residual in zip(demands, residuals):
            with torch.no_grad():
                remaining = (target - residual.detach()).abs().mean(
                    dim=1, keepdim=True
                )
                mean = remaining.flatten(2).mean(dim=2).view(-1, 1, 1, 1)
                # Relative local difficulty: mean error maps to 0.5 and values
                # at or above twice the mean saturate at 1.
                oracle = (remaining / (2.0 * mean + 1e-6)).clamp_(0.0, 1.0)
            losses.append(F.l1_loss(demand, oracle))
        return torch.stack(losses).mean()
