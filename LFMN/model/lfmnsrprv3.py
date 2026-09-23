"""N13 SRPRv3: output-consistent progressive reconstruction for LFMN."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lfmn import Net as BaselineNet


def _logit(probability):
    return math.log(probability / (1.0 - probability))


class SharedPackedReconstruction(nn.Module):
    """Predict a packed RGB x4 image increment on the LR grid."""

    def __init__(self, channels=48, hidden=32, scale=4):
        super().__init__()
        self.scale = int(scale)
        self.body = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 3 * self.scale * self.scale, 1),
        )
        nn.init.normal_(self.body[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, feature):
        return F.pixel_shuffle(self.body(feature), self.scale)


class SharedResidualFeedback(nn.Module):
    """Lift an LR observation residual into the LFMN feature space."""

    def __init__(self, channels=48, hidden=12):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )
        nn.init.normal_(self.body[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, residual):
        return self.body(residual)


class Net(BaselineNet):
    """LFMN whose sole SR output is progressively updated at four stages."""

    update_stages = (1, 3, 5, 7)  # zero-based: stages 2, 4, 6, 8

    def __init__(self, scale=4, n_feats=48, side_c=32, n_stage=8,
                 normalize_overlap=False):
        if int(scale) != 4:
            raise ValueError('SRPRv3 currently requires x4')
        if int(n_stage) != 8:
            raise ValueError('SRPRv3 requires the registered eight-stage backbone')
        super().__init__(scale, n_feats, side_c, n_stage, normalize_overlap)

        # The original one-shot tail is replaced rather than retained as a
        # second output path. The progressive image state is the only output.
        del self.upconv1
        del self.upconv2
        del self.pixel_shuffle
        del self.last_conv

        self.reconstruction = SharedPackedReconstruction(n_feats, hidden=32, scale=4)
        self.feedback = SharedResidualFeedback(n_feats, hidden=12)
        self.update_logits = nn.Parameter(torch.full((4,), _logit(0.25)))
        self.feedback_logits = nn.Parameter(torch.full((3,), _logit(0.10)))
        self.feedback_enabled = True
        self.last_diagnostics = {}

    @staticmethod
    def _analysis(image, output_size):
        return F.interpolate(
            image, size=output_size, mode='bicubic',
            align_corners=False, antialias=True,
        )

    def diagnostics_snapshot(self):
        return {
            key: value.detach().float().cpu().clone()
            for key, value in self.last_diagnostics.items()
        }

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        image_state = F.interpolate(
            x, scale_factor=4, mode='bilinear', align_corners=False,
        )
        previous_residual = x - self._analysis(image_state, x.shape[-2:])
        update_index = 0
        update_norms = []
        residual_norms = []
        consistency_ratios = []
        feedback_norms = []

        for stage_index in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[stage_index](fs)
            fm = beta * prev + gamma
            global_attention, local_attention = self.blocks[stage_index]
            transformed = global_attention(fm)
            transformed = local_attention(
                transformed, self.patch_size[stage_index]
            )
            feat = self.esas[stage_index](
                prev + self.mid_convs[stage_index](transformed)
            )

            if stage_index not in self.update_stages:
                continue

            update = torch.sigmoid(self.update_logits[update_index]) * \
                self.reconstruction(feat)
            image_state = image_state + update
            residual = x - self._analysis(image_state, x.shape[-2:])
            before_norm = previous_residual.float().square().mean().sqrt()
            after_norm = residual.float().square().mean().sqrt()
            update_norms.append(update.float().square().mean().sqrt())
            residual_norms.append(after_norm)
            consistency_ratios.append(after_norm / before_norm.clamp_min(1e-8))

            if update_index < 3:
                lifted = self.feedback(residual)
                feedback = torch.sigmoid(
                    self.feedback_logits[update_index]
                ) * lifted
                if self.feedback_enabled:
                    feat = feat + feedback
                feedback_norms.append(feedback.float().square().mean().sqrt())
            else:
                feedback_norms.append(after_norm.new_zeros(()))
            previous_residual = residual
            update_index += 1

        self.last_diagnostics = {
            'update_l2': torch.stack(update_norms),
            'residual_l2': torch.stack(residual_norms),
            'consistency_ratio': torch.stack(consistency_ratios),
            'feedback_l2': torch.stack(feedback_norms),
            'update_scale': torch.sigmoid(self.update_logits),
            'feedback_scale': torch.sigmoid(self.feedback_logits),
        }
        return image_state


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 4
    return Net(scale=scale)
