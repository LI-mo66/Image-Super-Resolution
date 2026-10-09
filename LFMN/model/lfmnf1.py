"""F1: identity-preserved cross-stage state with ESA-filtered increments."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class Net(BaselineNet):
    """Keep the previous state intact and apply ESA only to the new increment."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8,
                 normalize_overlap=False):
        super().__init__(
            scale=scale,
            n_feats=n_feats,
            side_c=side_c,
            n_stage=n_stage,
            normalize_overlap=normalize_overlap,
        )
        self.residual_scales = nn.Parameter(torch.ones(n_stage))

    def update_stage(self, previous, delta, stage):
        if previous.shape != delta.shape:
            raise ValueError(
                'F1 stage update requires equal shapes, got {} and {}'.format(
                    tuple(previous.shape), tuple(delta.shape)
                )
            )
        return previous + self.residual_scales[stage] * self.esas[stage](delta)

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for i in range(len(self.blocks)):
            previous = feat
            beta, gamma = self.sfmls[i](fs)
            modulated = beta * previous + gamma
            global_attn, local_attn = self.blocks[i]
            tokens = global_attn(modulated)
            spatial = local_attn(tokens, self.patch_size[i])
            delta = self.mid_convs[i](spatial)
            feat = self.update_stage(previous, delta, i)

        if self.scale == 4:
            upsampled = self.lrelu(
                self.pixel_shuffle(self.upconv1(x0 + feat))
            )
            upsampled = self.lrelu(
                self.pixel_shuffle(self.upconv2(upsampled))
            )
        else:
            upsampled = self.lrelu(
                self.pixel_shuffle(self.upconv(x0 + feat))
            )

        base = F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        return self.last_conv(upsampled) + base


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
