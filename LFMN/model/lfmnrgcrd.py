"""LFMN with training-only feature taps for RGCRD.

There are no additional trainable parameters.  Evaluation returns exactly the
same tensor as the original LFMN; only training exposes stage-4/stage-8 maps.
"""
import torch.nn.functional as F

from model.lfmn import Net as LFMNNet


class Net(LFMNNet):
    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        stage4 = None
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            fm = beta * prev + gamma
            g_attn, l_attn = self.blocks[i]
            t = g_attn(fm)
            s = l_attn(t, self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
            if i == 3:
                stage4 = feat

        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        base = F.interpolate(
            x, scale_factor=self.scale, mode='bilinear', align_corners=False
        )
        sr = self.last_conv(u) + base
        if self.training:
            return sr, {'stage4': stage4, 'stage8': feat}
        return sr


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
