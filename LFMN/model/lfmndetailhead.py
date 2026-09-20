"""LFMN with a direct high-frequency detail reconstruction path."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class DetailResidualBlock(nn.Module):
    """Cheap local refinement without changing spatial resolution."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        return x + self.pointwise(F.gelu(self.depthwise(x)))


class DirectDetailHead(nn.Module):
    """Predict an HR RGB residual and its spatial confidence from LR detail."""

    def __init__(self, shallow_channels=48, prior_channels=32, mid_channels=24,
                 scale=4):
        super().__init__()
        self.scale = int(scale)
        input_channels = shallow_channels + prior_channels
        self.reduce = nn.Conv2d(input_channels, mid_channels, 1)
        self.refine = nn.Sequential(
            DetailResidualBlock(mid_channels),
            DetailResidualBlock(mid_channels),
        )
        # PixelShuffle needs 3*s^2 residual channels and s^2 gate channels.
        self.expand = nn.Conv2d(
            mid_channels, 4 * self.scale * self.scale, 1
        )
        self.shuffle = nn.PixelShuffle(self.scale)
        self.residual_scale = 0.1

    @staticmethod
    def high_pass(x):
        local_mean = F.avg_pool2d(
            F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, stride=1
        )
        return x - local_mean

    def forward(self, shallow, prior):
        detail = torch.cat(
            (self.high_pass(shallow), self.high_pass(prior)), dim=1
        )
        detail = F.gelu(self.reduce(detail))
        detail = self.expand(self.refine(detail))
        detail = self.shuffle(detail)
        residual, gate = detail[:, :3], detail[:, 3:]
        # A factor centred on one keeps the direct path trainable everywhere
        # while still allowing spatial suppression or amplification.
        return self.residual_scale * residual * (2.0 * torch.sigmoid(gate))


class Net(BaselineNet):
    """Progressive baseline reconstruction plus an LR-detail direct path."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        self.detail_head = DirectDetailHead(
            shallow_channels=n_feats,
            prior_channels=side_c,
            mid_channels=24,
            scale=scale,
        )
        self.detail_enabled = True

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            fm = beta * prev + gamma
            global_attn, local_attn = self.blocks[i]
            transformed = global_attn(fm)
            transformed = local_attn(transformed, self.patch_size[i])
            feat = self.esas[i](
                prev + self.mid_convs[i](transformed)
            )

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
        output = self.last_conv(upsampled) + base
        if self.detail_enabled:
            output = output + self.detail_head(x0, fs)
        return output


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
