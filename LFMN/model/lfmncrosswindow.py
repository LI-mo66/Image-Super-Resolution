"""LFMN with shared cross-window directional feature exchange."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lfmn import Net as BaselineNet


class CrossWindowDirectionalExchange(nn.Module):
    """Shared long-axis mixer with stage-specific channel gains."""

    def __init__(self, channels=48, stages=8, kernel_size=11):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')
        padding = kernel_size // 2
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1,
            groups=channels, bias=False,
        )
        self.horizontal = nn.Conv2d(
            channels, channels, (1, kernel_size),
            padding=(0, padding), groups=channels, bias=False,
        )
        self.vertical = nn.Conv2d(
            channels, channels, (kernel_size, 1),
            padding=(padding, 0), groups=channels, bias=False,
        )
        self.fuse = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.stage_gain = nn.Parameter(
            torch.zeros(stages, channels, 1, 1)
        )
        self.enabled = True

    def forward(self, x, stage):
        if not self.enabled:
            return torch.zeros_like(x)
        local = F.gelu(self.local(x))
        directional = torch.cat(
            (self.horizontal(local), self.vertical(local)), dim=1
        )
        exchange = self.fuse(F.gelu(directional))
        return torch.tanh(self.stage_gain[stage]) * exchange


class Net(BaselineNet):
    """Insert one shared directional exchange after every LRSA reassembly."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        self.cross_window = CrossWindowDirectionalExchange(
            channels=n_feats, stages=n_stage, kernel_size=11
        )

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for index, (global_attn, local_attn) in enumerate(self.blocks):
            prev = feat
            beta, gamma = self.sfmls[index](fs)
            modulated = beta * prev + gamma
            transformed = local_attn(
                global_attn(modulated), self.patch_size[index]
            )
            transformed = transformed + self.cross_window(
                transformed, index
            )
            feat = self.esas[index](
                prev + self.mid_convs[index](transformed)
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
        return self.last_conv(upsampled) + base


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
