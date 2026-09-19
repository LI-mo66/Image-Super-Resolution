"""LFMN with a parameter-reducing cross-axis LRSA feed-forward mixer."""
import torch
import torch.nn as nn

from model.lfmn import Net as BaselineNet


class CrossAxisDepthwiseConv(nn.Module):
    """Replace 5x5 DWConv by mutually gated horizontal/vertical strips."""

    def __init__(self, hidden_features, kernel_size=11):
        super().__init__()
        if hidden_features % 2:
            raise ValueError('hidden_features must be even')
        if kernel_size % 2 != 1:
            raise ValueError('kernel_size must be odd')
        half = hidden_features // 2
        padding = kernel_size // 2
        self.hidden_features = hidden_features
        self.horizontal = nn.Conv2d(
            half, half, (1, kernel_size), padding=(0, padding),
            groups=half,
        )
        self.vertical = nn.Conv2d(
            half, half, (kernel_size, 1), padding=(padding, 0),
            groups=half,
        )
        self.enabled = True

    def forward(self, x, x_size):
        if not self.enabled:
            return torch.zeros_like(x)
        height, width = x_size
        feature = x.transpose(1, 2).reshape(
            x.shape[0], self.hidden_features, height, width
        ).contiguous()
        horizontal_input, vertical_input = feature.chunk(2, dim=1)
        horizontal = self.horizontal(horizontal_input)
        vertical = self.vertical(vertical_input)
        # Each orientation controls the other.  This preserves directionality
        # while suppressing responses unsupported by the orthogonal context.
        mixed = torch.cat(
            (
                horizontal * torch.sigmoid(vertical),
                vertical * torch.sigmoid(horizontal),
            ),
            dim=1,
        )
        return mixed.flatten(2).transpose(1, 2).contiguous()


class Net(BaselineNet):
    """Replace only the post-reassembly LRSA FFN spatial mixer."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        for _, local_attn in self.blocks:
            conv_ffn = local_attn.layer[1].fn
            conv_ffn.dwconv = CrossAxisDepthwiseConv(
                conv_ffn.dwconv.hidden_features, kernel_size=11
            )


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
