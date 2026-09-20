"""Near-parameter-matched N9/A1 PCSTR control."""
from model.lfmnpcstr import Net as PCSTRNet


class Net(PCSTRNet):
    """PCSTR with the retained ConvFFN widened from 96 to 123 channels."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale,
            n_feats=n_feats,
            side_c=side_c,
            n_stage=n_stage,
            mlp_dim=123,
        )


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
