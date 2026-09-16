"""LFMN control variant with coverage-normalized LRSA patch stitching.

This is a correctness/control experiment, not an architectural innovation.
It has exactly the same trainable state as the baseline LFMN.
"""
from model.lfmn import Net as BaselineNet


class Net(BaselineNet):
    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8):
        super().__init__(
            scale=scale,
            n_feats=n_feats,
            side_c=side_c,
            n_stage=n_stage,
            normalize_overlap=True,
        )


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
