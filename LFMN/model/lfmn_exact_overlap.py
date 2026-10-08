"""Explicit B0 engineering profile: exact coverage, no new model parameters.

Old full-image metrics are NOT reusable. Old weights may be reused only after
proving this intervention does not affect their fixed LR64 training protocol.
Original model.lfmn and its default legacy behavior are deliberately unchanged.
"""
from .lfmn import Net as BaselineNet


class Net(BaselineNet):
    def __init__(self, scale=4, normalize_overlap=True, **kwargs):
        if not normalize_overlap:
            raise ValueError('This named B0 profile requires exact coverage')
        super().__init__(scale=scale, normalize_overlap=True, **kwargs)


def make_model(args):
    return Net(scale=args.scale[0])
