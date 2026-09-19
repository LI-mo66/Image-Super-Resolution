"""LFMN with input-adaptive TAB prototype refinement at evaluation time."""
from model.lfmn import Net as BaselineNet


class Net(BaselineNet):
    """Refine stored TAB prototypes from each test image without new weights."""

    def __init__(self, scale=2, n_feats=48, side_c=32, n_stage=8,
                 token_refine_iters=3):
        super().__init__(
            scale=scale, n_feats=n_feats, side_c=side_c, n_stage=n_stage
        )
        token_refine_iters = int(token_refine_iters)
        if token_refine_iters < 1:
            raise ValueError('token_refine_iters must be at least one')
        self.token_refine_iters = token_refine_iters
        for global_attention, _ in self.blocks:
            global_attention.eval_refine_iters = token_refine_iters


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(
        scale=scale,
        token_refine_iters=getattr(args, 'token_refine_iters', 3),
    )
