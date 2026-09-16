"""RDSM-v2 with an identity-initialized direct spatial modulation path.

The first RDSM used a zero-initialized learned scalar at every stage. Although
its router fitted the residual-demand teacher, those scalars kept the applied
correction below 0.16% of the original SFML residual. This variant removes the
scalar bottleneck: a zero-initialized router predicts exactly 0.5 everywhere,
which preserves the baseline output, while a fixed bounded scale gives the
reconstruction loss a non-zero gradient to the router from the first step.
"""
import torch.nn as nn

from model.lfmn import Net as BaselineNet
from model.lfmnrdsm import Net as RDSMNet, ResidualDemandModulator


class DirectResidualDemandModulator(ResidualDemandModulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.stage_gain
        nn.init.zeros_(self.predict.weight)
        nn.init.zeros_(self.predict.bias)

    def strength(self, demand, stage_index):
        del stage_index
        return 1.0 + self.modulation_scale * (2.0 * demand - 1.0)


class Net(RDSMNet):
    def __init__(
            self, scale=2, n_feats=48, side_c=32, n_stage=8,
            rdsm_mid=4, rdsm_kernel=3, rdsm_scale=0.1,
            use_auxiliary=True):
        BaselineNet.__init__(
            self, scale=scale, n_feats=n_feats, side_c=side_c,
            n_stage=n_stage
        )
        self.use_auxiliary = bool(use_auxiliary)
        self.rdsm = DirectResidualDemandModulator(
            side_c=side_c,
            n_feats=n_feats,
            n_stage=n_stage,
            scale=scale,
            mid=rdsm_mid,
            kernel_size=rdsm_kernel,
            modulation_scale=rdsm_scale,
        )


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    loss_spec = getattr(args, 'loss', '')
    use_auxiliary = 'RRES' in loss_spec or 'RDEM' in loss_spec
    return Net(
        scale=scale,
        rdsm_mid=args.rdsm_mid,
        rdsm_kernel=args.rdsm_kernel,
        rdsm_scale=args.rdsm_direct_scale,
        use_auxiliary=use_auxiliary,
    )
