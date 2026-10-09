"""N24: structured hidden-channel pruning for every V1 ConvFFN.

The candidate preserves the complete LFMN+SRPRv2 forward graph and only
replaces the 16 internal ``48 -> 96 -> 48`` ConvFFNs with genuinely narrower
Linear/DWConv/Linear modules.  No mask remains in the inference graph.
"""
from collections import OrderedDict

import torch

from model.lfmn import ConvFFN
from model.lfmnsrprv2 import Net as V1Net


BASELINE_FFN_HIDDEN = 96
DEFAULT_FFN_HIDDEN = 88


def _ffn_modules(model):
    """Yield stable state-dict prefixes and all TAB/LRSA ConvFFNs."""
    for stage, block in enumerate(model.blocks):
        yield f'blocks.{stage}.0.mlp.fn', block[0].mlp.fn
        yield f'blocks.{stage}.1.layer.1.fn', block[1].layer[1].fn


def _normalized(component):
    component = component.detach().float()
    return component / component.mean().clamp_min(torch.finfo(component.dtype).eps)


def magnitude_scores(ffn):
    """Return a scale-balanced magnitude score for each hidden channel."""
    conv = ffn.dwconv.depthwise_conv[0]
    components = [
        ffn.fc1.weight.abs().mean(dim=1),
        ffn.fc1.bias.abs(),
        conv.weight.abs().flatten(1).mean(dim=1),
        conv.bias.abs(),
        ffn.fc2.weight.abs().mean(dim=0),
    ]
    return sum(_normalized(component) for component in components)


def select_magnitude_indices(ffn, keep):
    """Select top-scoring channels and return them in source channel order."""
    hidden = ffn.fc1.out_features
    if not 1 <= keep <= hidden:
        raise ValueError(f'keep must be in [1, {hidden}], got {keep}')
    ranking = torch.argsort(
        magnitude_scores(ffn), descending=True, stable=True
    )[:keep]
    return torch.sort(ranking).values.cpu()


class Net(V1Net):
    """V1 with a real, uniform ConvFFN hidden width below 96."""

    def __init__(
        self,
        scale=4,
        n_feats=48,
        side_c=32,
        n_stage=8,
        normalize_overlap=False,
        ffn_hidden=DEFAULT_FFN_HIDDEN,
    ):
        if not 1 <= int(ffn_hidden) <= BASELINE_FFN_HIDDEN:
            raise ValueError(
                f'ffn_hidden must be in [1, {BASELINE_FFN_HIDDEN}]'
            )
        self.ffn_hidden = int(ffn_hidden)
        super().__init__(
            scale=scale,
            n_feats=n_feats,
            side_c=side_c,
            n_stage=n_stage,
            normalize_overlap=normalize_overlap,
        )
        # The replacement weights are always overwritten by migration for the
        # pretrained candidate.  Preserve the global RNG stream so constructing
        # N24 consumes exactly the same random numbers as constructing V1.
        with torch.random.fork_rng(devices=[]):
            for block in self.blocks:
                block[0].mlp.fn = ConvFFN(n_feats, self.ffn_hidden)
                block[1].layer[1].fn = ConvFFN(n_feats, self.ffn_hidden)
        self.last_migration_report = None

    def load_pruned_from_v1(self, state_dict, selected_indices=None):
        """Strictly validate a V1 checkpoint and inherit selected FFN channels.

        ``selected_indices`` may map each stable FFN prefix returned by
        ``_ffn_modules`` to explicit source indices.  When omitted, a
        deterministic, per-FFN scale-balanced magnitude ranking is used.
        """
        # Checkpoint validation must not perturb the caller's training/data RNG.
        with torch.random.fork_rng(devices=[]):
            source = V1Net(
                scale=self.scale,
                n_feats=self.state_channels,
                n_stage=len(self.blocks),
                normalize_overlap=self.blocks[0][1].normalize_overlap,
            )
        strict_result = source.load_state_dict(state_dict, strict=True)
        if strict_result.missing_keys or strict_result.unexpected_keys:
            raise RuntimeError('strict V1 checkpoint validation unexpectedly failed')

        source_state = source.state_dict()
        target_state = self.state_dict()
        shape_mismatches = {
            key for key, value in target_state.items()
            if source_state[key].shape != value.shape
        }
        expected_mismatches = set()
        if self.ffn_hidden != BASELINE_FFN_HIDDEN:
            for prefix, _ in _ffn_modules(self):
                expected_mismatches.update({
                    f'{prefix}.fc1.weight',
                    f'{prefix}.fc1.bias',
                    f'{prefix}.dwconv.depthwise_conv.0.weight',
                    f'{prefix}.dwconv.depthwise_conv.0.bias',
                    f'{prefix}.fc2.weight',
                })
        if shape_mismatches != expected_mismatches:
            missing = sorted(expected_mismatches - shape_mismatches)
            extra = sorted(shape_mismatches - expected_mismatches)
            raise RuntimeError(
                f'unexpected migration shapes; missing={missing}, extra={extra}'
            )

        compatible = {
            key: value for key, value in source_state.items()
            if key not in shape_mismatches
        }
        load_result = self.load_state_dict(compatible, strict=False)
        if set(load_result.missing_keys) != expected_mismatches:
            raise RuntimeError(
                f'unexpected missing migration keys: {load_result.missing_keys}'
            )
        if load_result.unexpected_keys:
            raise RuntimeError(
                f'unexpected checkpoint keys: {load_result.unexpected_keys}'
            )

        requested = selected_indices or {}
        unknown = set(requested) - {name for name, _ in _ffn_modules(source)}
        if unknown:
            raise ValueError(f'unknown FFN prefixes: {sorted(unknown)}')

        chosen = OrderedDict()
        target_by_name = dict(_ffn_modules(self))
        with torch.no_grad():
            for prefix, source_ffn in _ffn_modules(source):
                if prefix in requested:
                    index = torch.as_tensor(requested[prefix], dtype=torch.long)
                else:
                    index = select_magnitude_indices(
                        source_ffn, self.ffn_hidden
                    )
                if index.ndim != 1 or index.numel() != self.ffn_hidden:
                    raise ValueError(
                        f'{prefix} must select exactly {self.ffn_hidden} channels'
                    )
                if torch.unique(index).numel() != index.numel():
                    raise ValueError(f'{prefix} contains duplicate indices')
                if index.min().item() < 0 or index.max().item() >= BASELINE_FFN_HIDDEN:
                    raise ValueError(f'{prefix} contains an out-of-range index')
                index = torch.sort(index).values
                target_ffn = target_by_name[prefix]
                source_conv = source_ffn.dwconv.depthwise_conv[0]
                target_conv = target_ffn.dwconv.depthwise_conv[0]
                target_ffn.fc1.weight.copy_(source_ffn.fc1.weight[index])
                target_ffn.fc1.bias.copy_(source_ffn.fc1.bias[index])
                target_conv.weight.copy_(source_conv.weight[index])
                target_conv.bias.copy_(source_conv.bias[index])
                target_ffn.fc2.weight.copy_(source_ffn.fc2.weight[:, index])
                chosen[prefix] = index.tolist()

        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        self.last_migration_report = {
            'source_strict_missing': [],
            'source_strict_unexpected': [],
            'target_expected_shape_mismatches': len(expected_mismatches),
            'target_parameter_elements': total_parameters,
            'target_parameter_elements_covered': total_parameters,
            'target_parameter_coverage': 1.0,
            'ffn_hidden': self.ffn_hidden,
            'selection': 'explicit' if selected_indices is not None else 'magnitude',
            'selected_indices': chosen,
        }
        return self.last_migration_report


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 4
    hidden = getattr(args, 'n24_ffn_hidden', DEFAULT_FFN_HIDDEN)
    return Net(scale=scale, ffn_hidden=hidden)
