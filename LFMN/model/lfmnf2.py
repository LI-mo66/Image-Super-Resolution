"""F2: fixed-budget LRSA FFN width allocation; original LFMN operations."""
import importlib.util
import torch

_spec = importlib.util.find_spec(__package__ + '.lfmn')
if _spec is None or not _spec.origin.endswith('.py'):
    raise RuntimeError('F2 requires the registered Python lfmn.py baseline, not a compiled extension; match the B0 environment/source')

from .lfmn import Net as OriginalNet, ConvFFN

TAB_WIDTHS = (96,) * 8
LRSA_WIDTHS = (64, 80, 112, 128, 64, 80, 112, 128)


def validate_widths(tab, lrsa):
    for widths in (tab, lrsa):
        if len(widths) != 8 or any(type(v) is not int or v <= 0 or v % 16 for v in widths):
            raise ValueError('F2 requires eight positive integer widths, each divisible by 16')
    if sum(tab) + sum(lrsa) != 1536:
        raise ValueError('F2 total FFN hidden width must remain 1536')


class Net(OriginalNet):
    def __init__(self, scale=4, tab_widths=None, lrsa_widths=None, init_seed=1):
        if scale != 4:
            raise ValueError('F2 is registered for scale 4 only')
        self.tab_widths = tuple(TAB_WIDTHS if tab_widths is None else tab_widths)
        self.lrsa_widths = tuple(LRSA_WIDTHS if lrsa_widths is None else lrsa_widths)
        validate_widths(self.tab_widths, self.lrsa_widths)
        super().__init__(scale=scale)
        # Preserve original initialization of all unchanged parameters/buffers.
        # New FFNs use their actual fan-in. Fork prevents consuming the shared RNG.
        for stage, block in enumerate(self.blocks):
            for kind, width in enumerate((self.tab_widths[stage], self.lrsa_widths[stage])):
                if width == 96:
                    continue
                with torch.random.fork_rng(devices=[]):
                    torch.default_generator.manual_seed(int(init_seed) + 2000 + stage * 2 + kind)
                    replacement = ConvFFN(48, width, 48, kernel_size=5)
                wrapper = block[0].mlp if kind == 0 else block[1].layer[1]
                wrapper.fn = replacement


def make_model(args):
    if args.scale != [4]:
        raise ValueError('F2 requires --scale 4')
    return Net(scale=4, init_seed=args.seed)
