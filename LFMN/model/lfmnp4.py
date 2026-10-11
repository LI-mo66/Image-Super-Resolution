"""P4: shared continuous relative-position bias in original LRSA attention only.

Net.forward, TAB, SFML, LRSA cropping/overlap and all baseline tensor names
are inherited unchanged. Coordinate caches contain no learned tensors.
"""
import math
import weakref
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .lfmn import Net as BaselineNet
from .lfmn import Attention as BaselineAttention


class ContinuousPositionBias(nn.Module):
    def __init__(self, heads=4, hidden=16):
        super().__init__()
        self.heads = int(heads)
        self.logscale = nn.Parameter(torch.full((2,), math.log(math.expm1(1.0))))
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, heads)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.enabled = True
        self._coordinate_cache = OrderedDict()
        self._cache_limit = 4

    def _apply(self, fn, recurse=True):
        # Ordinary cache tensors are intentionally absent from state_dict.
        self._coordinate_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def coordinates(self, patch_size, device):
        p = int(patch_size)
        if not 1 <= p <= 28:
            raise ValueError('P4 cache supports original LRSA patches from 1 to 28')
        key = (p, str(device))
        if key in self._coordinate_cache:
            self._coordinate_cache.move_to_end(key)
            return self._coordinate_cache[key]
        # Evaluation can warm this cache inside inference_mode. Cached tensors
        # must remain ordinary tensors so later training can save them backward.
        with torch.inference_mode(False):
            side = 2 * p - 1
            displacement = torch.arange(1 - p, p, device=device, dtype=torch.float32)
            dy, dx = torch.meshgrid(displacement, displacement, indexing='ij')
            offsets = torch.stack((dx, dy), -1).reshape(-1, 2) / max(p - 1, 1)
            positions = torch.arange(p * p, device=device, dtype=torch.long)
            ys, xs = positions // p, positions % p
            # Query minus key, row-major patch pixels; table order dy then dx.
            index = ((ys[:, None] - ys[None, :] + p - 1) * side
                     + xs[:, None] - xs[None, :] + p - 1)
        self._coordinate_cache[key] = (offsets, index)
        while len(self._coordinate_cache) > self._cache_limit:
            self._coordinate_cache.popitem(last=False)
        return offsets, index

    def forward(self, patch_size, device, dtype):
        offsets, index = self.coordinates(patch_size, device)
        offsets = offsets.to(dtype=self.logscale.dtype)
        scale = F.softplus(self.logscale)
        transformed = offsets.sign() * (-torch.expm1(-offsets.abs() * scale))
        # Recompute this learned table on every call; never cache an autograd graph.
        table = self.fc2(F.relu(self.fc1(transformed)))
        n = int(patch_size) ** 2
        bias = table[index.reshape(-1)].reshape(n, n, self.heads)
        return bias.permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)


class PositionAttention(BaselineAttention):
    def __init__(self, original, position_bias, patch_size):
        nn.Module.__init__(self)
        self.heads = original.heads
        # Reuse original submodules: no reinitialization or renamed baseline keys.
        self.to_q, self.to_k = original.to_q, original.to_k
        self.to_v, self.proj = original.to_v, original.proj
        self.patch_size = int(patch_size)
        self._position_bias_ref = weakref.ref(position_bias)

    def forward(self, x):
        generator = self._position_bias_ref()
        if generator is None:
            raise RuntimeError('Shared P4 position generator no longer exists')
        if x.shape[1] != self.patch_size ** 2:
            raise ValueError('LRSA patch tokens do not match registered row-major coordinates')
        if not generator.enabled:
            return super().forward(x)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        bias = generator(self.patch_size, q.device, q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class Net(BaselineNet):
    def __init__(self, scale=4, n_feats=48, side_c=32, n_stage=8, normalize_overlap=False):
        super().__init__(scale, n_feats, side_c, n_stage, normalize_overlap)
        # Register exactly once; LRSA modules retain only weak references.
        self.position_bias = ContinuousPositionBias(heads=4, hidden=16)
        for stage, (_, lrsa) in enumerate(self.blocks):
            original = lrsa.layer[0].fn
            lrsa.layer[0].fn = PositionAttention(original, self.position_bias, self.patch_size[stage])

    def load_baseline(self, state):
        current = self.state_dict()
        expected = {key for key in current if not key.startswith('position_bias.')}
        supplied = set(state)
        missing, unexpected = expected - supplied, supplied - expected
        if missing or unexpected:
            raise RuntimeError(f'Baseline keys mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        for key in expected:
            if current[key].shape != state[key].shape:
                raise RuntimeError(f'Baseline shape mismatch for {key}: {state[key].shape} vs {current[key].shape}')
        result = self.load_state_dict(state, strict=False)
        added = {key for key in current if key.startswith('position_bias.')}
        if set(result.missing_keys) != added or result.unexpected_keys:
            raise RuntimeError('Unexpected partial baseline loading')
        parameters = dict(self.named_parameters())
        covered_parameters = {key for key in parameters if key in expected}
        return {
            'baseline_key_count': len(expected), 'loaded_baseline_key_count': len(supplied),
            'baseline_parameter_elements': sum(parameters[key].numel() for key in covered_parameters),
            'loaded_baseline_parameter_elements': sum(parameters[key].numel() for key in covered_parameters),
            'baseline_parameter_coverage': 1.0,
            'new_parameter_keys': sorted(added),
            'new_parameter_elements': sum(parameters[key].numel() for key in added),
        }


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 4
    return Net(scale=scale)


def self_check():
    """CPU synthetic small-attention math only; full model checks run on server."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(404)
        generator = ContinuousPositionBias()
        offsets, index = generator.coordinates(3, torch.device('cpu'))
        torch.testing.assert_close(offsets[index[0, 1]], torch.tensor([-0.5, 0.0]))
        torch.testing.assert_close(offsets[index[0, 3]], torch.tensor([0.0, -0.5]))
        torch.testing.assert_close(offsets[index[8, 0]], torch.tensor([1.0, 1.0]))
        original = BaselineAttention(48, 4, 36)
        attention = PositionAttention(original, generator, 3)
        x, target = torch.randn(2, 9, 48), torch.randn(2, 9, 48)
        assert generator(3, x.device, x.dtype).shape == (1, 4, 9, 9)
        assert torch.count_nonzero(generator(3, x.device, x.dtype)) == 0
        torch.testing.assert_close(original(x), attention(x), rtol=1e-5, atol=1e-4)
        generator._coordinate_cache.clear()
        with torch.inference_mode():
            attention(x)
        assert all(not tensor.is_inference() for tensors in generator._coordinate_cache.values()
                   for tensor in tensors)
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
        first_upstream = None
        for step in range(2):
            optimizer.zero_grad()
            loss = (attention(x) - target).square().mean()
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in generator.parameters())
            if step == 0:
                first_upstream = float(generator.fc1.weight.grad.abs().sum())
                assert first_upstream == 0.0
                assert generator.fc2.weight.grad.abs().sum() > 0
            else:
                assert generator.fc1.weight.grad.abs().sum() > 0
                assert generator.logscale.grad.abs().sum() > 0
            optimizer.step()
        assert torch.count_nonzero(generator(3, x.device, x.dtype)) > 0
        assert all(key in ('logscale', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias')
                   for key in generator.state_dict())
        return {'row_major_offsets': True, 'bias_shape': True, 'zero_recovers_attention': True,
                'last_layer_first_step_grad': True, 'upstream_second_step_grad': True,
                'inference_cache_then_training_backward': True,
                'first_upstream_grad_sum': first_upstream, 'full_model_checked': False}
