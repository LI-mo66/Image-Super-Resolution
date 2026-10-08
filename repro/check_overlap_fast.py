"""Regression for shared evaluation-only ordered overlap optimization."""
import copy
import io
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'LFMN'))
from model.lfmn import patch_divide, patch_reverse
from model.overlap_fast import OrderedOverlap
from model.lfmn_exact_overlap import Net as BaselineNet
from model.lfmn_n23 import Net


def disable(net):
    for block in net.blocks:
        block[1].fast_eval_enabled = False


def main():
    torch.manual_seed(41)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {'purpose': 'EQUIVALENT_EVAL_IMPLEMENTATION_ONLY', 'optimizer_steps': 0,
              'geometry_cases': 0, 'full_network': {}}
    for device, dtype in [('cpu', torch.float64), ('cuda', torch.float32)]:
        engine = OrderedOverlap().to(device)
        for height, width in [(33, 47), (64, 64), (65, 97), (128, 128)]:
            for window in (8, 16, 20, 24, 28, 32, 64):
                if min(height, width) < window:
                    continue
                image = torch.randn(2, 3, height, width, device=device, dtype=dtype)
                old, _, _ = patch_divide(image, window-2, window)
                new = engine.extract(image, window)
                assert torch.equal(old, new), ('extract differs', device, height, width, window)
                arbitrary = torch.randn_like(old, requires_grad=True)
                old_reverse = patch_reverse(arbitrary, image, window-2, window, normalize_overlap=True)
                new_reverse = engine.reverse(arbitrary, image, window)
                assert torch.equal(old_reverse, new_reverse), ('ordered reverse differs', device, height, width, window)
                probe = torch.randn_like(old_reverse)
                old_grad = torch.autograd.grad((old_reverse*probe).sum(), arbitrary)[0]
                new_grad = torch.autograd.grad((new_reverse*probe).sum(), arbitrary)[0]
                assert torch.equal(old_grad, new_grad), ('reverse gradient differs', device, height, width, window)
                assert engine.cache_size == 1, 'geometry cache grows across input shapes'
                report['geometry_cases'] += 1
        # NaN at a real contribution must affect only the same output pixels,
        # not absent slots which use clamped lookup indices.
        image = torch.ones(1, 1, 33, 47, device=device, dtype=dtype)
        crops, _, _ = patch_divide(image, 6, 8)
        crops[0, 0, 0, 0, 0] = float('nan')
        ref_nan = patch_reverse(crops, image, 6, 8, normalize_overlap=True)
        got_nan = engine.reverse(crops, image, 8)
        assert torch.equal(torch.isnan(ref_nan), torch.isnan(got_nan))
        assert torch.equal(torch.nan_to_num(ref_nan), torch.nan_to_num(got_nan))
        assert engine.cache_size > 0
        engine.cpu()
        assert engine.cache_size == 0, 'device conversion retained old caches'

    device = torch.device('cuda')
    for label, factory in [('b0', BaselineNet), ('n23', Net)]:
        torch.manual_seed(7)
        fast = factory(scale=4).to(device)
        # Static deployment-like EMA buffers, no training or optimizer call.
        for block in fast.blocks:
            block[0].initted.fill_(True)
        slow = copy.deepcopy(fast)
        disable(slow)
        fast.eval()
        slow.eval()
        assert set(fast.state_dict()) == set(slow.state_dict())
        assert all(torch.equal(v, slow.state_dict()[k]) for k, v in fast.state_dict().items())
        errors = []
        with torch.no_grad():
            for height, width in [(33, 47), (64, 64), (96, 160)]:
                image = torch.rand(1, 3, height, width, device=device)*255
                y0, y1 = slow(image), fast(image)
                assert torch.isfinite(y0).all() and torch.isfinite(y1).all()
                error = float((y0-y1).abs().max())
                assert error == 0, (label, height, width, error)
                errors.append(error)
            # Disabling exact normalization must fall back to the legacy graph,
            # even with the inference fast switch still enabled.
            for model in (slow, fast):
                for block in model.blocks:
                    block[1].normalize_overlap = False
            probe_image = torch.rand(1, 3, 33, 47, device=device)*255
            assert torch.equal(slow(probe_image), fast(probe_image)), 'legacy diagnostic flag ignored'
            for model in (slow, fast):
                for block in model.blocks:
                    block[1].normalize_overlap = True
        assert any(block[1].fast_patches.cache_size for block in fast.blocks)
        buffer = io.BytesIO()
        torch.save(fast.state_dict(), buffer)
        buffer.seek(0)
        reload = factory(scale=4).to(device).eval()
        reload.load_state_dict(torch.load(buffer, weights_only=True, map_location=device), strict=True)
        with torch.no_grad():
            assert torch.equal(fast(image), reload(image))
        # eval+grad and train must not construct or use fast caches.
        fast.cpu()
        slow.cpu()
        assert all(block[1].fast_patches.cache_size == 0 for block in fast.blocks)
        torch.manual_seed(17)
        x0 = (torch.rand(1, 3, 64, 64)*255).requires_grad_(True)
        x1 = x0.detach().clone().requires_grad_(True)
        a, b = slow(x0), fast(x1)
        assert torch.equal(a, b)
        a.mean().backward()
        b.mean().backward()
        assert torch.equal(x0.grad, x1.grad)
        assert all(block[1].fast_patches.cache_size == 0 for block in fast.blocks)
        fast.zero_grad(set_to_none=True)
        slow.zero_grad(set_to_none=True)
        fast.train()
        slow.train()
        x0 = x0.detach().requires_grad_(True)
        x1 = x1.detach().requires_grad_(True)
        a, b = slow(x0), fast(x1)
        assert torch.equal(a, b)
        a.mean().backward()
        b.mean().backward()
        assert torch.equal(x0.grad, x1.grad)
        for (n0, p0), (n1, p1) in zip(slow.named_parameters(), fast.named_parameters()):
            assert n0 == n1 and (p0.grad is None) == (p1.grad is None)
            if p0.grad is not None:
                assert torch.equal(p0.grad, p1.grad), n0
        assert all(block[1].fast_patches.cache_size == 0 for block in fast.blocks)
        report['full_network'][label] = {'eval_output_errors': errors,
                                        'grad_enabled_eval_and_training': 'UNCHANGED',
                                        'strict_reload': 'PASS', 'cache_device_release': 'PASS'}
        del fast, slow, reload, a, b, x0, x1
        torch.cuda.empty_cache()
    report['status'] = 'PASS; compile/CUDA graphs/AMP unverified'
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
