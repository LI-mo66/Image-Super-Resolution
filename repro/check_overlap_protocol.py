"""Audit exact overlap averaging without optimizer steps or checkpoint writes."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet, patch_divide, patch_reverse
from model.lfmn_n23 import Net as CandidateNet
import model.lfmn as core


def coordinates(height, width, window):
    step = window - 2
    return [(min(y, height-window), min(x, width-window))
            for y in range(0, height+step-window, step)
            for x in range(0, width+step-window, step)]


def scatter_reference(patches, height, width, window):
    """Independent indexed accumulation; same patch order, no legacy reverse."""
    b, n, c, _, _ = patches.shape
    positions = coordinates(height, width, window)
    assert n == len(positions)
    offsets = torch.arange(window, device=patches.device)
    indices = torch.stack([((y+offsets[:, None])*width+x+offsets[None, :]).flatten()
                           for y, x in positions]).flatten()
    values = patches.permute(0, 2, 1, 3, 4).reshape(b, c, -1)
    result = torch.zeros(b, c, height*width, device=patches.device, dtype=patches.dtype)
    result.scatter_add_(2, indices[None, None].expand(b, c, -1), values)
    coverage = torch.bincount(indices, minlength=height*width).to(patches.dtype)
    assert (coverage > 0).all()
    return (result/coverage).reshape(b, c, height, width), coverage.reshape(height, width)


def set_mode(net, exact):
    modules = [block[1] for block in net.blocks]
    assert len(modules) == 8
    for module in modules:
        module.normalize_overlap = bool(exact)


def max_difference(a, b):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return float((a-b).abs().max().detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(31)
    report = {'purpose': 'ENGINEERING_ONLY', 'optimizer_steps': 0,
              'fixed_training_lr_hw': [64, 64], 'geometry': [], 'full_network': {}}
    training_cases = 0
    for height, width in [(33, 47), (64, 64), (65, 97), (64, 97), (97, 64)]:
        for window in (8, 16, 20, 24, 28, 32, 64):
            if min(height, width) < window:
                continue
            image = torch.randn(2, 3, height, width, dtype=torch.float64)
            patches, _, _ = patch_divide(image, window-2, window)
            # Test arbitrary patch responses, not just patches of one image.
            patches = torch.randn_like(patches, requires_grad=True)
            exact = patch_reverse(patches, image, window-2, window, normalize_overlap=True)
            reference, coverage = scatter_reference(patches, height, width, window)
            reference_error = max_difference(exact, reference)
            assert reference_error < 1e-12, reference_error
            probe = torch.randn_like(exact)
            exact_grad = torch.autograd.grad((exact*probe).sum(), patches)[0]
            reference_grad = torch.autograd.grad((reference*probe).sum(), patches)[0]
            assert max_difference(exact_grad, reference_grad) < 1e-12
            ones = torch.ones_like(image)
            unit_patches, _, _ = patch_divide(ones, window-2, window)
            unit_exact = patch_reverse(unit_patches, ones, window-2, window, normalize_overlap=True)
            assert torch.equal(unit_exact, ones)
            legacy = patch_reverse(patches, image, window-2, window)
            legacy_unit = patch_reverse(unit_patches, ones, window-2, window)
            row = {'hw': [height, width], 'window': window,
                   'reference_max': reference_error,
                   'legacy_unit_error': max_difference(legacy_unit, ones),
                   'coverage_values': coverage.unique().tolist()}
            if height == width == 64:
                legacy_grad = torch.autograd.grad((legacy*probe).sum(), patches)[0]
                assert torch.equal(legacy, exact), ('training forward changed', window)
                assert torch.equal(legacy_grad, exact_grad), ('training gradient changed', window)
                row['training_forward_and_gradient_bitwise_equal'] = True
                training_cases += 1
            report['geometry'].append(row)
    assert training_cases == 7
    report['gpu_operator_equivalence'] = []
    if args.device.startswith('cuda'):
        for window in (8, 16, 20, 24, 28, 32, 64):
            image = torch.randn(2, 3, 64, 64, device=args.device)
            patches, _, _ = patch_divide(image, window-2, window)
            patches = torch.randn_like(patches, requires_grad=True)
            old = patch_reverse(patches, image, window-2, window)
            new = patch_reverse(patches, image, window-2, window, normalize_overlap=True)
            probe = torch.randn_like(old)
            go = torch.autograd.grad((old*probe).sum(), patches)[0]
            gn = torch.autograd.grad((new*probe).sum(), patches)[0]
            assert torch.equal(old, new) and torch.equal(go, gn), window
            report['gpu_operator_equivalence'].append({'window': window, 'bitwise': True})

    lr_path = args.data/'DIV2K_train_LR_bicubic/X4/0001x4.png'
    hr_path = args.data/'DIV2K_train_HR/0001.png'
    assert lr_path.is_file() and hr_path.is_file()
    lr = torch.from_numpy(imageio.imread(lr_path).copy()).permute(2, 0, 1).float()[None, :, :64, :64]
    hr = torch.from_numpy(imageio.imread(hr_path).copy()).permute(2, 0, 1).float()[None, :, :256, :256]
    device = torch.device(args.device)
    lr, hr = lr.to(device), hr.to(device)
    for label, factory in [('B0', BaselineNet), ('N23', CandidateNet)]:
        torch.manual_seed(7)
        original = factory(scale=4).to(device)
        # Initialize original TAB buffers once on a real batch, no optimizer.
        with torch.no_grad():
            original.train()
            original(lr)
        fixed = copy.deepcopy(original)
        set_mode(original, False)
        set_mode(fixed, True)
        assert set(original.state_dict()) == set(fixed.state_dict())
        assert all(torch.equal(v, fixed.state_dict()[k]) for k, v in original.state_dict().items())
        original.train()
        fixed.train()
        x0 = lr.clone().requires_grad_(True)
        x1 = lr.clone().requires_grad_(True)
        # Freeze stochastic center buffers by using identical initialized copies;
        # the only intentional intervention is exact overlap averaging.
        y0, y1 = original(x0), fixed(x1)
        forward_error = max_difference(y0, y1)
        F.l1_loss(y0, hr).backward()
        F.l1_loss(y1, hr).backward()
        input_error = max_difference(x0.grad, x1.grad)
        grad_errors = []
        for (name0, p0), (name1, p1) in zip(original.named_parameters(), fixed.named_parameters()):
            assert name0 == name1
            assert (p0.grad is None) == (p1.grad is None)
            if p0.grad is not None:
                grad_errors.append(max_difference(p0.grad, p1.grad))
        gradient_error = max(grad_errors)
        # GPU center reductions may be non-bitwise deterministic. Do not weaken
        # a hard equivalence check by accepting that as an unexplained drift.
        if max(forward_error, input_error, gradient_error) > 0:
            report['full_network'][label] = {'forward_max': forward_error,
                                            'input_gradient_max': input_error,
                                            'parameter_gradient_max': gradient_error,
                                            'status': 'NONBITWISE_REQUIRES_ROUTE_ISOLATION'}
        else:
            report['full_network'][label] = {'forward_max': 0, 'input_gradient_max': 0,
                                            'parameter_gradient_max': 0, 'status': 'BITWISE_PASS'}
        if device.type == 'cuda':
            # Diagnostic only: replay discrete cluster assignments, not detached
            # centers. IRCA's third center update is differentiable and its
            # original scatter/normalization gradient MUST be preserved.
            # No production model function or on-disk source is changed.
            fixed.load_state_dict(original.state_dict(), strict=True)
            original.zero_grad(set_to_none=True)
            fixed.zero_grad(set_to_none=True)
            centers = []
            original_iter = core.center_iter
            previous_deterministic = torch.backends.cudnn.deterministic
            previous_algorithms = torch.are_deterministic_algorithms_enabled()
            def capture(x, means, buckets=None):
                if buckets is None:
                    with torch.no_grad():
                        _, buckets = core.dists_and_buckets(x, means)
                result = original_iter(x, means, buckets)
                centers.append((x.detach().clone(), means.detach().clone(),
                                buckets.detach().clone(), result.detach().clone()))
                return result
            position = [0]
            def replay(x, means, buckets=None):
                expected_x, expected_means, assignment, expected_result = centers[position[0]]
                assert torch.equal(x, expected_x) and torch.equal(means, expected_means), \
                    ('different input before center update', label, position[0])
                position[0] += 1
                result = original_iter(x, means, assignment)
                assert torch.equal(result, expected_result), 'fixed-assignment centers changed'
                return result
            try:
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                core.center_iter = capture
                a = lr.clone().requires_grad_(True)
                old_output = original(a)
                core.center_iter = replay
                b = lr.clone().requires_grad_(True)
                new_output = fixed(b)
                assert position[0] == len(centers) == 24
                assert torch.equal(old_output, new_output), 'fixed-center forward changed'
                F.l1_loss(old_output, hr).backward()
                F.l1_loss(new_output, hr).backward()
                assert torch.equal(a.grad, b.grad), 'fixed-center input gradient changed'
                for (n0, p0), (n1, p1) in zip(original.named_parameters(), fixed.named_parameters()):
                    assert n0 == n1 and (p0.grad is None) == (p1.grad is None)
                    if p0.grad is not None:
                        assert torch.equal(p0.grad, p1.grad), ('fixed-center parameter gradient changed', n0)
                report['full_network'][label]['fixed_assignments_gpu'] = 'BITWISE_FORWARD_AND_GRADIENT_PASS'
            finally:
                core.center_iter = original_iter
                torch.backends.cudnn.deterministic = previous_deterministic
                torch.use_deterministic_algorithms(previous_algorithms)
            del a, b, old_output, new_output, centers
        elif max(forward_error, input_error, gradient_error) != 0:
            raise AssertionError('CPU training equivalence failed')
        del original, fixed, x0, x1, y0, y1
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    report['scope'] = ('Training overlap operator equivalence only for LR64 square and listed windows; '
                       'historical metric curves are NOT reusable after evaluation correction; '
                       'checkpoint reuse also requires complete protocol audit.')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
