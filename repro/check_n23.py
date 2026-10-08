"""Read-only engineering checks: no optimizer, training run or PSNR claim."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet, patch_divide, patch_reverse
from model.lfmn_n23 import Net
from model.n23_scc import SCC


def count(module):
    return sum(p.numel() for p in module.parameters())


def finite(tensor):
    assert torch.isfinite(tensor).all(), 'nonfinite tensor'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True, help='DIV2K directory')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = {'purpose': 'ENGINEERING_ONLY_NOT_ACCURACY', 'optimizer_steps': 0,
              'torch': torch.__version__, 'device': args.device}

    # Equivalence of low-memory positional aggregation, including its gradient.
    torch.manual_seed(23)
    primitive = SCC(16).double()
    fast, direct = primitive.position_bias(), primitive.reference_bias()
    error = float((fast-direct).abs().max().detach())
    assert error < 1e-10, error
    probe = torch.randn_like(fast)
    params = tuple(primitive.pos.parameters())
    gf = torch.autograd.grad((fast*probe).sum(), params)
    gd = torch.autograd.grad((direct*probe).sum(), params)
    grad_error = max(float((a-b).abs().max()) for a, b in zip(gf, gd))
    assert grad_error < 1e-8, grad_error
    report['bias_reference'] = {'forward_max': error, 'gradient_max': grad_error}
    torch.manual_seed(37)
    large = SCC(64)
    table = large._offset_table()
    positions = [(0, 0, 0, 0), (63, 63, 7, 7), (0, 63, 0, 7),
                 (63, 0, 7, 0), (31, 32, 3, 4)]
    fast_large = large.position_bias()
    direct_values, fast_values = [], []
    for qy, qx, ay, ax in positions:
        sy = torch.arange(ay*8, ay*8+8)
        sx = torch.arange(ax*8, ax*8+8)
        direct_values.append(table[qy-sy[:, None]+63, qx-sx[None, :]+63].mean((0, 1)))
        fast_values.append(fast_large[:, qy*64+qx, ay*8+ax])
    direct_values, fast_values = torch.stack(direct_values), torch.stack(fast_values)
    large_error = float((direct_values-fast_values).abs().max().detach())
    assert large_error < 1e-4, large_error
    weight = torch.randn_like(direct_values)
    gp = tuple(large.pos.parameters())
    lgf = torch.autograd.grad((fast_values*weight).sum(), gp)
    lgd = torch.autograd.grad((direct_values*weight).sum(), gp)
    large_grad_error = max(float((a-b).abs().max()) for a, b in zip(lgf, lgd))
    assert large_grad_error < 1e-4, large_grad_error
    report['p64_float32_spot_bias'] = {'positions': len(positions),
                                     'forward_max': large_error,
                                     'gradient_max': large_grad_error}
    try:
        SCC(64).reference_bias()
        raise AssertionError('quadratic reference not blocked')
    except ValueError:
        pass
    del primitive, fast, direct, gf, gd

    torch.manual_seed(7)
    baseline = BaselineNet(scale=4)
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    candidate = Net(scale=4)
    assert all(block[1].normalize_overlap for block in candidate.blocks), \
        'N23 must share exact coverage with the named corrected B0'
    assert torch.equal(torch.get_rng_state(), baseline_rng), 'data RNG changed'
    before, after = baseline.state_dict(), candidate.state_dict()
    common = {k for k in set(before) & set(after) if '.1.layer.0.fn.' not in k}
    assert all(torch.equal(before[k], after[k]) for k in common), 'common init differs'
    assert count(baseline) == 759627 and count(candidate) == 746733
    removed = sorted(set(before)-set(after))
    # New projection intentionally shares the old key name, not the old tensor.
    assert len(removed) == 24 and all('.1.layer.0.fn.' in k for k in removed)
    report['structure'] = {'baseline_params': count(baseline),
                           'candidate_params': count(candidate),
                           'common_keys_equal': len(common),
                           'removed_old_qkv_keys': len(removed),
                           'replaced_old_attention_keys': 32}
    device = torch.device(args.device)
    candidate = candidate.to(device)
    del before, after, baseline

    # A finite forward is not enough: zero-response residual reconstruction
    # must preserve a constant. Keep inherited legacy behavior, report failure,
    # and do not silently change the comparison protocol to make tests pass.
    overlap = []
    for height, width in [(33, 47), (64, 64), (65, 97)]:
        for window in (8, 16, 20, 24, 28, 32, 64):
            if min(height, width) < window:
                continue
            ones = torch.ones(1, 1, height, width)
            patches, _, _ = patch_divide(ones, window-2, window)
            legacy = patch_reverse(patches, ones, window-2, window)
            exact = patch_reverse(patches, ones, window-2, window, normalize_overlap=True)
            assert torch.equal(exact, ones)
            overlap.append({'hw': [height, width], 'window': window,
                            'legacy_identity_error': float((legacy-ones).abs().max())})
    report['legacy_overlap_audit'] = overlap
    report['candidate_overlap_profile'] = 'exact_coverage_v1; legacy defects retained in audit only'
    wrapper_errors = []
    for index in range(4):
        wrapper = copy.deepcopy(candidate.blocks[index][1]).cpu().double()
        # Diagnostic zero attention/FFN: the residual reconstruction itself
        # must be identity, including candidate padding and crop order.
        wrapper.layer[0].fn.correlate = lambda tokens: torch.zeros_like(tokens)
        with torch.no_grad():
            for parameter in wrapper.layer[1].fn.parameters():
                parameter.zero_()
        x = torch.randn(1, 48, 33, 47, dtype=torch.float64, requires_grad=True)
        result = wrapper(x, wrapper.window)
        probe = torch.randn_like(result)
        gradient = torch.autograd.grad((result*probe).sum(), x)[0]
        error = float((result-x).abs().max().detach())
        grad_error = float((gradient-probe).abs().max())
        assert error < 1e-12 and grad_error < 1e-12
        wrapper_errors.append({'window': wrapper.window, 'identity_max': error,
                               'input_gradient_identity_max': grad_error})
    report['corrected_wrapper_identity'] = wrapper_errors

    # Full Net has legacy ESA minimum sizes; tiny tests concern LRSA only.
    with torch.no_grad():
        lrsa = candidate.blocks[3][1].eval()
        for shape in [(2, 48, 3, 5), (1, 48, 33, 65), (1, 48, 65, 31)]:
            x = torch.randn(*shape, device=device)
            y = lrsa(x, 64)
            assert y.shape == x.shape
            finite(y)
        candidate.eval()
        for shape in [(1, 3, 32, 32), (2, 3, 33, 47), (1, 3, 48, 65)]:
            x = torch.rand(*shape, device=device)*255
            y = candidate(x)
            assert y.shape == (shape[0], 3, shape[2]*4, shape[3]*4)
            finite(y)
    report['shapes'] = 'PASS: LRSA tiny/asymmetric and Net odd/B2'

    # Real LR64/HR256 patch, backward only. This is not an optimization step.
    lr_path = args.data / 'DIV2K_train_LR_bicubic/X4/0001x4.png'
    hr_path = args.data / 'DIV2K_train_HR/0001.png'
    assert lr_path.is_file() and hr_path.is_file(), 'missing real batch'
    lr = torch.from_numpy(imageio.imread(lr_path).copy()).permute(2, 0, 1).float()
    hr = torch.from_numpy(imageio.imread(hr_path).copy()).permute(2, 0, 1).float()
    lr, hr = lr[None, :, :64, :64].to(device), hr[None, :, :256, :256].to(device)
    candidate.train()
    stage_rms = {}
    hooks = []
    for index, block in enumerate(candidate.blocks):
        def capture(module, inp, out, key=str(index)):
            stage_rms[key] = {'input_rms': float(inp[0].detach().square().mean().sqrt()),
                              'output_rms': float(out.detach().square().mean().sqrt())}
        hooks.append(block[1].register_forward_hook(capture))
    output = candidate(lr)
    finite(output)
    F.l1_loss(output, hr).backward()
    new = [(n, p) for n, p in candidate.named_parameters() if n not in common]
    assert len(new) == 208
    for name, param in candidate.named_parameters():
        if param.grad is not None:
            finite(param.grad)
    missing = [name for name, p in new if p.grad is None or not bool(p.grad.abs().sum() > 0)]
    assert not missing, ('inactive new parameter tensors', missing)
    for hook in hooks:
        hook.remove()
    report['real_backward'] = {'new_parameter_tensors_nonzero': len(new),
                               'lrsa_rms': stage_rms, 'image': 'DIV2K/0001 top-left LR64'}
    candidate.zero_grad(set_to_none=True)
    candidate.eval()
    with torch.no_grad():
        original = candidate(lr)
        buffer = io.BytesIO()
        torch.save(candidate.state_dict(), buffer)
        buffer.seek(0)
        reloaded = Net(scale=4).to(device).eval()
        reloaded.load_state_dict(torch.load(buffer, map_location=device, weights_only=True), strict=True)
        error = float((original-reloaded(lr)).abs().max())
        assert error == 0, error
    report['strict_reload_max_error'] = error
    report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
    report['status'] = 'GATE0_AND_BACKWARD_PRECHECK_PASS_WITH_EXACT_COVERAGE'
    report['unverified'] = ['optimizer smoke', 'AMP', 'target-server latency', 'PSNR benefit']
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
