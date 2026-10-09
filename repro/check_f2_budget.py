"""Independent F2 engineering checks; synthetic tensors are not PSNR evidence."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
import torch
from model import Model
from model.lfmn import Net as OriginalNet
from model.lfmnf2 import Net as F2Net

TAB_WIDTHS = [96] * 8
LRSA_WIDTHS = [64, 80, 112, 128, 64, 80, 112, 128]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str(tensor.dtype).encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def ffns(net):
    for stage, (tab, lrsa) in enumerate(net.blocks):
        yield f'blocks.{stage}.0.mlp.fn', tab.mlp.fn
        yield f'blocks.{stage}.1.layer.1.fn', lrsa.layer[1].fn


def check(device):
    report = {'device': str(device), 'torch_version': torch.__version__,
              'cuda_version': torch.version.cuda, 'self_ensemble': False,
              'scope': 'synthetic engineering checks; no PSNR or deployment claim'}
    if device.type == 'cuda':
        report['gpu'] = torch.cuda.get_device_name(device)
    torch.manual_seed(1)
    original = OriginalNet(scale=4)
    original_rng = torch.get_rng_state().clone()
    original_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    torch.manual_seed(1)
    equal = F2Net(scale=4, tab_widths=[96]*8, lrsa_widths=[96]*8, init_seed=1)
    require(digest(original.state_dict()) == digest(equal.state_dict()),
            'Uniform 96-channel F2 does not exactly restore baseline initial state')
    require(torch.equal(original_rng, torch.get_rng_state()), 'Uniform F2 construction changed RNG')
    torch.manual_seed(1)
    candidate = F2Net(scale=4, init_seed=1)
    require(torch.equal(original_rng, torch.get_rng_state()), 'F2 construction changes baseline CPU RNG stream')
    if original_cuda_rng:
        require(all(torch.equal(a, b) for a, b in zip(original_cuda_rng, torch.cuda.get_rng_state_all())),
                'F2 construction changes CUDA RNG stream')
    changed_prefixes = tuple(f'blocks.{i}.1.layer.1.fn.' for i in range(8))
    shared_original = {k:v for k,v in original.state_dict().items() if not k.startswith(changed_prefixes)}
    shared_f2 = {k:v for k,v in candidate.state_dict().items() if not k.startswith(changed_prefixes)}
    require(digest(shared_original) == digest(shared_f2), 'Untouched model initial state differs')
    report['shared_state_sha256'] = digest(shared_f2)
    report['uniform_initial_state_sha256'] = digest(equal.state_dict())
    report['construction_rng_identical'] = True
    report['construction_cuda_rng_identical'] = True if original_cuda_rng else None
    require(sum(p.numel() for p in original.parameters()) == 759627, 'Unexpected baseline parameter count')
    require(sum(p.numel() for p in candidate.parameters()) == 759627, 'F2 parameter budget differs')
    report['params_original'] = report['params_f2'] = 759627
    report['ffn_layers'] = []
    actual_widths = []
    for name, layer in ffns(candidate):
        width = layer.fc1.out_features
        actual_widths.append(width)
        require(layer.fc1.in_features == 48 and layer.fc2.out_features == 48, 'FFN outer width changed')
        require(layer.fc2.in_features == width, 'FFN internal width mismatch')
        conv = layer.dwconv.depthwise_conv[0]
        require(conv.in_channels == width and conv.out_channels == width and conv.groups == width,
                'FFN depthwise configuration mismatch')
        require(conv.kernel_size == (5,5), 'FFN kernel changed')
        params = sum(p.numel() for p in layer.parameters())
        require(params == 123 * width + 48, 'FFN parameter formula mismatch')
        report['ffn_layers'].append({'name': name, 'hidden': width, 'params': params,
                                     'mac_per_lr_pixel': 121 * width})
    expected_widths = [v for pair in zip(TAB_WIDTHS, LRSA_WIDTHS) for v in pair]
    require(actual_widths == expected_widths, 'Unexpected F2 stage allocation')
    require(sum(actual_widths) == 1536, 'Hidden width budget differs')
    report['ffn_mac_per_lr_pixel_original'] = report['ffn_mac_per_lr_pixel_f2'] = 121 * 1536
    report['ffn_budget_scope'] = 'Linear and DWConv multiplications; excludes activation/LN/bias; remaining paths unchanged'
    original, equal, candidate = original.to(device), equal.to(device), candidate.to(device)
    original.eval(); equal.eval(); candidate.eval()
    report['shape_checks'] = []
    with torch.no_grad():
        for height, width in [(31,35), (32,32)]:
            x = torch.randn(1,3,height,width,device=device)
            y0, y1, yf = original(x), equal(x), candidate(x)
            require(torch.equal(y0,y1), f'Uniform F2 output differs at {height}x{width}')
            require(yf.shape == (1,3,4*height,4*width), 'F2 output shape mismatch')
            require(torch.isfinite(yf).all().item(), 'Nonfinite output')
            report['shape_checks'].append({'lr': [height,width], 'sr': list(yf.shape),
                                            'uniform_equals_original': True})
    del original, equal
    # Execute the actual common wrapper evaluation branch and trap any x8 call.
    wrapper = Model.__new__(Model)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = candidate
    wrapper.self_ensemble = False
    wrapper.chop = False
    wrapper.n_GPUs = 1
    wrapper.idx_scale = 0
    def forbidden_x8(*args, **kwargs):
        raise AssertionError('Self-ensemble unexpectedly invoked')
    wrapper.forward_x8 = forbidden_x8
    wrapper.eval()
    calls = []
    direct_forward = candidate.forward
    def counted_forward(x):
        calls.append(1)
        return direct_forward(x)
    candidate.forward = counted_forward
    with torch.no_grad():
        wrapper(torch.randn(1,3,32,32,device=device), 0)
    candidate.forward = direct_forward
    require(len(calls) == 1, 'OFF evaluation did not execute exactly one model forward')
    report['actual_wrapper_off_single_forward'] = True
    candidate.train()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.0002, betas=(0.9,0.999), weight_decay=0)
    registered = [id(p) for group in optimizer.param_groups for p in group['params']]
    require(len(registered) == len(set(registered)), 'Duplicate optimizer parameter')
    require(set(registered) == {id(p) for p in candidate.parameters() if p.requires_grad}, 'Optimizer misses parameters')
    before = {name:layer.fc1.weight.detach().clone() for name,layer in ffns(candidate)}
    output = candidate(torch.randn(1,3,32,32,device=device))
    loss = torch.nn.functional.l1_loss(output, torch.randn_like(output))
    require(torch.isfinite(loss).item(), 'Nonfinite L1 loss')
    loss.backward()
    report['ffn_gradient_checks'] = []
    for name, layer in ffns(candidate):
        norm = 0.0
        for param in layer.parameters():
            require(param.grad is not None, f'Missing FFN gradient: {name}')
            require(torch.isfinite(param.grad).all().item(), f'Nonfinite FFN gradient: {name}')
            norm += param.grad.detach().float().square().sum().item()
        require(norm > 0, f'Inactive FFN: {name}')
        report['ffn_gradient_checks'].append({'name':name,'gradient_l2':norm**0.5})
    optimizer.step()
    for name, layer in ffns(candidate):
        require(not torch.equal(before[name], layer.fc1.weight.detach()), f'Adam did not update {name}')
    require(all(block[0].initted.item() for block in candidate.blocks), 'TAB prototype state was not initialized')
    report['synthetic_l1_loss'] = float(loss.detach())
    report['adam_updates_all_ffns'] = True
    report['optimizer_parameter_coverage'] = 'complete, no duplicates'
    candidate.eval()
    memory = io.BytesIO()
    torch.save(candidate.state_dict(), memory)
    memory.seek(0)
    restored = F2Net(scale=4, init_seed=1).to(device)
    restored.load_state_dict(torch.load(memory, map_location=device, weights_only=True), strict=True)
    require(digest(candidate.state_dict()) == digest(restored.state_dict()), 'Strict reload state differs')
    restored.eval()
    with torch.no_grad():
        x = torch.randn(1,3,31,35,device=device)
        require(torch.equal(candidate(x), restored(x)), 'Strict reload output differs')
    report['checkpoint_strict_roundtrip'] = True
    report['passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = check(device)
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
