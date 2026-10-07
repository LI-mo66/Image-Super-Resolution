"""Validate frozen PCSTR and a real DIV2K batch; never a performance run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
import ast
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmnpcstr import Net
from run_n9_m1_server import audit_baseline, decide, dataset_args
from model.lfmn import Net as Baseline
from data.div2k import DIV2K


def check_legacy_baseline(baseline):
    # Commit is inferred, not a missing historical manifest magically recovered.
    # Test the actual old implementation as well as comparing pipeline files.
    source = subprocess.check_output(['git', 'show', 'bf68bfe:LFMN/model/lfmn.py'],
                                     cwd=ROOT, text=True, encoding='utf-8')
    module = types.ModuleType('n9_legacy_baseline')
    exec(compile(source, '<bf68bfe:lfmn.py>', 'exec'), module.__dict__)
    # Atomic CUDA scatter can perturb hard clustering between identical runs.
    # Compare source equivalence on CPU with deterministic accumulation.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    reference, current = module.Net(scale=4), Baseline(scale=4)
    state = torch.load(baseline / 'model/model_20.pt', map_location='cpu', weights_only=True)
    reference.load_state_dict(state, strict=True)
    current.load_state_dict(state, strict=True)
    x = torch.rand(1, 3, 32, 32) * 255
    reference.eval()
    current.eval()
    with torch.inference_mode():
        assert torch.equal(reference(x), current(x)), 'legacy B0 eval changed'
    reference.train()
    current.train()
    old_y, new_y = reference(x), current(x)
    assert torch.allclose(old_y, new_y, atol=1e-4, rtol=1e-5), 'legacy B0 training changed'
    old_y.abs().mean().backward()
    new_y.abs().mean().backward()
    for (name, old), (new_name, new) in zip(reference.named_parameters(), current.named_parameters()):
        assert name == new_name
        assert old.grad is not None and new.grad is not None
        assert torch.allclose(old.grad, new.grad, atol=1e-5, rtol=1e-4), name
    for path in ['LFMN/data/__init__.py', 'LFMN/data/common.py', 'LFMN/data/div2k.py',
                 'LFMN/data/srdata.py', 'LFMN/model/__init__.py', 'LFMN/main.py',
                 'LFMN/loss/__init__.py']:
        old = subprocess.check_output(['git', 'show', f'bf68bfe:{path}'], cwd=ROOT)
        assert old.replace(b'\r\n', b'\n') == (ROOT / path).read_bytes().replace(b'\r\n', b'\n'), path
    import utility
    utility_source = subprocess.check_output(['git', 'show', 'bf68bfe:LFMN/utility.py'],
                                             cwd=ROOT, text=True, encoding='utf-8')
    old_utility = types.ModuleType('n9_legacy_utility')
    exec(compile(utility_source, '<bf68bfe:utility.py>', 'exec'), old_utility.__dict__)
    old_tree = ast.parse(utility_source)
    new_tree = ast.parse((ROOT / 'LFMN/utility.py').read_text(encoding='utf-8'))
    for name in ('quantize', 'calc_psnr', 'calc_ssim'):
        old_node = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new_node = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(old_node) == ast.dump(new_node), name
    settings = dict(optimizer='ADAM', lr=2e-4, weight_decay=0,
                    betas=(.9, .999), epsilon=1e-8, momentum=.9,
                    scheduler='cosine', eta_min=1e-6, decay='200-400-600-800', gamma=.5)
    old_optimizer = old_utility.make_optimizer(SimpleNamespace(**settings, epochs=150), reference)
    new_optimizer = utility.make_optimizer(SimpleNamespace(**settings, epochs=20, scheduler_t_max=150), current)
    for _ in range(20):
        assert old_optimizer.get_lr() == new_optimizer.get_lr(), 'cosine trajectory changed'
        old_optimizer.step()
        new_optimizer.step()
        old_optimizer.schedule()
        new_optimizer.schedule()
    print('Legacy bf68bfe B0: eval identical, training/gradients match; data/model loader/loss sources identical')
    print('PSNR/SSIM/quantize source identical; first20 Adam/cosine learning rates identical')
    torch.set_num_threads(previous_threads)


def check_candidate_precision():
    source = subprocess.check_output(['git', 'show', 'b198de7:LFMN/model/lfmnpcstr.py'],
                                     cwd=ROOT, text=True, encoding='utf-8')
    module = types.ModuleType('n9_frozen_pcstr')
    exec(compile(source, '<b198de7:lfmnpcstr.py>', 'exec'), module.__dict__)
    old = module.PriorConditionedSoftTokenRouter().cuda()
    from model.lfmnpcstr import PriorConditionedSoftTokenRouter
    new = PriorConditionedSoftTokenRouter().cuda()
    new.load_state_dict(old.state_dict(), strict=True)
    feature = torch.randn(2, 48, 17, 23, device='cuda')
    prior = torch.randn(2, 32, 17, 23, device='cuda')
    old_output, new_output = old(feature, prior), new(feature, prior)
    assert torch.equal(old_output, new_output), 'FP32 fix changed frozen candidate output'
    old_output.square().mean().backward()
    new_output.square().mean().backward()
    for (name, before), (after_name, after) in zip(old.named_parameters(), new.named_parameters()):
        assert name == after_name and torch.equal(before.grad, after.grad), name
    new.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        output = new(feature, prior)
        loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.parameters())
    print('FP32 output/gradients unchanged by autocast fix; AMP backward finite')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.baseline:
        audit_baseline(args.baseline.resolve())
    subprocess.run([sys.executable, str(ROOT / 'repro/check_pcstr.py')], check=True)
    assert decide(-.01, -.001, -.01, 0, 0) == 'STOP'
    assert decide(.02, 0, .01, 0, 5).startswith('EXTEND_ELIGIBLE')
    assert decide(.005, .002, .001, 0, 4).startswith('GRAY_ELIGIBLE')
    assert decide(.02, 0, .01, -.001, 5) == 'STOP'
    if not torch.cuda.is_available():
        raise RuntimeError('real batch verification requires CUDA')
    check_candidate_precision()
    if args.baseline:
        check_legacy_baseline(args.baseline.resolve())
    net = Net(scale=4).cuda().eval()
    train_dataset = DIV2K(dataset_args(args.data_root.resolve()), name='DIV2K', train=True)
    validation_dataset = DIV2K(dataset_args(args.data_root.resolve()), name='DIV2K', train=False)
    assert len(train_dataset) == 4000 and len(validation_dataset) == 100
    lr, hr, _ = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=4)))
    net.train()
    prediction = net(lr.cuda())
    loss = torch.nn.functional.l1_loss(prediction, hr.cuda())
    assert torch.isfinite(prediction).all() and torch.isfinite(loss)
    loss.backward()
    gradient_min = float('inf')
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        magnitude = float(parameter.grad.abs().sum())
        assert magnitude > 0, f'zero real-batch gradient: {name}'
        gradient_min = min(gradient_min, magnitude)
    net.zero_grad(set_to_none=True)
    net.eval()
    print('All full-model parameters have finite nonzero real-batch gradients; min=', gradient_min)
    with torch.inference_mode():
        for height, width in [(17, 23), (64, 64)]:
            x = torch.rand(1, 3, height, width, device='cuda') * 255
            y = net(x)
            assert y.shape == (1, 3, height * 4, width * 4)
            assert torch.isfinite(y).all()
    del net
    torch.cuda.empty_cache()
    command = [sys.executable, 'main.py', '--dir_data', str(args.data_root.resolve()),
               '--model', 'LFMNPCSTR', '--data_train', 'DIV2K', '--data_test', 'DIV2K',
               '--data_range', '1-800/801-801', '--scale', '4', '--patch_size', '256',
               '--batch_size', '4', '--n_threads', '8', '--ext', 'img', '--epochs', '1',
               '--max_train_batches', '1', '--test_every', '1000', '--lr', '2e-4',
               '--scheduler', 'cosine', '--scheduler_t_max', '150', '--eta_min', '1e-6',
               '--loss', '1*L1', '--seed', '1', '--save_per_image_metrics', '--save', str(output)]
    subprocess.run(command, cwd=ROOT / 'LFMN', check=True)
    state = torch.load(output / 'model/model_1.pt', map_location='cpu', weights_only=True)
    first, restored = Net(scale=4).cuda().eval(), Net(scale=4).cuda().eval()
    first.load_state_dict(state, strict=True)
    restored.load_state_dict(state, strict=True)
    x = torch.rand(1, 3, 17, 23, device='cuda') * 255
    with torch.inference_mode():
        assert torch.equal(first(x), restored(x))
    rows = torch.load(output / 'per_image_metrics/epoch_0001.pt', map_location='cpu', weights_only=True)
    assert len(rows) == 1 and rows[0]['filename'] == '0801'
    assert all(torch.isfinite(torch.tensor(rows[0][key])) for key in ('psnr', 'ssim'))
    for block in restored.blocks:
        block[0].collect_routing_stats = True
    with torch.inference_mode():
        restored(x)
    stats = [{key: float(value) for key, value in block[0].last_routing_stats.items()}
             for block in restored.blocks]
    assert all(s['effective_tokens_mean'] > 8 and s['mass_ratio'] < 100 for s in stats)
    report = {'status': 'PASS', 'parameters': sum(p.numel() for p in restored.parameters()),
              'real_batch': 'B4 LR64 HR256, Adam step; full 0801 validation',
              'reload': 'strict, identical outputs', 'routing': stats,
              'real_batch_gradient_min': gradient_min,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'performance_claim': False}
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
