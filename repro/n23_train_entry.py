"""Private B0/N23 framework entry; audits without changing public defaults."""
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from option import args

if args.model not in ('lfmn_exact_overlap', 'lfmn_n23'):
    raise ValueError('This entry accepts only registered B0/N23 models')
if (args.load or args.pre_train or args.reset or args.test_only
        or args.resume not in (-1, 0)):
    raise ValueError('Scratch-only entry: no resume, reset, preload or test-only')
if (args.precision != 'single' or args.n_GPUs != 1
        or args.rgcrd_mode != 'off' or args.scale != [4]
        or args.rgb_range != 255 or args.chop or args.self_ensemble):
    raise ValueError('Registered entry requires FP32, one GPU, x4, no teacher/chop/x8')
SAVE = (Path(args.experiment_root).resolve() / args.save).resolve()
if SAVE.exists():
    raise FileExistsError(f'Refusing to overwrite existing experiment: {SAVE}')
REFERENCE = None
if os.environ.get('N23_BASELINE_MANIFEST'):
    REFERENCE = json.loads(Path(os.environ['N23_BASELINE_MANIFEST']).read_text())['baseline_audit']
elif os.environ.get('N23_PAIR_MANIFEST'):
    REFERENCE = json.loads(Path(os.environ['N23_PAIR_MANIFEST']).read_text())['pair_reference']

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.set_num_threads(4)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

import model
import data
import utility
from model.lfmn import Net as Baseline
from trainer import Trainer


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def legacy_state_hash(state):
    """Exact historical N21/N22 proof format; do not mix hash schemas."""
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def append(name, record):
    with (SAVE / name).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


class FingerprintedLoader:
    def __init__(self, loader):
        self.loader = loader
        self.epochs = 0

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        self.epochs += 1
        for index, batch in enumerate(self.loader):
            if index == 0:
                record = {
                    'epoch': self.epochs,
                    'lr_sha256': tensor_hash(batch[0]),
                    'hr_sha256': tensor_hash(batch[1]),
                    'lr_shape': list(batch[0].shape),
                    'hr_shape': list(batch[1].shape),
                }
                if REFERENCE:
                    expected_batch = REFERENCE['batch_fingerprints'][self.epochs - 1]
                    assert record['epoch'] == expected_batch['epoch']
                    assert record['lr_sha256'] == expected_batch['lr_sha256'], 'B0/N23 data stream mismatch before optimizer update'
                    if 'hr_sha256' in expected_batch:
                        assert record['hr_sha256'] == expected_batch['hr_sha256'], 'B0/N23 HR stream mismatch before optimizer update'
                append('batch_fingerprints.jsonl', record)
            yield batch


original_data = data.Data.__init__


def checked_data(self, parsed):
    original_data(self, parsed)
    for subset, text in (
        (self.loader_train.dataset.datasets[0], parsed.data_range.split('/')[0]),
        (self.loader_test[0].dataset, parsed.data_range.split('/')[1]),
    ):
        first, last = map(int, text.split('-'))
        assert [Path(p).stem for p in subset.images_hr] == [
            f'{i:04d}' for i in range(first, last + 1)
        ], 'Dataset range/scan mismatch'
    self.loader_train = FingerprintedLoader(self.loader_train)


data.Data.__init__ = checked_data
original_build = model.Model._build_model


def checked_build(self, module, parsed):
    with torch.random.fork_rng(devices=[]):
        baseline = Baseline(scale=4)
        expected_rng = torch.get_rng_state()
    net = original_build(self, module, parsed)
    assert torch.equal(expected_rng, torch.get_rng_state()), 'Initialization changed data RNG'
    expected = baseline.state_dict()
    if REFERENCE:
        assert legacy_state_hash(expected) == REFERENCE['initial_state_sha256'], 'B0/N23 initial state mismatch before optimizer update'
    actual = net.state_dict()
    retained = {k: v for k, v in expected.items() if '.1.layer.0.fn.' not in k}
    assert len(retained) == 403, len(retained)
    for key, value in retained.items():
        assert key in actual and torch.equal(value, actual[key]), key
    if parsed.model == 'lfmn_exact_overlap':
        assert set(expected) == set(actual)
        for key, value in expected.items():
            assert torch.equal(value, actual[key]), key
    parameters = sum(p.numel() for p in net.parameters())
    assert parameters <= 759627
    new_names = [name for name, _ in net.named_parameters()
                 if '.1.layer.0.fn.' in name] if parsed.model == 'lfmn_n23' else []
    assert not new_names or len(new_names) == 208, len(new_names)
    net.n23_new_parameter_names = new_names
    net.n23_capture_eval = True
    (SAVE / 'initial_state_proof.json').write_text(json.dumps({
        'model': parsed.model, 'seed': parsed.seed, 'tf32': False,
        'parameters': parameters,
        'common_initial_state_sha256': state_hash(expected),
        'legacy_common_initial_state_sha256': legacy_state_hash(expected),
        'retained_state_sha256': state_hash(retained),
        'retained_state_count': len(retained),
        'new_parameter_count': len(new_names),
        'cpu_rng_sha256': tensor_hash(expected_rng),
    }, indent=2), encoding='utf-8')

    def finite_forward(module_, inputs, output):
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite model output')
        if not module_.training and module_.n23_capture_eval:
            append('eval_output_fingerprints.jsonl', {
                'epoch': module_.n23_epoch,
                'input_sha256': tensor_hash(inputs[0]),
                'output_sha256': tensor_hash(output),
                'input_shape': list(inputs[0].shape),
                'output_shape': list(output.shape),
            })
            module_.n23_capture_eval = False

    # Model.forward's evaluation path calls net.forward directly, bypassing
    # nn.Module hooks. Wrap that private instance's callable, not public code.
    original_forward = net.forward
    def checked_forward(*inputs, **kwargs):
        output = original_forward(*inputs, **kwargs)
        finite_forward(net, inputs, output)
        return output
    net.forward = checked_forward
    return net


model.Model._build_model = checked_build
original_optimizer = utility.make_optimizer


def checked_optimizer(parsed, target):
    optimizer = original_optimizer(parsed, target)
    original_step = optimizer.step
    optimizer.checked_steps = 0
    optimizer.n23_new_nonzero_gradients = 0

    def step(*a, **kw):
        pairs = [(name, p.grad.detach()) for name, p in target.named_parameters()
                 if p.grad is not None]
        gradients = [grad for _, grad in pairs]
        norms = torch.stack(torch._foreach_norm(gradients)) if gradients else None
        if norms is None or not bool(torch.isfinite(norms).all()):
            raise FloatingPointError('Missing/nonfinite gradients')
        net = target.model
        gradient_indices = {name: index for index, (name, _) in enumerate(pairs)}
        new_indices = []
        for name in net.n23_new_parameter_names:
            full_name = 'model.' + name
            if full_name not in gradient_indices:
                raise FloatingPointError(f'Missing SCC gradient: {name}')
            new_indices.append(gradient_indices[full_name])
        active = int((norms[new_indices] > 0).sum()) if new_indices else 0
        optimizer.n23_new_nonzero_gradients = active
        result = original_step(*a, **kw)
        optimizer.checked_steps += 1
        return result

    step._wrapped_by_lr_sched = getattr(original_step, '_wrapped_by_lr_sched', False)
    optimizer.step = step
    return optimizer


utility.make_optimizer = checked_optimizer
original_train = Trainer.train


def logged_train(self):
    if os.environ.get('N23_PROTOCOL_PROBE_ONLY') == '1':
        # Formal loader/initialization audit before any optimizer update.
        next(iter(self.loader_train))
        assert self.optimizer.checked_steps == 0
        (SAVE/'protocol_probe.json').write_text(json.dumps(
            {'status': 'INITIALIZATION_AND_FIRST_BATCH_PASS', 'optimizer_steps': 0}), encoding='utf-8')
        raise SystemExit(0)
    net = self.model.model
    net.n23_epoch = self.optimizer.get_last_epoch() + 1
    net.n23_capture_eval = True
    before = self.optimizer.checked_steps
    lr = self.optimizer.get_lr()
    parameters = dict(net.named_parameters())
    previous = {name: parameters[name].detach().cpu().clone()
                for name in net.n23_new_parameter_names}
    original_train(self)
    updates = self.optimizer.checked_steps - before
    assert updates == (self.args.max_train_batches or len(self.loader_train))
    if not torch.isfinite(self.loss.log[-1]).all():
        raise FloatingPointError('Nonfinite loss')
    if not all(bool(torch.isfinite(p).all()) for p in net.parameters()):
        raise FloatingPointError('Nonfinite parameters')
    changed = sum(not torch.equal(previous[name], parameters[name].detach().cpu())
                  for name in previous)
    append('mechanism.jsonl', {
        'epoch': self.optimizer.get_last_epoch(), 'updates': updates,
        'total_updates': self.optimizer.checked_steps, 'learning_rate': lr,
        'learning_rate_after_schedule': self.optimizer.get_lr(),
        'loss': [float(v) for v in self.loss.log[-1]],
        'scc_nonzero_gradients_last_step': self.optimizer.n23_new_nonzero_gradients,
        'scc_parameters_changed': changed,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if not args.cpu else 0,
    })


Trainer.train = logged_train
if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'LFMN/main.py'), run_name='__main__')
