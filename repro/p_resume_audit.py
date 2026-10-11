"""Separate serialization correctness from repeatability of a training backend."""
import copy
import random

import torch


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    return copy.deepcopy(value)


def assert_exact_tree(actual, expected, path='state'):
    if torch.is_tensor(expected):
        if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual.cpu(), expected.cpu()):
            raise AssertionError(f'Serialized tensor mismatch: {path}')
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f'Serialized keys mismatch: {path}')
        for key in expected:
            assert_exact_tree(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f'Serialized sequence mismatch: {path}')
        for i, item in enumerate(expected):
            assert_exact_tree(actual[i], item, f'{path}[{i}]')
    elif actual != expected:
        raise AssertionError(f'Serialized value mismatch: {path}')


def compare_states(actual, reference, parameter_keys, rtol=1e-4, atol=1e-5):
    if set(actual) != set(reference):
        raise AssertionError('State key sets differ')
    rows = []
    for key, expected in reference.items():
        value = actual[key]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise AssertionError(f'State shape/dtype mismatch: {key}')
        if value.is_floating_point():
            if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
                raise FloatingPointError(f'Nonfinite continuation state: {key}')
            delta = (value - expected).abs()
            mismatch = delta > atol + rtol * expected.abs()
            if mismatch.any():
                rows.append({'key': key, 'kind': 'parameter' if key in parameter_keys else 'buffer',
                             'shape': list(value.shape), 'mismatched_elements': int(mismatch.sum()),
                             'elements': value.numel(), 'max_absolute_difference': float(delta.max())})
        elif not torch.equal(value, expected):
            raise AssertionError(f'Nonfloating continuation state differs: {key}')
    return {'within_tolerance': not rows, 'rtol': rtol, 'atol': atol,
            'mismatches': rows, 'meaning': 'Backend continuation repeatability, not checkpoint accuracy'}


def deterministic_cpu_replay(factory, source_model, source_optimizer, loaded,
                             lr, hr, loss_fn):
    """Exact CPU next-step comparison, no change to production GPU training."""
    assert_exact_tree(source_model, loaded['model'], 'model_before_replay')
    assert_exact_tree(source_optimizer, loaded['optimizer'], 'optimizer_before_replay')
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    original_python = random.getstate()
    outcomes = []
    try:
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            for model_state, opt_state in ((source_model, source_optimizer),
                                           (loaded['model'], loaded['optimizer'])):
                print(f'CPU deterministic resume replay {len(outcomes) + 1}/2 starting', flush=True)
                net = factory().cpu()
                net.load_state_dict(copy.deepcopy(model_state), strict=True)
                opt = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(.9, .999))
                opt.load_state_dict(copy.deepcopy(opt_state))
                torch.set_rng_state(loaded['torch_rng'])
                random.setstate(loaded['python_rng'])
                net.train()
                opt.zero_grad(set_to_none=True)
                prediction = net(lr.cpu())
                loss = loss_fn(prediction, hr.cpu())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite deterministic CPU replay loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise FloatingPointError('Nonfinite deterministic CPU replay gradient')
                opt.step()
                outcomes.append({'model': cpu_tree(net.state_dict()),
                                 'optimizer': cpu_tree(opt.state_dict()), 'loss': float(loss.detach())})
                print(f'CPU deterministic replay {len(outcomes)}/2 loss={outcomes[-1]["loss"]:.10f}', flush=True)
                del prediction, loss, opt, net
        assert_exact_tree(outcomes[0], outcomes[1], 'deterministic_cpu_next_update')
        return {'passed': True, 'backend': 'CPU deterministic algorithms',
                'model_optimizer_loss_equal_exactly': True,
                'scope': 'Same real batch and saved state; verifies portable serialization/update semantics, not GPU bitwise repeatability'}
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        random.setstate(original_python)
