"""Mandatory server engineering gates before P paired adaptation."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import torch

from train_p_adaptation import (ROOT, atomic_json, batch_from_plan, configuration,
                                criterion, evaluate, make_net, save_checkpoint,
                                sha, training_plan)
from p_adapt_logging import RunLogger
from check_p_adapt_logging import run_checks as logging_checks
from model.lfmn import Net as Baseline
from p_resume_audit import cpu_tree, assert_exact_tree, compare_states, deterministic_cpu_replay


def run(args):
    if os.name == 'nt':
        raise RuntimeError('Actual LFMN/real-data gates run on server only')
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'checking', 'scheme': args.scheme, 'checkpoint_sha256': sha(args.checkpoint),
              'gpu': torch.cuda.get_device_name(torch.device(args.device)), 'torch': torch.__version__,
              'model_sha256': sha(ROOT / 'LFMN/model/lfmn.py'),
              'runner_sha256': sha(ROOT / 'repro/train_p_adaptation.py'),
              'candidate_source_sha256': sha(ROOT / 'LFMN/model/lfmnp4.py') if args.scheme == 'P4' else
                                         (sha(ROOT / 'repro/p3_objective.py') if args.scheme == 'P3' else None),
              'data_root': str(Path(args.data_root).resolve()), 'checks': {}, 'performance_evidence': False}
    path = out / 'check_report.json'
    atomic_json(path, report)
    try:
        # Isolated logger smoke runs before any SR optimization.
        report['checks']['logging'] = logging_checks(out / 'logging')
        if not report['checks']['logging']['passed']:
            raise AssertionError('Log infrastructure gate failed')
        torch.set_num_threads(4)
        common = SimpleNamespace(**vars(args), steps=2000, steps_per_epoch=100,
                                 batch_size=4, patch_lr=48, seed=1, lr=1e-5,
                                 resume=None, checks_report=str(path), benchmark_final=False)
        plan, plan_hash = training_plan(args.data_root, 2000, 4, 48, 1)
        config = configuration(common, plan_hash)
        config.update(epochs=1, experiment_name=args.scheme + '_engineering',
                      scheme=args.scheme, resume_start_epoch=None, resume_start_step=None)
        with RunLogger(out, config['experiment_name'], 4, 1, config) as log:
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            net = make_net(args.scheme, args.checkpoint, args.device).eval()
            params = sum(p.numel() for p in net.parameters())
            baseline = make_net('P0', args.checkpoint, args.device).eval()
            base_count = sum(p.numel() for p in baseline.parameters())
            report['checks']['parameters'] = {'baseline': base_count, 'candidate': params,
                                             'added': params - base_count}
            keys = set(baseline.state_dict())
            assert keys <= set(net.state_dict())
            for key in keys:
                assert torch.equal(baseline.state_dict()[key], net.state_dict()[key])
            state_before = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            with torch.inference_mode():
                for h, w in ((32, 32), (33, 35), (35, 33)):
                    x = torch.linspace(0, 255, 3 * h * w, device=args.device).reshape(1, 3, h, w)
                    expected, actual = baseline(x), net(x)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-4)
                    assert actual.shape == (1, 3, h * 4, w * 4)
                    assert torch.isfinite(actual).all()
                    if args.scheme == 'P4':
                        net.position_bias.enabled = False
                        torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
                        net.position_bias.enabled = True
            assert all(torch.equal(net.state_dict()[k].cpu(), v) for k, v in state_before.items())
            report['checks']['shape_zero_bias_disabled_state'] = True
            if args.scheme == 'P4':
                from model.lfmnp4 import self_check
                report['checks']['position_math'] = self_check()
            del baseline, state_before
            torch.cuda.empty_cache()
            lr, hr, bh = batch_from_plan(args.data_root, plan[0], 48)
            with torch.inference_mode():
                initial_real = net(lr.to(args.device))
            assert initial_real.shape == hr.shape
            del initial_real
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, betas=(.9, .999))
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            random.seed(1)
            upstream = []
            for step in (1, 2):
                net.train()
                optimizer.zero_grad(set_to_none=True)
                result = net(lr.to(args.device))
                loss = criterion(args.scheme, result, hr.to(args.device))
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
                if args.scheme == 'P4':
                    b = net.position_bias
                    assert b.fc2.weight.grad is not None and b.fc2.weight.grad.abs().sum() > 0
                    upstream.append(float(b.fc1.weight.grad.abs().sum()))
                    if step == 2:
                        assert upstream[-1] > 0 and b.logscale.grad.abs().sum() > 0
                optimizer.step()
                print(f'Real batch engineering step={step} loss={float(loss):.10f}', flush=True)
            validation = evaluate(net, args.data_root, args.device, 1, 2, limit=1)
            log.record({'epoch': 1, 'global_step': 2, 'learning_rate': 1e-5,
                        'train_loss': float(loss), 'validation_psnr': validation[0]['psnr'],
                        'validation_ssim': validation[0]['ssim'], 'is_best': True, 'elapsed_seconds': 0})
            checkpoint = log.directory / 'checkpoints/last.pt'
            save_checkpoint(checkpoint, net, optimizer, 2, 1, config, {'psnr': validation[0]['psnr']})
            net.eval()
            with torch.inference_mode():
                reference = net(lr[:1].to(args.device)).cpu()
            loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
            source_model = cpu_tree(net.state_dict())
            source_optimizer = cpu_tree(optimizer.state_dict())
            assert_exact_tree(source_model, loaded['model'], 'serialized_model')
            assert_exact_tree(source_optimizer, loaded['optimizer'], 'serialized_optimizer')
            restored = make_net(args.scheme, args.checkpoint, args.device)
            restored.load_state_dict(loaded['model'], strict=True)
            restored.eval()
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-5, betas=(.9, .999))
            restored_optimizer.load_state_dict(copy.deepcopy(loaded['optimizer']))
            with torch.inference_mode():
                torch.testing.assert_close(restored(lr[:1].to(args.device)).cpu(), reference, rtol=0, atol=0)
            assert len(restored_optimizer.state) == len(optimizer.state)
            source_state = optimizer.state_dict()['state']
            recovered_state = restored_optimizer.state_dict()['state']
            assert source_state.keys() == recovered_state.keys()
            for parameter_id, fields in source_state.items():
                for key, value in fields.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(recovered_state[parameter_id][key], value, rtol=0, atol=0)
                    else:
                        assert recovered_state[parameter_id][key] == value
            def gpu_update(target, opt):
                torch.set_rng_state(loaded['torch_rng'])
                torch.cuda.set_rng_state_all(loaded['cuda_rng'])
                random.setstate(loaded['python_rng'])
                target.train()
                opt.zero_grad(set_to_none=True)
                criterion(args.scheme, target(lr.to(args.device)), hr.to(args.device)).backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in target.parameters()):
                    raise FloatingPointError('Nonfinite GPU continuation gradient')
                opt.step()
                return cpu_tree(target.state_dict())
            direct = gpu_update(net, optimizer)
            resumed = gpu_update(restored, restored_optimizer)
            parameter_keys = set(dict(net.named_parameters()))
            continuation = compare_states(resumed, direct, parameter_keys)
            report['checks']['gpu_continuation'] = continuation
            if not continuation['within_tolerance']:
                # Hard-route changes can propagate into parameter gradients too.
                # Serialization is already exact; require no-reload GPU controls
                # to reproduce each affected state class before CPU replay.
                if any(row['kind'] == 'buffer' and not row['key'].endswith('.means')
                       for row in continuation['mismatches']):
                    raise AssertionError(f'Unexplained continuation mismatch: {continuation}')
                repeats = []
                for repeat in range(2):
                    net.load_state_dict(source_model, strict=True)
                    optimizer.load_state_dict(copy.deepcopy(source_optimizer))
                    replayed = gpu_update(net, optimizer)
                    repeats.append(compare_states(replayed, direct, parameter_keys))
                    print(f'GPU same-instance control repeat {repeat + 1}: {repeats[-1]}', flush=True)
                report['checks']['gpu_same_instance_repeat_controls'] = repeats
                if not any(any(row['key'].endswith('.means') for row in control['mismatches']) for control in repeats):
                    raise AssertionError('GPU repeat controls do not explain prototype drift; keep gate blocked')
                parameter_drift = any(row['kind'] == 'parameter' for row in continuation['mismatches'])
                if parameter_drift and not any(any(row['kind'] == 'parameter' for row in control['mismatches']) for control in repeats):
                    raise AssertionError('GPU controls do not reproduce parameter-update variation; keep gate blocked')
                report['checks']['gpu_repeatability_scope'] = {
                    'same_instance_no_checkpoint_reload_control': True,
                    'parameter_variation_reproduced': parameter_drift,
                    'specific_cause': 'GPU training nonrepeatability observed; atomic clustering/hard routing is a plausible mechanism, not isolated proof'}
                print('GPU prototype repeatability limitation reproduced without checkpoint reconstruction. '
                      'Running exact CPU replay of the same real batch; production training is unchanged.', flush=True)
                report['checks']['cpu_resume_replay'] = deterministic_cpu_replay(
                    lambda: make_net(args.scheme, args.checkpoint, 'cpu'), source_model,
                    source_optimizer, loaded, lr, hr,
                    lambda prediction, target: criterion(args.scheme, prediction, target))
                continuation_scope = 'Exact serialization + exact deterministic CPU replay; GPU prototype continuation is not bitwise reproducible'
            else:
                continuation_scope = 'GPU next update within registered tolerance; exact serialization and eval output'
            report['checks']['real_batch_validation_checkpoint'] = {'passed': True, 'batch_hash': bh,
                         'finite_loss_gradient': True, 'strict_reload': True,
                         'optimizer_moments_restored_exactly': True,
                         'next_update_verification_scope': continuation_scope,
                         'new_bias_upstream_grad_steps': upstream}
            report['checks']['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(torch.device(args.device))
        report['status'] = 'verified'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        atomic_json(path, report)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scheme', choices=('P0', 'P3', 'P4'), required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--checkpoint', default=str(ROOT / 'LFMN/model/scale4_model_939.pt'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-root', required=True)
    run(parser.parse_args())
