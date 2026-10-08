"""Controller and scheduler contract tests; MOCKS ARE NOT SR RESULTS."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import run_n23_screen_server as server


class Contracts(unittest.TestCase):
    def setUp(self):
        parent = server.ROOT/'experiment/all_runs'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='n23_contract_', dir=parent)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_schedule_and_updates(self):
        lr = 1e-6 + (2e-4-1e-6)*(1+math.cos(math.pi*40/150))/2
        torch.save(dict(T_max=150, eta_min=1e-6, last_epoch=40, base_lrs=[2e-4], _last_lr=[lr]), self.root/'scheduler.pt')
        optimizer = dict(param_groups=[dict(lr=lr, betas=(.9, .999), eps=1e-8, weight_decay=0)],
                         state={0: dict(step=torch.tensor(40000.))})
        torch.save(optimizer, self.root/'optimizer.pt')
        rates = []
        for epoch in range(1, 41):
            rate = 1e-6+(2e-4-1e-6)*(1+math.cos(math.pi*(epoch-1)/150))/2
            rates.append(f'[Epoch {epoch}]\tLearning rate: {rate:.2e}')
        (self.root/'log.txt').write_text('\n'.join(rates))
        self.assertEqual(server.audit_schedule(self.root)['status'], 'PASS')
        optimizer['state'][0]['step'] = torch.tensor(39999.)
        torch.save(optimizer, self.root/'optimizer.pt')
        with self.assertRaises(AssertionError):
            server.audit_schedule(self.root)

    def test_config_duplicate(self):
        path = self.root/'config.txt'
        path.write_text('seed: 1\nseed: 1\n')
        with self.assertRaises(ValueError):
            server.config(path)

    def test_shell_helper_is_syntax_checked_not_executed(self):
        helper = self.root/'shutdown'
        helper.write_text('echo fixture\n')
        with patch.object(server, 'Path', return_value=helper), patch.object(server.sys, 'platform', 'linux'), \
                patch.object(server.subprocess, 'run') as run:
            command = server.shutdown_command()
            self.assertEqual(command, ['/bin/bash', str(helper)])
            run.assert_called_once_with(['/bin/bash', '-n', str(helper)], check=True)

    def test_command(self):
        args = SimpleNamespace(data_root=self.root/'data', output=self.root/'results')
        command = list(map(str, server.training_command(args, 4)))
        self.assertEqual(command[command.index('--epochs')+1], '40')
        self.assertEqual(command[command.index('--scheduler_t_max')+1], '150')
        self.assertNotIn('--pre_train', command)
        self.assertNotIn('--load', command)

    def test_child_does_not_inherit_stale_private_references(self):
        stale = dict(N23_BASELINE_MANIFEST='unavailable_old_file', N23_PAIR_MANIFEST='old_pair',
                     N23_PROTOCOL_PROBE_ONLY='1')
        with patch.dict(os.environ, stale), patch.object(server.subprocess, 'Popen') as popen:
            popen.return_value.stdout = []
            popen.return_value.wait.return_value = 0
            server.run(['MOCK_COMMAND_NOT_EXECUTED'], self.root/'child.log')
            env = popen.call_args.kwargs['env']
            self.assertTrue(all(key not in env for key in stale))

    def queue(self, fail=False, launch=True, single=False):
        output = self.root/'results'
        calls = []
        proof = dict(common_initial_state_sha256='fixture', retained_state_sha256='fixture', cpu_rng_sha256='fixture',
                     legacy_common_initial_state_sha256='fixture')
        batches = [dict(epoch=e, lr_sha256=f'fixture{e}', hr_sha256=f'fixture{e}') for e in range(1, 41)]
        def mocked_run(cmd, logfile, cwd=server.ROOT, extra_env=None):
            parts = list(map(str, cmd))
            calls.append(parts)
            if any(p.endswith('check_n23_smoke.py') for p in parts):
                if single:
                    self.assertIn('--n23-only', parts)
                cp = output/('smoke/n23/model/model_1.pt' if single else 'smoke/b0/model/model_1.pt')
                cp.parent.mkdir(parents=True)
                cp.write_bytes(b'not weights: controller mock')
            if any(p.endswith('check_n23_efficiency.py') for p in parts) and fail:
                raise subprocess.CalledProcessError(2, parts)
            if single and any(p.endswith('check_n23_efficiency.py') for p in parts):
                self.assertIn('--n23-only-checkpoint', parts)
                self.assertNotIn('--baseline-checkpoint', parts)
            if '--model' in parts:
                group = 'b0' if parts[parts.index('--model')+1] == 'lfmn_exact_overlap' else 'n23'
                if group == 'b0':
                    self.assertEqual(extra_env, {'N23_PAIR_MANIFEST': str(output/'manifest.json')})
                directory = output/group
                directory.mkdir()
                (directory/'initial_state_proof.json').write_text(json.dumps(proof))
                (directory/'batch_fingerprints.jsonl').write_text('\n'.join(json.dumps(row) for row in batches))
        argv = ['controller', '--n23-only' if single else '--train-new-baseline',
                '--data-root', str(self.root/'data'), '--output', str(output)]
        if launch:
            argv.append('--run')
        env = dict(total_mib=24576, gpu='MOCK_NOT_HARDWARE')
        with patch.object(sys, 'argv', argv), patch.object(server, 'environment', return_value=env), \
                patch.object(server, 'git', side_effect=lambda *a, **kw: '' if a[0] == 'status' else 'MOCK_COMMIT'), \
                patch.object(server, 'data_hashes', return_value={'fixture': 'not actual data'}), \
                patch.object(server, 'source_hashes', return_value={'fixture': 'not actual source'}), \
                patch.object(server, 'gpu_jobs', return_value=[]), \
                patch.object(server.torch.cuda, 'is_available', return_value=True), \
                patch.object(server.shutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)), \
                patch.object(server, 'audit_schedule', return_value={'status': 'PASS'}), \
                patch.object(server, 'audit_baseline', side_effect=AssertionError('Old B0 must not be read')), \
                patch.object(server, 'run', side_effect=mocked_run):
            if fail:
                with self.assertRaises(subprocess.CalledProcessError):
                    server.main()
            else:
                server.main()
        return output, calls

    def test_audit_only_does_not_create_results_or_call_children(self):
        output, calls = self.queue(launch=False)
        self.assertFalse(output.exists())
        self.assertEqual(calls, [])

    def test_fail_closed_before_training(self):
        output, calls = self.queue(fail=True)
        self.assertFalse(any('--model' in cmd for cmd in calls))
        self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'FAILED_NO_SHUTDOWN')

    def test_mocked_queue_order_and_required_check_arguments(self):
        output, calls = self.queue()
        overlap = next(cmd for cmd in calls if any(p.endswith('check_overlap_protocol.py') for p in cmd))
        self.assertIn('--data', overlap)
        self.assertEqual([cmd[cmd.index('--model')+1] for cmd in calls if '--model' in cmd], ['lfmn_n23', 'lfmn_exact_overlap'])
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'COMPLETE')
        self.assertEqual(manifest['comparison_audit']['status'], 'PASS')
        self.assertFalse(manifest['shutdown_requested'])

    def test_n23_only_never_reads_or_trains_b0(self):
        output, calls = self.queue(single=True)
        self.assertEqual([cmd[cmd.index('--model')+1] for cmd in calls if '--model' in cmd], ['lfmn_n23'])
        self.assertFalse((output/'b0').exists())
        self.assertFalse((output/'smoke/b0').exists())
        self.assertFalse(any(any(p.endswith('evaluate_n23_b0.py') for p in cmd) for cmd in calls))
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'COMPLETE_N23_ONLY_BASELINE_PENDING')
        self.assertEqual(manifest['comparison_audit']['status'], 'BASELINE_PENDING')

    def test_single_summary_never_claims_gain(self):
        from summarize_n23_only import single_report
        curves = torch.full((40, 1, 1), 28.)
        report = single_report(curves, torch.full_like(curves, .8))
        self.assertFalse(report['performance_claim'])
        self.assertEqual(report['decision'], 'BASELINE_PENDING_NO_COMPARISON')
        self.assertNotIn('final_delta', report)
        with self.assertRaises(ValueError):
            single_report(curves[:39], curves[:39])

    def test_single_summary_file_contract_not_accuracy(self):
        from summarize_n23_only import summarize
        from summarize_n23_screen import trainer_float32_mean
        group = self.root/'n23'
        (group/'per_image_metrics').mkdir(parents=True)
        (group/'model').mkdir()
        protocol = dict(mode='scratch', epochs=40, overlap='exact_coverage_v1',
                        expected_config=server.expected_config(4))
        manifest = dict(run_mode='N23_ONLY', comparison_audit={'status': 'BASELINE_PENDING'},
                        protocol=protocol, data_hashes={'MOCK': 'NOT_DATA'},
                        commit='MOCK_NOT_EXPERIMENT', environment={'MOCK': True})
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        (group/'config.txt').write_text('model: lfmn_n23\n' + '\n'.join(
            f'{key}: {value}' for key, value in protocol['expected_config'].items()))
        rows = [dict(dataset='DIV2K', scale=4, filename=f'{i:04d}', psnr=28.02, ssim=.8)
                for i in range(801, 901)]
        for metric in ('psnr', 'ssim'):
            mean = trainer_float32_mean([row[metric] for row in rows])
            torch.save(torch.full((40, 1, 1), mean), group/f'{metric}_log.pt')
        mechanism = []
        batches = []
        for epoch in range(1, 41):
            torch.save(rows, group/'per_image_metrics'/f'epoch_{epoch:04d}.pt')
            torch.save({'MOCK_NOT_TRAINED_WEIGHTS': torch.zeros(1)}, group/'model'/f'model_{epoch}.pt')
            mechanism.append(dict(epoch=epoch, updates=1000, total_updates=epoch*1000))
            batches.append(dict(epoch=epoch))
        for name, values in (('mechanism', mechanism), ('batch_fingerprints', batches)):
            (group/f'{name}.jsonl').write_text('\n'.join(json.dumps(row) for row in values))
        result = summarize(self.root)
        self.assertFalse(result['performance_claim'])
        self.assertEqual(result['decision'], 'BASELINE_PENDING_NO_COMPARISON')
        self.assertEqual(len((self.root/'n23_epoch_metrics.csv').read_text().splitlines()), 41)
        self.assertNotIn('final_delta', result)
        (group/'model/model_40.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            summarize(self.root)


if __name__ == '__main__':
    unittest.main()
