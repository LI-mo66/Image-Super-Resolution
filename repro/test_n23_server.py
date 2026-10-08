"""Controller and scheduler contract tests; MOCKS ARE NOT SR RESULTS."""
import json
import math
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

    def queue(self, fail=False, launch=True):
        output = self.root/'results'
        calls = []
        proof = dict(common_initial_state_sha256='fixture', retained_state_sha256='fixture', cpu_rng_sha256='fixture',
                     legacy_common_initial_state_sha256='fixture')
        batches = [dict(epoch=e, lr_sha256=f'fixture{e}', hr_sha256=f'fixture{e}') for e in range(1, 41)]
        def mocked_run(cmd, logfile, cwd=server.ROOT, extra_env=None):
            parts = list(map(str, cmd))
            calls.append(parts)
            if any(p.endswith('check_n23_smoke.py') for p in parts):
                cp = output/'smoke/b0/model/model_1.pt'
                cp.parent.mkdir(parents=True)
                cp.write_bytes(b'not weights: controller mock')
            if any(p.endswith('check_n23_efficiency.py') for p in parts) and fail:
                raise subprocess.CalledProcessError(2, parts)
            if '--model' in parts:
                group = 'b0' if parts[parts.index('--model')+1] == 'lfmn_exact_overlap' else 'n23'
                if group == 'b0':
                    self.assertEqual(extra_env, {'N23_PAIR_MANIFEST': str(output/'manifest.json')})
                directory = output/group
                directory.mkdir()
                (directory/'initial_state_proof.json').write_text(json.dumps(proof))
                (directory/'batch_fingerprints.jsonl').write_text('\n'.join(json.dumps(row) for row in batches))
        argv = ['controller', '--train-new-baseline', '--data-root', str(self.root/'data'), '--output', str(output)]
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


if __name__ == '__main__':
    unittest.main()
