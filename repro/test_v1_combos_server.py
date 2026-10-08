"""Cheap contract tests; synthetic metrics are never experimental evidence."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from v1_combo_protocol import EXPECTED, ROOT, TEACHER_COMMIT, TEACHER_SHA, command, validate_v1_config
from run_v1_combos_screen_server import shutdown_command

TEST_TMP = ROOT/'experiment/all_runs/v1_combo_unit_tmp'
TEST_TMP.mkdir(parents=True,exist_ok=True)


class ProtocolTests(unittest.TestCase):
    def config(self, directory, blocks):
        path = Path(directory)/'config.txt'
        path.write_text('\n'.join('\n'.join(f'{k}: {v}' for k,v in block.items()) for block in blocks),encoding='utf-8')
        return path

    def test_three_historical_blocks(self):
        blocks = []
        for epochs,resume in [(20,0),(40,20),(150,40)]:
            block = dict(EXPECTED,epochs=str(epochs),resume=str(resume),load='srpr_c1' if resume else '')
            if resume:
                block['resume_data_epochs'] = str(resume)
            blocks.append(block)
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            self.assertEqual(len(validate_v1_config(self.config(d,blocks))),3)

    def test_changed_scheduler_and_missing_replay_block(self):
        for changes in [dict(scheduler_t_max='20'),dict(n_threads='0'),dict(resume='40',load='srpr_c1')]:
            block = dict(EXPECTED,epochs='150',resume='0',load='',**changes) if 'resume' not in changes else dict(EXPECTED,epochs='150',**changes)
            with tempfile.TemporaryDirectory(dir=TEST_TMP) as d, self.assertRaises(ValueError):
                validate_v1_config(self.config(d,[block]))

    def test_commands_never_train_baseline(self):
        for g in ('n21','n22','n23'):
            cmd = command(g,Path('data'),Path('teacher'),Path('weight'),ROOT/'experiment/all_runs/new'/g)
            self.assertEqual(cmd[cmd.index('--model')+1],'lfmn_v1'+g)
            self.assertEqual(cmd[cmd.index('--n_threads')+1],'8')
            self.assertEqual(cmd[cmd.index('--scheduler_t_max')+1],'150')
            self.assertEqual(cmd[cmd.index('--epochs')+1],'20')
            self.assertNotIn('--rgcrd_teacher_amp',cmd)
            self.assertNotIn('--betas',cmd)  # legacy tuple CLI parser is unsafe
            self.assertNotIn('--load',cmd)
        with self.assertRaises(ValueError):
            command('v1',Path('d'),Path('t'),Path('w'),Path('o'))

    def test_shutdown_text_helper(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            path = Path(d)/'shutdown'
            path.write_bytes(b'echo testing\n')
            with patch('run_v1_combos_screen_server.subprocess.run') as run:
                self.assertEqual(shutdown_command(path),['/bin/bash',str(path)])
                run.assert_called_once_with(['/bin/bash','-n',str(path)],check=True)

    def test_metrics_ordered_float32_and_pending(self):
        import torch
        from summarize_v1_combos import metrics,summarize
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            root = Path(d)
            group = root/'n21'
            group.mkdir()
            (group/'per_image_metrics').mkdir()
            rows = [dict(dataset='DIV2K',scale=4,filename=f'{i:04d}',psnr=28+i/10000,ssim=.8+i/1000000) for i in range(801,901)]
            p,s = torch.tensor(0.),torch.tensor(0.)
            for r in rows:
                p+=r['psnr']; s+=r['ssim']
            torch.save(p.repeat(20,1,1)/100,group/'psnr_log.pt')
            torch.save(s.repeat(20,1,1)/100,group/'ssim_log.pt')
            for e in range(1,21):
                torch.save(rows,group/'per_image_metrics'/f'epoch_{e:04d}.pt')
            self.assertEqual(len(metrics(group)[2]),100)
            with contextlib.redirect_stdout(io.StringIO()):
                payload = summarize(root,['n21'],group)
            self.assertEqual(payload['results']['n21']['decision'],'REFERENCE_ONLY_PROVENANCE_PENDING')
            rows[1]['filename']='0801'
            torch.save(rows,group/'per_image_metrics/epoch_0020.pt')
            with self.assertRaises(ValueError):
                metrics(group)

    def test_shared_core_untouched(self):
        import subprocess
        self.assertEqual(subprocess.check_output(['git','diff','982463c','--','LFMN/model/lfmn.py',
            'LFMN/model/lfmnsrprv2.py','LFMN/trainer.py','LFMN/utility.py','LFMN/data','LFMN/rgcrd'],cwd=ROOT),b'')

    def test_audit_only_never_launches_optimizer(self):
        from run_v1_combos_screen_server import main
        def checked(cmd,cwd=ROOT):
            return '' if 'status' in cmd else TEACHER_COMMIT
        output = TEST_TMP/'audit_only_new'
        with patch('sys.argv',['runner','--collect-only','--data-root','unused','--output',str(output)]), \
             patch('run_v1_combos_screen_server.checked',side_effect=checked), \
             patch('run_v1_combos_screen_server.sha256',return_value=TEACHER_SHA), \
             patch('run_v1_combos_screen_server.gpu_environment',return_value={'gpu':'mock'}), \
             patch('run_v1_combos_screen_server.data_fingerprint',return_value={}), \
             patch('run_v1_combos_screen_server.source_fingerprint',return_value={}), \
             patch('run_v1_combos_screen_server.run_child') as child, contextlib.redirect_stdout(io.StringIO()):
            main()
            child.assert_not_called()
            self.assertFalse(output.exists())

    def test_dirty_blocks_before_child(self):
        from run_v1_combos_screen_server import main
        with patch('sys.argv',['runner','--collect-only','--data-root','unused','--output',str(TEST_TMP/'dirty_new')]), \
             patch('run_v1_combos_screen_server.checked',return_value='?? launcher.log'), \
             patch('run_v1_combos_screen_server.run_child') as child, self.assertRaises(RuntimeError):
            main()
        child.assert_not_called()

    def test_refuse_existing_output(self):
        from run_v1_combos_screen_server import main
        with patch('sys.argv',['runner','--collect-only','--data-root','unused','--output',str(TEST_TMP)]), \
             self.assertRaises(FileExistsError):
            main()

    def test_shutdown_requires_dedicated(self):
        from run_v1_combos_screen_server import main
        with patch('sys.argv',['runner','--collect-only','--data-root','unused','--output',str(TEST_TMP/'power_new'),
                               '--shutdown-on-success']), \
             patch('run_v1_combos_screen_server.checked',return_value=''), self.assertRaises(ValueError):
            main()


if __name__ == '__main__':
    unittest.main()
