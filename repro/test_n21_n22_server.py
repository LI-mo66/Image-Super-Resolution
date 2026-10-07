"""Power-off safety tests: every external operation is mocked; no real shutdown."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_n21_n22_screen_server as queue


class QueueSafety(unittest.TestCase):
    def execute(self, failure=None, other=False, power_failure=False):
        scratch=queue.ROOT/'experiment'
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='n21_n22_safety_',dir=scratch) as temp:
            output=Path(temp)/'output'
            output.mkdir()
            data=Path(temp)/'data'
            data.mkdir()
            events=[]
            p=queue.protocol(False)
            manifest={'status':'PREPARED','commit':'unit','source_hashes':{'test':'unit'},
                      'data_hashes':{'test':'unit'},'protocol':p,'data_root':str(data.resolve()),
                      'output':str(output.resolve()),'environment':{'unit':'unit'},'group_exit_codes':{}}
            queue.write(output/'manifest.json',manifest)
            def mocked_git(*args):
                return 'unit' if args[0]=='rev-parse' else ''
            def run(cmd,cwd,log):
                group=Path(cmd[cmd.index('--save')+1]).name
                events.append(group)
                if failure==group:
                    raise RuntimeError('Injected training failure')
            def summary(_):
                events.append('summary')
                if failure=='summary':
                    raise ValueError('Injected metric corruption')
                return {'comparisons':{}}
            def shutdown(cmd,check):
                assert cmd==['/usr/bin/shutdown'] and check
                assert json.loads((output/'manifest.json').read_text())['status']=='COMPLETE_SHUTDOWN_REQUESTED'
                events.append('shutdown')
                if power_failure:
                    raise subprocess.CalledProcessError(1,cmd)
            with contextlib.ExitStack() as stack:
                for target,value in [('sys.platform','linux'),('sys.argv',['queue','--prepared',
                        '--data-root',str(data),'--output',str(output),'--shutdown-on-success','--dedicated-instance'])]:
                    stack.enter_context(patch(target,value))
                stack.enter_context(patch.object(queue,'git',side_effect=mocked_git))
                stack.enter_context(patch.object(queue,'code_hashes',return_value={'test':'unit'}))
                stack.enter_context(patch.object(queue,'environment',return_value={'unit':'unit'}))
                stack.enter_context(patch.object(queue,'data_hashes',return_value={'test':'unit'}))
                stack.enter_context(patch.object(queue,'other_gpu_pids',side_effect=[[],[123456] if other else []]))
                stack.enter_context(patch.object(queue,'run',side_effect=run))
                stack.enter_context(patch.object(queue,'summarize',side_effect=summary))
                stack.enter_context(patch.object(queue.torch.cuda,'is_available',return_value=True))
                stack.enter_context(patch.object(queue.torch.cuda,'get_device_properties',return_value=SimpleNamespace(total_memory=24*1024**3)))
                stack.enter_context(patch.object(queue.shutil,'disk_usage',return_value=SimpleNamespace(free=10*1024**3)))
                stack.enter_context(patch.object(queue.os,'sync',side_effect=lambda:events.append('sync'),create=True))
                stack.enter_context(patch.object(queue.os,'access',return_value=True))
                stack.enter_context(patch.object(queue.subprocess,'run',side_effect=shutdown))
                original_is_file=Path.is_file
                stack.enter_context(patch.object(Path,'is_file',lambda path:True if str(path).replace('\\','/')=='/usr/bin/shutdown' else original_is_file(path)))
                try:
                    queue.main()
                except (RuntimeError,ValueError,subprocess.CalledProcessError):
                    if not (failure or other or power_failure):
                        raise
            return events,json.loads((output/'manifest.json').read_text())

    def test_success_flushes_before_shutdown(self):
        events,state=self.execute()
        self.assertEqual(events,['n21','n22','b0','summary','sync','shutdown'])
        self.assertEqual(state['status'],'COMPLETE_SHUTDOWN_REQUESTED')

    def test_training_failure_no_shutdown_no_b0(self):
        events,state=self.execute(failure='n22')
        self.assertEqual(events,['n21','n22'])
        self.assertEqual(state['status'],'FAILED_NO_SHUTDOWN')

    def test_bad_metrics_no_shutdown(self):
        events,state=self.execute(failure='summary')
        self.assertNotIn('shutdown',events)
        self.assertEqual(state['status'],'FAILED_NO_SHUTDOWN')

    def test_other_gpu_blocks_shutdown(self):
        events,state=self.execute(other=True)
        self.assertNotIn('shutdown',events)
        self.assertEqual(state['status'],'COMPLETE_SHUTDOWN_BLOCKED_OTHER_GPU_JOB')

    def test_shutdown_failure_preserves_results_status(self):
        events,state=self.execute(power_failure=True)
        self.assertEqual(state['status'],'COMPLETE_SHUTDOWN_FAILED')
        self.assertEqual(state['group_exit_codes'],{'n21':0,'n22':0,'b0':0})

    def test_shutdown_requires_dedicated_acknowledgment(self):
        with patch('sys.platform','linux'), patch('sys.argv',['queue','--data-root','data',
                '--output','output','--shutdown-on-success']), patch.object(queue.subprocess,'run') as power:
            with self.assertRaises(SystemExit) as caught:
                queue.main()
            self.assertEqual(caught.exception.code,2)
            power.assert_not_called()

if __name__=='__main__':
    unittest.main()
