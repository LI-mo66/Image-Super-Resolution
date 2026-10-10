"""Mock only. These tests never request a real shutdown."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import shutdown_f4_completed_run as helper

class ShutdownContracts(unittest.TestCase):
    def setUp(self):
        (Path(__file__).resolve().parents[1]/'experiment').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'experiment')
        self.group=Path(self.tmp.name)
        d=self.group/'F4_x4_seed1';(d/'model').mkdir(parents=True)
        ckpt=d/'model/model_20.pt';ckpt.write_bytes(b'mock')
        digest=hashlib.sha256(ckpt.read_bytes()).hexdigest()
        (d/'config.json').write_text(json.dumps(dict(status='completed')))
        (self.group/'summary.json').write_text(json.dumps(dict(status='COMPLETE_F4_ONLY_BASELINE_PENDING',
            protocol=dict(scheduler_t_max=150),final=dict(epoch=20),checkpoint_sha256=digest,
            benchmarks={k:{} for k in ['Set5','Set14','B100','Urban100','Manga109']})))
        (self.group/'resources.json').write_text(json.dumps(dict(checkpoint_sha256=digest,
            results={'LR64x64':{},'LR128x128':{},'LR96x160':{}})))
    def tearDown(self):self.tmp.cleanup()
    def test_no_invoke_never_calls_subprocess(self):
        with patch.object(helper.subprocess,'run') as call:
            helper.shutdown(self.group);call.assert_not_called()
    def test_success_order_mock_only(self):
        with (patch.object(helper.sys,'platform','linux'),
              patch.object(helper.os,'sync',create=True) as sync,
              patch.object(helper.subprocess,'run',return_value=SimpleNamespace(stdout='')) as call):
            helper.shutdown(self.group,True)
            self.assertEqual(call.call_args_list[-1].args[0],['bash','-c','shutdown -h now'])
            sync.assert_called_once()
    def test_other_gpu_job_blocks(self):
        with patch.object(helper.sys,'platform','linux'),patch.object(helper.subprocess,'run',
                return_value=SimpleNamespace(stdout='1234\n')) as call:
            with self.assertRaisesRegex(RuntimeError,'Other GPU'):helper.shutdown(self.group,True)
            self.assertEqual(call.call_count,1)
    def test_corrupt_checkpoint_blocks(self):
        (self.group/'F4_x4_seed1/model/model_20.pt').write_bytes(b'changed')
        with patch.object(helper.subprocess,'run') as call:
            with self.assertRaises(ValueError):helper.shutdown(self.group,True)
            call.assert_not_called()

if __name__=='__main__':unittest.main()
