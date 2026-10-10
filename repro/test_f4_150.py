"""Continuation contracts and mock-only shutdown; no real training in these tests."""
import csv
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
import run_f4_150_server as run
import f4_150_monitor as monitor
import shutdown_f4_150 as shutdown
import summarize_f4_150 as summary

class Contracts(unittest.TestCase):
    def test_resume_total150_not_new_cosine(self):
        a=SimpleNamespace(data_root=Path('/data'),workers=4,cpu=False)
        c=run.training_command(a,Path('/out/F4_x4_seed1'),20)
        for k,v in [('--epochs','150'),('--scheduler_t_max','150'),('--resume','20'),('--data_test','DIV2K')]:
            self.assertEqual(c[c.index(k)+1],v)
        self.assertTrue(c[2].endswith('f4_150_train_entry.py'))
        for forbidden in ['--pre_train','--self_ensemble','--chop']:
            self.assertNotIn(forbidden,c)
    def test_private_entry_rejects_budget_reset_and_ON(self):
        a=SimpleNamespace(load='F4_x4_seed1',test_only=False,model='LFMNF4',data_test=['DIV2K'],
            self_ensemble=False,chop=False,scheduler='cosine',scheduler_t_max=150,
            precision='single',scale=[4],rgcrd_mode='off',resume=20,epochs=150)
        monitor.validate_continuation_args(a,{})
        a.epochs=1000
        with self.assertRaisesRegex(ValueError,'budget'):monitor.validate_continuation_args(a,{})
        a.epochs=150;a.scheduler_t_max=130
        with self.assertRaisesRegex(ValueError,'T150'):monitor.validate_continuation_args(a,{})
        a.scheduler_t_max=150;a.self_ensemble=True
        with self.assertRaisesRegex(ValueError,'OFF'):monitor.validate_continuation_args(a,{})

    def test_monitor_preserves_rng_even_exception(self):
        random.seed(3);np.random.seed(3);torch.manual_seed(3)
        py=random.getstate();nr=np.random.get_state();tr=torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with monitor.preserve_rng():
                random.random();np.random.rand();torch.rand(2);raise RuntimeError('fixture')
        self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]))
        self.assertTrue(torch.equal(tr,torch.get_rng_state()))
    def test_missing_restoration_state_blocks(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as tmp:
            with self.assertRaises(FileNotFoundError):run.check_boundary(Path(tmp),20)
    def test_parent_changes_detected(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as tmp:
            root=Path(tmp);parent=root/'parent';(parent/'F4_x4_seed1').mkdir(parents=True)
            model=parent/'F4_x4_seed1/model.pt';model.write_bytes(b'first')
            (root/'continuation_manifest.json').write_text(json.dumps(dict(parent_group=str(parent),
                parent_snapshot=run.snapshot(parent/'F4_x4_seed1'))))
            run.assert_parent_unchanged(root);model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'parent artifacts changed'):run.assert_parent_unchanged(root)
    def test_shutdown_success_and_other_gpu_mock_only(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as tmp:
            group=Path(tmp);(group/'F4_x4_seed1').mkdir()
            (group/'summary150.json').write_text(json.dumps(dict(status='COMPLETE_F4_150_OFF_BASELINE_PENDING',total_epochs=150)))
            (group/'F4_x4_seed1/config.json').write_text(json.dumps(dict(git_commit='fixture')))
            with patch.object(shutdown,'validate_completion'),patch.object(shutdown.sys,'platform','linux'), \
                 patch.object(shutdown.os,'sync',create=True) as sync, \
                 patch.object(shutdown.subprocess,'run',return_value=SimpleNamespace(stdout='')) as call:
                shutdown.shutdown(group,True)
                self.assertEqual(call.call_args_list[-1].args[0],['bash','-c','shutdown -h now'])
                sync.assert_called_once()
            with patch.object(shutdown,'validate_completion'),patch.object(shutdown.sys,'platform','linux'), \
                 patch.object(shutdown.subprocess,'run',return_value=SimpleNamespace(stdout='1234\n')) as call:
                with self.assertRaisesRegex(RuntimeError,'Other GPU'):shutdown.shutdown(group,True)
                self.assertEqual(call.call_count,1)
    def test_shutdown_never_after_invalid_completion(self):
        with patch.object(shutdown,'validate_completion',side_effect=ValueError('invalid')), \
             patch.object(shutdown.subprocess,'run') as call:
            with self.assertRaises(ValueError):shutdown.shutdown('/fixture',True)
            call.assert_not_called()

    def test_full150_summary_requires_all_epochs_actual_OFF_and_resources(self):
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as tmp:
            group=Path(tmp);train=group/'F4_x4_seed1'
            (train/'model').mkdir(parents=True);(train/'per_image_metrics').mkdir()
            curve=[];monitors={}
            for e in range(1,151):
                curve.append(dict(epoch=str(e),validation_psnr='28',validation_ssim='.8'))
                checkpoint=train/'model'/f'model_{e}.pt';checkpoint.write_bytes(str(e).encode())
                monitors[str(e)]=dict(directory='fixture',checkpoint_sha256=run.digest(checkpoint))
                torch.save([dict(dataset='DIV2K',scale=4,filename=str(i),psnr=28.,ssim=.8)
                    for i in range(100)],train/'per_image_metrics'/f'epoch_{e:04d}.pt')
            cfg=dict(status='completed',training_mode='CONTINUATION',checkpoint_selection='fixed epoch150',
                     resolved_arguments=dict(epochs=150,scheduler_t_max=150,self_ensemble=False,chop=False,data_test=['DIV2K']))
            (train/'config.json').write_text(json.dumps(cfg))
            (group/'set5_paths.json').write_text(json.dumps(monitors))
            sha=run.digest(train/'model/model_150.pt')
            (group/'final_evaluation.json').write_text(json.dumps(dict(directory='fixture',checkpoint_sha256=sha)))
            resources=dict(checkpoint_sha256=sha,parameters=766859,self_ensemble=False,
                results={k:dict(counted_flops=1,median_ms=1.,p90_ms=1.,peak_allocated_mib=1.)
                         for k in ['LR64x64','LR128x128','LR96x160']})
            (group/'resources150.json').write_text(json.dumps(resources))
            per=[dict(dataset=name,scale=4,filename=str(i),psnr=30.,ssim=.8)
                 for name,count in run.BENCHMARKS.items() for i in range(count)]
            with patch.object(run,'check_boundary',return_value=curve),patch.object(run,'assert_parent_unchanged'), \
                 patch.object(summary,'read_evaluation',return_value=per):
                rows,benchmarks,_,_=summary.validate_completion(group)
                self.assertEqual(len(rows),150);self.assertEqual(set(benchmarks),set(run.BENCHMARKS))
                del monitors['150'];(group/'set5_paths.json').write_text(json.dumps(monitors))
                with self.assertRaisesRegex(ValueError,'all150'):summary.validate_completion(group)
                monitors['150']=dict(directory='fixture',checkpoint_sha256=sha)
                (group/'set5_paths.json').write_text(json.dumps(monitors))
                cfg['resolved_arguments']['self_ensemble']=True
                (train/'config.json').write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError,'protocol differs'):summary.validate_completion(group)
                cfg['resolved_arguments']['self_ensemble']=False
                (train/'config.json').write_text(json.dumps(cfg));resources['checkpoint_sha256']='wrong'
                (group/'resources150.json').write_text(json.dumps(resources))
                with self.assertRaisesRegex(ValueError,'resources checkpoint'):summary.validate_completion(group)

if __name__=='__main__':unittest.main()
