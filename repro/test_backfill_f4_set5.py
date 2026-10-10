"""Posthoc integrity and evaluation-only command contracts."""
import contextlib
import csv
import io
import sys
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import backfill_f4_set5 as report

class Contracts(unittest.TestCase):
    def test_test_only_and_OFF(self):
        c=report.command(SimpleNamespace(data_root=Path('/data'),workers=4,cpu=False),
            Path('/model_7.pt'),Path('/report/Set5_epoch07_OFF'),['Set5'])
        self.assertIn('--test_only',c)
        self.assertEqual(c[c.index('--pre_train')+1],str(Path('/model_7.pt')))
        self.assertEqual(c[c.index('--data_test')+1],'Set5')
        for forbidden in ['--self_ensemble','--chop','--load','--resume','--epochs']:
            self.assertNotIn(forbidden,c)
    def test_aggregate_validates_duplicates_and_nonfinite(self):
        rows=[dict(dataset='Set5',scale=4,filename=str(i),psnr=30+i,ssim=.8) for i in range(5)]
        self.assertEqual(report.aggregate(rows,'Set5',5)['psnr'],32)
        rows[1]['filename']='0'
        with self.assertRaisesRegex(ValueError,'duplicate'):report.aggregate(rows,'Set5',5)
        rows[1]['filename']='1';rows[1]['psnr']=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):report.aggregate(rows,'Set5',5)
    def test_eval_requires_matching_checkpoint_and_actual_OFF(self):
        import torch
        (report.ROOT/'experiment').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=report.ROOT/'experiment') as tmp:
            root=Path(tmp);ckpt=root/'model.pt';ckpt.write_bytes(b'fixture')
            (root/'per_image_metrics').mkdir()
            torch.save([],root/'per_image_metrics/epoch_0000.pt')
            cfg=dict(status='completed',input_checkpoint_sha256=report.digest(ckpt),
                     resolved_arguments=dict(model='LFMNF4',self_ensemble=False,chop=False))
            (root/'config.json').write_text(json.dumps(cfg))
            self.assertEqual(report.read_evaluation(root,ckpt),[])
            cfg['resolved_arguments']['self_ensemble']=True
            (root/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'OFF integrity'):report.read_evaluation(root,ckpt)
            cfg['resolved_arguments']['self_ensemble']=False
            (root/'config.json').write_text(json.dumps(cfg));ckpt.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'OFF integrity'):report.read_evaluation(root,ckpt)

    def test_complete_twenty_epoch_reporting_without_training(self):
        import torch
        with tempfile.TemporaryDirectory(dir=report.ROOT/'experiment') as tmp:
            group=Path(tmp);train=group/'F4_x4_seed1'
            (train/'model').mkdir(parents=True);(train/'per_image_metrics').mkdir()
            cfg=dict(report.PROTOCOL,status='completed',data_root=str(group.resolve()),
                     source_sha256={},data_fingerprint='fixture',git_commit='fixture')
            (train/'config.json').write_text(json.dumps(cfg))
            with (train/'metrics.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['epoch','validation_psnr','validation_ssim']);w.writeheader()
                for e in range(1,21):
                    w.writerow(dict(epoch=e,validation_psnr=28+e*.01,validation_ssim=.8))
                    (train/'model'/f'model_{e}.pt').write_bytes(str(e).encode())
                    torch.save([dict(dataset='DIV2K',scale=4,filename=str(i),psnr=28.,ssim=.8)
                        for i in range(100)],train/'per_image_metrics'/f'epoch_{e:04d}.pt')
            calls=[]
            def fake_evaluate(args,checkpoint,directory,datasets):
                calls.append((checkpoint.name,list(datasets)))
                return [dict(dataset=name,scale=4,filename=str(i),psnr=30.,ssim=.8)
                        for name in datasets for i in range(report.BENCHMARKS[name])]
            argv=['script','--group',str(group),'--data-root',str(group),'--cpu']
            with patch.object(sys,'argv',argv),patch.object(report,'check_data'), \
                 patch.object(report,'dataset_fingerprint',return_value='fixture'), \
                 patch.object(report,'evaluate',fake_evaluate),contextlib.redirect_stdout(io.StringIO()):
                report.main()
            self.assertEqual([c[0] for c in calls[:20]],[f'model_{e}.pt' for e in range(1,21)])
            self.assertTrue(all(c[1]==['Set5'] for c in calls[:20]))
            self.assertEqual(calls[-1][0],'model_20.pt')
            out=list(group.glob('posthoc_Set5_*'))[0]
            result=json.loads((out/'combined_report.json').read_text())
            self.assertTrue(result['original_artifacts_unchanged'])
            self.assertEqual(result['optimizer_steps'],0)
            self.assertEqual(len(result['epochs']),20)
            self.assertEqual(set(result['benchmarks_epoch20']),set(report.BENCHMARKS))
            with (out/'epochs_DIV2K_Set5.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),20)

if __name__=='__main__':unittest.main()
