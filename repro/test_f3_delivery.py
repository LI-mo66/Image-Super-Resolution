"""Controller/result integrity contracts; synthetic fixtures are not experiments."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_f3_screen_server as run

class Contracts(unittest.TestCase):
    def test_only_candidate_and_pause_horizon(self):
        args=SimpleNamespace(data_root=Path('/data'),workers=4,cpu=False)
        c=run.train_command(args,Path('/output/F3_x4_seed1'))
        self.assertEqual(c[c.index('--model')+1],'LFMNF3')
        self.assertEqual(c[c.index('--scheduler_t_max')+1],'150')
        self.assertEqual(c[c.index('--epochs')+1],'20')
        self.assertEqual(c[c.index('--data_test')+1],'DIV2K')
        self.assertNotIn('--self_ensemble',c)
        self.assertNotIn('--pre_train',c)
    def test_resume_keeps_horizon(self):
        args=SimpleNamespace(data_root=Path('/data'),workers=4,cpu=False)
        c=run.train_command(args,Path('/output/F3_x4_seed1'),resume=7)
        self.assertEqual(c[c.index('--resume')+1],'7')
        self.assertEqual(c[c.index('--scheduler_t_max')+1],'150')
    def test_smoke_launches_only_f3(self):
        args=SimpleNamespace(data_root=Path('/data'),workers=4,cpu=False)
        calls=[]
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as temp:
            def launch(c,d,cfg,cwd,resume=False):
                calls.append((c,resume));d.mkdir(exist_ok=True)
                if resume:
                    (d/'metrics.csv').write_text('epoch\n1\n2\n')
                    (d/'train_log.txt').write_text('Restored RNG/DataLoader state at epoch 1')
            with patch.object(run,'launch',launch),patch.object(run,'config',lambda *a,**kw:dict(kw)):
                run.smoke(args,Path(temp))
        self.assertEqual(len(calls),2)
        self.assertTrue(all(c[c.index('--model')+1]=='LFMNF3' for c,_ in calls))
        self.assertEqual([resume for _,resume in calls],[False,True])
    def test_summary_rejects_incomplete_epochs(self):
        import summarize_f3_screen as summary
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as temp:
            group=Path(temp);d=group/'F3_x4_seed1';d.mkdir()
            (d/'config.json').write_text(json.dumps(dict(run.PROTOCOL,status="completed")))
            (d/'metrics.csv').write_text('epoch\n1\n')
            with patch.object(sys,'argv',['summary',str(group)]),self.assertRaisesRegex(ValueError,'complete epochs'):
                summary.main()
    def test_complete_summary_and_duplicate_benchmark_rejection(self):
        import torch
        import summarize_f3_screen as summary
        with tempfile.TemporaryDirectory(dir=run.ROOT/'experiment') as temp:
            group=Path(temp);d=group/'F3_x4_seed1'
            (d/'model').mkdir(parents=True);(d/'per_image_metrics').mkdir()
            (d/'config.json').write_text(json.dumps(dict(run.PROTOCOL,status="completed")))
            with (d/'metrics.csv').open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=['epoch','validation_psnr','validation_ssim'])
                w.writeheader()
                for epoch in range(1,21):
                    w.writerow(dict(epoch=epoch,validation_psnr=30,validation_ssim=.8))
                    (d/'model'/f'model_{epoch}.pt').write_bytes(b'SYNTHETIC_CONTRACT_FIXTURE')
                    per=[dict(dataset='DIV2K',scale=4,filename=str(i),psnr=30.,ssim=.8) for i in range(100)]
                    torch.save(per,d/'per_image_metrics'/f'epoch_{epoch:04d}.pt')
            b=group/'bench';(b/'per_image_metrics').mkdir(parents=True)
            rows=[dict(dataset=name,scale=4,filename=str(i),psnr=30.,ssim=.8)
                  for name,count in run.BENCHMARKS.items() for i in range(count)]
            torch.save(rows,b/'per_image_metrics/epoch_0000.pt')
            (group/'evaluation_paths.json').write_text(json.dumps(dict(F3=str(b))))
            with patch.object(sys,'argv',['summary',str(group)]),contextlib.redirect_stdout(io.StringIO()):
                summary.main()
            self.assertEqual(json.loads((group/'summary.json').read_text())['status'],
                             'COMPLETE_F3_ONLY_BASELINE_PENDING')
            rows[1]['filename']=rows[0]['filename'];torch.save(rows,b/'per_image_metrics/epoch_0000.pt')
            with patch.object(sys,'argv',['summary',str(group)]),self.assertRaisesRegex(ValueError,'benchmark count'):
                summary.main()

if __name__=='__main__':unittest.main()
