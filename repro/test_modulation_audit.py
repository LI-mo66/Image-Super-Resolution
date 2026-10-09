"""Synthetic artifact contract tests; no optimizer and no performance claims."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import collect_modulation_audit as audit

class AuditContracts(unittest.TestCase):
    def test_pair_stats_and_duplicate_rejection(self):
        a=[dict(dataset='Set5',filename=str(i),psnr=30.,ssim=.8) for i in range(5)]
        b=[dict(r,psnr=r['psnr']+.1) for r in a]
        stats,rows=audit.compare_rows(a,b)
        self.assertAlmostEqual(stats['Set5']['mean'],.1)
        self.assertEqual(stats['Set5']['positive'],5)
        self.assertEqual(len(rows),5)
        b[1]['filename']='0'
        with self.assertRaises(ValueError):audit.compare_rows(a,b)
    def test_locate_requires_checkpoint_hash(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT/'experiment') as temp:
            root=Path(temp);run=root/'F3_x4_seed1';(run/'model').mkdir(parents=True)
            (run/'config.json').write_text('{}');(run/'model/model_20.pt').write_bytes(b'mock')
            expected=hashlib.sha256(b'mock').hexdigest()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.locate(root,'F3',expected),run.resolve())
            with self.assertRaises(FileNotFoundError):audit.locate(root,'F3','bad')
    def test_csv_does_not_overwrite_input(self):
        with tempfile.TemporaryDirectory(dir=audit.ROOT/'experiment') as temp:
            target=Path(temp)/'output.csv';audit.write_csv(target,[dict(epoch=1,psnr=30.)])
            self.assertEqual(audit.csv_rows(target)[0]['epoch'],'1')
    def test_complete_metadata_collection_no_training(self):
        import torch
        with tempfile.TemporaryDirectory(dir=audit.ROOT/'experiment') as temp:
            root=Path(temp);digests={}
            for label in ['B0','F3']:
                run=root/(label+'_x4_seed1');(run/'model').mkdir(parents=True)
                (run/'per_image_metrics').mkdir()
                (run/'model/model_20.pt').write_bytes(label.encode())
                digests[label]=audit.sha(run/'model/model_20.pt')
                (run/'config.json').write_text(json.dumps(dict(source_sha256={},status='completed')))
                audit.write_csv(run/'metrics.csv',[dict(epoch=i,validation_psnr=30.,validation_ssim=.8) for i in range(1,21)])
                for epoch in range(1,21):
                    torch.save([dict(dataset='DIV2K',scale=4,filename=str(i),psnr=30.,ssim=.8) for i in range(100)],
                               run/'per_image_metrics'/f'epoch_{epoch:04d}.pt')
                bench=root/(label+'_benchmarks_epoch20_OFF')/'per_image_metrics';bench.mkdir(parents=True)
                torch.save([dict(dataset=ds,scale=4,filename=str(i),psnr=30.,ssim=.8)
                            for ds,n in audit.BENCHMARKS.items() for i in range(n)],bench/'epoch_0000.pt')
            argv=['audit','--search-root',str(root),'--data-root',str(root),'--device','cpu',
                  '--expected-b0-sha',digests['B0'],'--expected-f3-sha',digests['F3']]
            with patch.object(audit,'ROOT',root),patch.object(audit.sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                audit.main()
            output=next((root/'experiment/audits').glob('*/audit.json'))
            report=json.loads(output.read_text())
            self.assertEqual(len(report['validation_by_epoch']),20)
            self.assertEqual(report['benchmark_comparison']['Manga109']['n'],109)

if __name__=='__main__':unittest.main()
