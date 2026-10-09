"""Synthetic engineering fixtures for F2 summary auditing; no experiment results."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def fixture(group):
    run = group / 'F2_train_x4_seed1'
    (run / 'model').mkdir(parents=True, exist_ok=True)
    (run / 'per_image_metrics').mkdir(exist_ok=True)
    (run / 'config.json').write_text(json.dumps({'status':'completed', 'synthetic_fixture':True}), encoding='utf-8')
    checkpoints, images, metrics = [], [], []
    for epoch in range(1,21):
        path = run / 'model' / f'model_{epoch}.pt'
        torch.save({'synthetic_fixture':torch.tensor([epoch])}, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Set5 grows with epoch, but independent validation peaks at epoch7.
        psnr = 25.0 + epoch * 0.01
        raw = [{'dataset':'Set5', 'filename':f'synthetic_set5_{i}', 'psnr':psnr, 'ssim':0.8} for i in range(5)]
        raw += [{'dataset':'DIV2K', 'filename':f'{i:04d}', 'psnr':30.0-abs(epoch-7)*0.01, 'ssim':0.85} for i in range(801,901)]
        torch.save(raw, run/'per_image_metrics'/f'epoch_{epoch:04d}.pt')
        checkpoints.append({'epoch':epoch, 'checkpoint_sha256':digest, 'self_ensemble':False,
                            'set5_psnr':psnr, 'set5_ssim':0.8})
        images += [{'epoch':epoch, 'filename':f'synthetic_set5_{i}', 'checkpoint_sha256':digest,
                    'psnr':psnr, 'ssim':0.8} for i in range(5)]
        metrics.append({'epoch':epoch, 'validation_psnr':30.0-abs(epoch-7)*0.01})
    write_csv(run/'set5_per_checkpoint.csv', checkpoints)
    write_csv(run/'set5_per_image.csv', images)
    write_csv(run/'metrics.csv', metrics)
    means = [{'dataset':dataset, 'image_count':count, 'checkpoint_sha256':checkpoints[-1]['checkpoint_sha256'],
              'self_ensemble':False, 'psnr':25.2 if dataset=='Set5' else 30.0, 'ssim':0.8}
             for dataset,count in [('Set5',5),('Set14',14),('B100',100),('Urban100',100),('Manga109',109)]]
    write_csv(group/'benchmark_epoch20.csv',means)
    for name in ['summary.json','summary.md']:
        path = group/name
        if path.exists():
            path.unlink()
    return run


def invoke(group):
    return subprocess.run([sys.executable,'-X','utf8',str(ROOT/'repro'/'summarize_f2_screen.py'),str(group)],
                          capture_output=True,text=True,encoding='utf-8',timeout=60)


def main():
    base = ROOT/'experiment'
    base.mkdir(exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix='F2_SYNTHETIC_summary_check_', dir=base) as directory:
        group = Path(directory)
        fixture(group)
        success = invoke(group)
        if success.returncode:
            raise AssertionError(success.stdout + success.stderr)
        report = json.loads((group/'summary.json').read_text(encoding='utf-8'))
        assert report['status']=='WAIT_BASELINE_AUDIT'
        assert report['best_by_DIV2K']['epoch']==7
        assert report['fixed_epoch20']['epoch']==20
        assert report['self_ensemble'] is False and report['no_auto_continue'] is True
        assert len(report['per_epoch_validation'])==20
        assert '| 20 |' in (group/'summary.md').read_text(encoding='utf-8')
        results['valid_fixture_waits_for_baseline'] = True
        results['checkpoint_selection_uses_div2k_not_set5'] = True
        cases = [
            ('missing_epoch','set5_per_checkpoint.csv','Expected exactly'),
            ('wrong_hash','set5_per_checkpoint.csv','Checkpoint hash'),
            ('duplicate_filename','set5_per_image.csv','five Set5 image'),
            ('wrong_set5_mean','set5_per_checkpoint.csv','Set5 aggregation mismatch'),
            ('benchmark_mixed_checkpoint','benchmark_epoch20.csv','Benchmark protocol/checkpoint mismatch'),
        ]
        for case,filename,error in cases:
            run = fixture(group)
            path = group/filename if filename.startswith('benchmark') else run/filename
            rows = read_csv(path)
            if case=='missing_epoch':
                rows.pop(4)
            elif case=='wrong_hash':
                rows[0]['checkpoint_sha256']='0'*64
            elif case=='duplicate_filename':
                rows[1]['filename']=rows[0]['filename']
            elif case=='wrong_set5_mean':
                rows[0]['set5_psnr']=str(float(rows[0]['set5_psnr'])+0.1)
            else:
                rows[1]['checkpoint_sha256']='0'*64
            write_csv(path,rows)
            failed = invoke(group)
            if failed.returncode==0 or error not in failed.stderr:
                raise AssertionError(f'{case} did not produce expected rejection: {failed.stdout} {failed.stderr}')
            assert not (group/'summary.json').exists(), f'{case} emitted a misleading summary'
            results['reject_'+case]=True
    print(json.dumps({'passed':True,'scope':'synthetic engineering fixtures only; no training or PSNR evidence',
                      'checks':results}, indent=2))


if __name__=='__main__':
    main()
