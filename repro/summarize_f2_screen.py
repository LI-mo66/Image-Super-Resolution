#!/usr/bin/env python3
"""Audit twenty checkpoint-linked Set5 records; select by DIV2K only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
import torch
from run_logging import write_json


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group',type=Path)
    args=parser.parse_args()
    group=args.group.resolve(); run=group/'F2_train_x4_seed1'
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    if config['status']!='completed': raise ValueError('Training did not finish successfully')
    rows=read_csv(run/'set5_per_checkpoint.csv')
    images=read_csv(run/'set5_per_image.csv')
    metrics=read_csv(run/'metrics.csv')
    expected=list(range(1,21))
    if [int(r['epoch']) for r in rows]!=expected or [int(r['epoch']) for r in metrics]!=expected:
        raise ValueError('Expected exactly checkpoint/metrics epochs1..20')
    validation=[]
    for row in rows:
        epoch=int(row['epoch']); path=run/'model'/('model_{}.pt'.format(epoch))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=row['checkpoint_sha256'] or row['self_ensemble']!='False':
            raise ValueError('Checkpoint hash or ensemble mismatch')
        selected=[r for r in images if int(r['epoch'])==epoch]
        if len(selected)!=5 or len({r['filename'] for r in selected})!=5:
            raise ValueError('Each checkpoint requires five Set5 image records')
        if any(r['checkpoint_sha256']!=digest for r in selected):
            raise ValueError('Set5 per-image checkpoint differs')
        for key,dest in [('psnr','set5_psnr'),('ssim','set5_ssim')]:
            mean=sum(float(r[key]) for r in selected)/5
            if not math.isfinite(mean) or abs(mean-float(row[dest]))>1e-10:
                raise ValueError('Set5 aggregation mismatch')
        raw=torch.load(run/'per_image_metrics'/('epoch_{:04d}.pt'.format(epoch)),
                       map_location='cpu',weights_only=True)
        valid=[r for r in raw if r['dataset']=='DIV2K']
        if len(valid)!=100 or {r['filename'] for r in valid}!={str(i).zfill(4) for i in range(801,901)}:
            raise ValueError('Expected DIV2K validation801..900 at each epoch')
        if any(not math.isfinite(r[key]) for r in valid for key in ('psnr','ssim')):
            raise ValueError('Nonfinite DIV2K validation metric')
        validation.append(dict(epoch=epoch,psnr=sum(r['psnr'] for r in valid)/100,
                               ssim=sum(r['ssim'] for r in valid)/100))
    best=max(validation,key=lambda r:r['psnr'])
    final=validation[-1]
    means=read_csv(group/'benchmark_epoch20.csv')
    counts={'Set5':5,'Set14':14,'B100':100,'Urban100':100,'Manga109':109}
    finalhash=rows[-1]['checkpoint_sha256']
    if len(means)!=5 or {r['dataset'] for r in means}!=set(counts): raise ValueError('Missing fixed epoch20 benchmark')
    for r in means:
        if any(not math.isfinite(float(r[k])) for k in ('psnr','ssim')):
            raise ValueError('Nonfinite benchmark mean')
        if r['checkpoint_sha256']!=finalhash or int(r['image_count'])!=counts[r['dataset']] or r['self_ensemble']!='False':
            raise ValueError('Benchmark protocol/checkpoint mismatch')
    set5=next(r for r in means if r['dataset']=='Set5')
    if abs(float(set5['psnr'])-float(rows[-1]['set5_psnr']))>1e-6:
        raise ValueError('Reloaded epoch20 Set5 disagrees with original epoch20 evaluation')
    result=dict(status='WAIT_BASELINE_AUDIT',training_completed=True,
                performance_claim='No delta or GO decision until actual B0 protocol/checkpoints audited',
                fixed_epoch20=final,best_by_DIV2K=best,
                validation_last5_psnr=sum(r['psnr'] for r in validation[-5:])/5,
                set5_last5_psnr=sum(float(r['set5_psnr']) for r in rows[-5:])/5,
                per_epoch_validation=validation,benchmark_epoch20=means,
                final_checkpoint_sha256=finalhash,
                no_auto_continue=True,self_ensemble=False)
    write_json(group/'summary.json',result)
    lines=['# F2 20 epoch results (Self-Ensemble OFF)',
           '', 'Status: WAIT_BASELINE_AUDIT; no improvement claim or automatic continuation.',
           '', 'DIV2K fixed epoch20 PSNR: {:.10f}; last5: {:.10f}'.format(final['psnr'],result['validation_last5_psnr']),
           'Best by independent DIV2K only: epoch{} PSNR {:.10f}'.format(best['epoch'],best['psnr']),
           '', '## Every saved checkpoint: Set5 x4 OFF', '',
           '| Epoch | Set5 PSNR | Set5 SSIM | Checkpoint SHA256 |',
           '|---:|---:|---:|---|']
    for r in rows:
        lines.append('| {} | {:.10f} | {:.10f} | {} |'.format(r['epoch'],float(r['set5_psnr']),float(r['set5_ssim']),r['checkpoint_sha256']))
    lines+=['', '## Fixed epoch20 benchmark results', '',
            '| Dataset | PSNR | SSIM | Images |', '|---|---:|---:|---:|']
    for r in means:
        lines.append('| {} | {:.10f} | {:.10f} | {} |'.format(r['dataset'],float(r['psnr']),float(r['ssim']),r['image_count']))
    lines+=['', 'No cross-dataset mean. No Set5-based checkpoint selection.',
            'Keep config.json, train_log.txt, metrics.csv, per-image metrics and model_1..20.pt.',
            'Compare only against a verified B0 with the same complete training/metric protocol.']
    (group/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('SUMMARY WRITTEN: '+str(group/'summary.md'),flush=True)


if __name__=='__main__': main()
