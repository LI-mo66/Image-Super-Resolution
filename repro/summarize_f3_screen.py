"""Validate candidate collection and print paste-ready absolute results."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))

def main():
    import torch
    from run_f3_screen_server import BENCHMARKS,PROTOCOL
    group=Path(sys.argv[1]).resolve();directory=group/'F3_x4_seed1'
    cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    if cfg.get('status')!='completed':raise ValueError('training did not complete normally')
    for key,value in PROTOCOL.items():
        if cfg.get(key)!=value:raise ValueError('protocol differs: '+key)
    with (directory/'metrics.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    if [int(r['epoch']) for r in rows]!=list(range(1,21)):raise ValueError('requires complete epochs1-20')
    for epoch in range(1,21):
        if not (directory/'model'/f'model_{epoch}.pt').is_file():raise FileNotFoundError('checkpoint '+str(epoch))
        per=torch.load(directory/'per_image_metrics'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
        if len(per)!=100 or len({r['filename'] for r in per})!=100:raise ValueError('DIV2K count')
        if any(r['dataset']!='DIV2K' or r['scale']!=4 for r in per):raise ValueError('validation protocol')
        if not all(math.isfinite(r[k]) for r in per for k in ('psnr','ssim')):raise ValueError('nonfinite')
    paths=json.loads((group/'evaluation_paths.json').read_text(encoding='utf-8'))
    bench=torch.load(Path(paths['F3'])/'per_image_metrics/epoch_0000.pt',map_location='cpu',weights_only=True)
    report=dict(status='COMPLETE_F3_ONLY_BASELINE_PENDING',protocol=PROTOCOL,
        checkpoint_sha256=hashlib.sha256((directory/'model/model_20.pt').read_bytes()).hexdigest(),
        final=rows[-1],last5={k:sum(float(r[k]) for r in rows[-5:])/5 for k in ('validation_psnr','validation_ssim')},
        benchmarks={},comparison='not available; B0/F1 results to be supplied by user')
    for name,count in BENCHMARKS.items():
        selected=[r for r in bench if r['dataset']==name and r['scale']==4]
        if len(selected)!=count or len({r['filename'] for r in selected})!=count:raise ValueError('benchmark count '+name)
        if not all(math.isfinite(r[k]) for r in selected for k in ('psnr','ssim')):raise ValueError('nonfinite benchmark')
        report['benchmarks'][name]={k:sum(r[k] for r in selected)/count for k in ('psnr','ssim')}
    with (group/'benchmark_per_image.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset','scale','filename','psnr','ssim']);writer.writeheader();writer.writerows(bench)
    text=['F3 ONLY | scratch x4 | seed1 | T_max150 | fixed epoch20 | ensemble OFF',
        'DIV2K final PSNR={} SSIM={}'.format(rows[-1]['validation_psnr'],rows[-1]['validation_ssim']),
        'DIV2K last5 PSNR={validation_psnr:.9f} SSIM={validation_ssim:.9f}'.format(**report['last5'])]
    for name,values in report['benchmarks'].items():text.append('{} PSNR={psnr:.9f} SSIM={ssim:.9f}'.format(name,**values))
    text+=['Checkpoint SHA256: '+report['checkpoint_sha256'],
        'STOP AT EPOCH20. Baseline comparison pending. No GO/NO-GO or architecture gain claim.']
    output='\n'.join(text);(group/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (group/'summary.txt').write_text(output,encoding='utf-8');print(output,flush=True)

if __name__=='__main__':main()
