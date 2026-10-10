"""Read-only Set5 evaluation of all F4 checkpoints and consolidated reporting."""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from run_f4_screen_server import BENCHMARKS,PROTOCOL,config,check_data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_fingerprint(root):
    data_hash=hashlib.sha256()
    folders=[root/'DIV2K'/name for name in (
        'DIV2K_train_HR','DIV2K_train_LR_bicubic/X4','DIV2K_valid_HR','DIV2K_valid_LR_bicubic/X4')]
    folders += [root/'benchmark'/name/sub for name in BENCHMARKS for sub in ('HR','LR_bicubic/X4')]
    for folder in folders:
        for image in sorted(folder.glob('*.png')):
            data_hash.update(str(image.relative_to(root)).replace('\\','/').encode())
            image_hash=hashlib.sha256()
            with image.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):image_hash.update(block)
            data_hash.update(image_hash.digest())
    return data_hash.hexdigest()


def aggregate(rows,name,count):
    selected=[r for r in rows if r['dataset']==name and r['scale']==4]
    if len(selected)!=count or len({r['filename'] for r in selected})!=count:
        raise ValueError('dataset count/duplicate: '+name)
    if not all(math.isfinite(float(r[k])) for r in selected for k in ('psnr','ssim')):
        raise ValueError('nonfinite metric: '+name)
    return {k:sum(float(r[k]) for r in selected)/count for k in ('psnr','ssim')}


def command(args,checkpoint,directory,datasets):
    c=[sys.executable,'-u',str(ROOT/'repro/f4_train_entry.py'),
       '--dir_data',str(args.data_root),'--model','LFMNF4','--scale','4',
       '--test_only','--pre_train',str(checkpoint),'--data_test','+'.join(datasets),
       '--n_threads',str(args.workers),'--ext','img','--save_per_image_metrics',
       '--experiment_root',str(directory.parent),'--save',directory.name]
    if args.cpu:c+=['--cpu']
    return c


def read_evaluation(directory,checkpoint):
    import torch
    directory=Path(directory)
    cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    actual=cfg.get('resolved_arguments',{})
    if (cfg.get('status')!='completed' or cfg.get('input_checkpoint_sha256')!=digest(checkpoint)
        or actual.get('model')!='LFMNF4' or actual.get('self_ensemble') is not False
        or actual.get('chop') is not False):
        raise ValueError('evaluation checkpoint/model/actual OFF integrity')
    return torch.load(directory/'per_image_metrics/epoch_0000.pt',map_location='cpu',weights_only=True)


def evaluate(args,checkpoint,directory,datasets):
    c=command(args,checkpoint,directory,datasets)
    launch(c,directory,config(args,c,test_only=True,checkpoint=str(checkpoint),
        kind='POSTHOC_READONLY_NO_TRAINING'),ROOT/'LFMN')
    return read_evaluation(directory,checkpoint)


def main():
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--reevaluate-final',action='store_true')
    args=p.parse_args();args.group=args.group.resolve();args.data_root=args.data_root.resolve()
    if args.workers<1:raise ValueError('workers must be >=1')
    train=args.group/'F4_x4_seed1'
    cfg=json.loads((train/'config.json').read_text(encoding='utf-8'))
    if cfg.get('status')!='completed':raise ValueError('requires completed F4 epoch20')
    for k,v in PROTOCOL.items():
        if cfg.get(k)!=v:raise ValueError('protocol differs: '+k)
    if Path(cfg['data_root']).resolve()!=args.data_root:raise ValueError('use original data-root')
    # Same evaluator and model as training; new reporting code does not modify them.
    for name,expected in cfg['source_sha256'].items():
        if name.startswith('LFMN/') or name=='repro/f4_train_entry.py':
            if digest(ROOT/name)!=expected:raise ValueError('evaluation source differs: '+name)
    check_data(args.data_root)
    args.data_fingerprint=dataset_fingerprint(args.data_root)
    if args.data_fingerprint!=cfg.get('data_fingerprint'):
        raise ValueError('data fingerprint differs from completed training')
    protected=[train/'config.json',train/'metrics.csv']+[train/'model'/f'model_{e}.pt' for e in range(1,21)]
    protected += [train/'per_image_metrics'/f'epoch_{e:04d}.pt' for e in range(1,21)]
    before={str(path):digest(path) for path in protected}
    with (train/'metrics.csv').open(encoding='utf-8') as f:curve=list(csv.DictReader(f))
    if [int(r['epoch']) for r in curve]!=list(range(1,21)):raise ValueError('need complete epochs1-20')
    for e in range(1,21):
        rows=torch.load(train/'per_image_metrics'/f'epoch_{e:04d}.pt',map_location='cpu',weights_only=True)
        aggregate(rows,'DIV2K',100)
        if not all(math.isfinite(float(curve[e-1][k])) for k in ('validation_psnr','validation_ssim')):
            raise ValueError('nonfinite DIV2K curve')
    out=args.group/('posthoc_Set5_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(exist_ok=False)
    print('REPORT DIR:',out,flush=True)
    write_json(out/'input_manifest.json',dict(original_training_commit=cfg.get('git_commit'),
        protected_sha256=before,data_root=str(args.data_root),data_fingerprint=args.data_fingerprint,
        optimizer_steps=0,purpose='posthoc evaluation only; fixed epoch20; no selection'))
    combined=[];all_set5=[]
    for e,row in enumerate(curve,1):
        checkpoint=train/'model'/f'model_{e}.pt'
        per=evaluate(args,checkpoint,out/f'Set5_epoch{e:02d}_OFF',['Set5'])
        score=aggregate(per,'Set5',5)
        item=dict(epoch=e,DIV2K_PSNR=float(row['validation_psnr']),DIV2K_SSIM=float(row['validation_ssim']),
                  Set5_PSNR=score['psnr'],Set5_SSIM=score['ssim'],checkpoint_sha256=digest(checkpoint))
        combined.append(item)
        all_set5.extend(dict(epoch=e,**r) for r in per)
        print('EPOCH {epoch:02d} DIV2K PSNR={DIV2K_PSNR:.9f} SSIM={DIV2K_SSIM:.9f} '
              'Set5 PSNR={Set5_PSNR:.9f} SSIM={Set5_SSIM:.9f}'.format(**item),flush=True)
        write_json(out/'progress.json',dict(completed_epochs=e,rows=combined))
    paths=args.group/'evaluation_paths.json'
    if not args.reevaluate_final and paths.exists():
        old=Path(json.loads(paths.read_text(encoding='utf-8'))['F4'])
        final=read_evaluation(old,train/'model/model_20.pt')
        final_source=str(old)
    else:
        old=out/'benchmarks_epoch20_OFF'
        final=evaluate(args,train/'model/model_20.pt',old,list(BENCHMARKS))
        final_source=str(old)
    benchmarks={name:aggregate(final,name,count) for name,count in BENCHMARKS.items()}
    if any(abs(benchmarks['Set5'][k]-combined[-1]['Set5_'+k.upper()])>1e-4 for k in ('psnr','ssim')):
        raise ValueError('epoch20 Set5 differs between evaluations; inspect environment/data before reporting')
    after={str(path):digest(path) for path in protected}
    if before!=after:raise ValueError('original artifact changed')
    for filename,rows in [('epochs_DIV2K_Set5.csv',combined),('Set5_per_image.csv',all_set5),('benchmark_epoch20_per_image.csv',final)]:
        with (out/filename).open('x',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    last5={k:sum(r[k] for r in combined[-5:])/5 for k in ('DIV2K_PSNR','DIV2K_SSIM','Set5_PSNR','Set5_SSIM')}
    report=dict(status='COMPLETE_POSTHOC_OFF_NO_TRAINING',epochs=combined,last5=last5,
        benchmarks_epoch20=benchmarks,final_benchmark_source=final_source,optimizer_steps=0,
        original_artifacts_unchanged=True,checkpoint_sha256=digest(train/'model/model_20.pt'))
    lines=['F4 | posthoc OFF | epochs1-20 | fixed epoch20 final benchmarks | optimizer_steps=0']
    lines += ['EPOCH {epoch:02d} DIV2K PSNR={DIV2K_PSNR:.9f} SSIM={DIV2K_SSIM:.9f} '
              'Set5 PSNR={Set5_PSNR:.9f} SSIM={Set5_SSIM:.9f}'.format(**r) for r in combined]
    lines += ['LAST5 '+json.dumps(last5)]
    lines += ['EPOCH20 {} PSNR={psnr:.9f} SSIM={ssim:.9f}'.format(name,**score) for name,score in benchmarks.items()]
    lines += ['Checkpoint SHA256: '+report['checkpoint_sha256'],
              'Original artifacts unchanged. Baseline attribution pending. No checkpoint selection.']
    write_json(out/'combined_report.json',report)
    text='\n'.join(lines);(out/'paste_report.txt').write_text(text,encoding='utf-8')
    print('\n'+text,flush=True)
    print('REPORT DIR:',out,flush=True)


if __name__=='__main__':main()
