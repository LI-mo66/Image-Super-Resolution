#!/usr/bin/env python3
"""F2 only: scratch cosine150, stop20, checkpoint-linked Set5 every epoch."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from run_logging import launch, write_json

BENCHMARKS = {'Set5': 5, 'Set14': 14, 'B100': 100, 'Urban100': 100, 'Manga109': 109}
PROTOCOL = dict(scale=4, seed=1, patch_size=256, batch_size=4, optimizer='ADAM',
                betas=[0.9, 0.999], epsilon=1e-8, weight_decay=0, lr=2e-4,
                eta_min=1e-6, scheduler='cosine', scheduler_t_max=150,
                planned_epochs=150, stop_epoch=20, test_every=1000,
                data_range='1-800/801-900', loss='1*L1', initialization='scratch',
                self_ensemble=False, chop=False, pretrained_checkpoint=None,
                weight_ema=False, tab_centroid_ema=True, precision='single',
                rgb_range=255, tab_widths=[96]*8,
                lrsa_widths=[64,80,112,128,64,80,112,128])
SOURCE_PATHS = ['LFMN/model/lfmn.py','LFMN/model/lfmnf2.py','LFMN/model/__init__.py',
                'LFMN/trainer.py','LFMN/utility.py','LFMN/option.py',
                'LFMN/data/__init__.py','LFMN/data/common.py','LFMN/data/srdata.py',
                'LFMN/data/div2k.py','LFMN/loss/__init__.py','LFMN/run_logging.py',
                'repro/f2_train_entry.py','repro/run_f2_screen_server.py',
                'repro/check_f2_budget.py','repro/check_training_run_logging.py','repro/check_f2_summary.py',
                'repro/summarize_f2_screen.py']


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def source_hashes():
    # Normalize line endings so Windows/Linux checkouts fingerprint the same code.
    return {p: hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in SOURCE_PATHS}


def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def preflight(args, smoke=False):
    from PIL import Image
    pairs=[]
    for split, ids in [('train',[1] if smoke else range(1,801)),
                       ('valid',[801] if smoke else range(801,901))]:
        for i in ids:
            stem='{:04d}'.format(i)
            pairs.append((args.data_root/'DIV2K'/('DIV2K_'+split+'_HR')/(stem+'.png'),
                          args.data_root/'DIV2K'/('DIV2K_'+split+'_LR_bicubic')/'X4'/(stem+'x4.png')))
    for name,count in ({'Set5':5} if smoke else BENCHMARKS).items():
        folder=args.data_root/'benchmark'/name
        high=sorted((folder/'HR').glob('*.png'))
        if len(high)!=count:
            raise ValueError('{}: expected {} PNG images, got {}'.format(name,count,len(high)))
        pairs.extend((p,folder/'LR_bicubic'/'X4'/(p.stem+'x4.png')) for p in high)
    for hr,lr in pairs:
        if not hr.is_file() or not lr.is_file():
            raise FileNotFoundError('Missing LR/HR pair: {} / {}'.format(lr,hr))
        with Image.open(hr) as image: hs=image.size
        with Image.open(lr) as image: ls=image.size
        if any(h//4!=l for h,l in zip(hs,ls)):
            raise ValueError('x4 dimensions mismatch: {}'.format(hr))
    print('DATA PREFLIGHT: {} pairs; content identity with B0 must be confirmed'.format(len(pairs)),flush=True)


def config(args, command, **extra):
    import torch
    result=dict(PROTOCOL, experiment_name='F2', model='LFMNF2',
                dataset='DIV2K', train_range='1-800', validation_range='801-900',
                run_id=('F2_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')),
                start_time=datetime.datetime.now().astimezone().isoformat(),
                model_source='LFMN/model/lfmnf2.py',epochs=20,steps_per_epoch=1000,
                learning_rate=2e-4,ema_enabled=False,self_ensemble_enabled=False,
                degradation='provided DIV2K/benchmark bicubic X4 LR',augmentation=True,
                command=command,cwd=str(ROOT/'LFMN'),workers=args.workers,
                data_root=str(args.data_root),gpu_name='CPU' if args.cpu else torch.cuda.get_device_name(0),
                torch_version=torch.__version__,cuda_version=torch.version.cuda,
                device='cpu' if args.cpu else 'cuda',world_size=1,
                git_commit=git('rev-parse','HEAD'),git_dirty=bool(git('status','--porcelain')),
                git_diff_summary=git('diff','--stat'),source_sha256=source_hashes(),
                baseline_source_commit='88bdc6a',baseline_run=None,
                baseline_reuse='pending actual B0 config/checkpoint audit',
                metric_protocol='original utility.py; quantize255; DIV2K RGB/crop10; benchmark PSNR Y BT601-256/crop4; SSIM MATLAB Y/crop4',
                checkpoint_selection='fixed epoch20; best by independent DIV2K only',
                **extra)
    result.setdefault('resume_checkpoint',None)
    result.setdefault('resume_start_epoch',0)
    if extra.get('smoke'):
        result.update(data_range='1-1/801-801',train_range='1-1',validation_range='801-801',
                      test_every=1,max_train_batches=1,steps_per_epoch=1,
                      stop_epoch=int(command[command.index('--epochs')+1]),
                      epochs=int(command[command.index('--epochs')+1]))
    return result


def train_command(args, directory, smoke=False, resume=0, stop=20):
    command=[sys.executable,'-u',str(ROOT/'repro/f2_train_entry.py'),
             '--dir_data',str(args.data_root),'--model','LFMNF2',
             '--data_train','DIV2K','--data_test','DIV2K+Set5',
             '--data_range','1-1/801-801' if smoke else PROTOCOL['data_range'],
             '--scale','4','--patch_size','256','--batch_size','4',
             '--n_threads',str(args.workers),'--ext','img','--epochs',str(stop),
             '--test_every','1' if smoke else '1000','--lr','2e-4','--optimizer','ADAM',
             '--scheduler','cosine','--scheduler_t_max','150','--eta_min','1e-6',
             '--loss','1*L1','--seed','1','--precision','single',
             '--print_every','1' if smoke else '100','--save_per_image_metrics',
             '--experiment_root',str(directory.parent),'--save',directory.name]
    if args.cpu: command+=['--cpu']
    if smoke: command+=['--max_train_batches','1']
    if resume: command+=['--load',directory.name,'--resume',str(resume)]
    return command


def evaluate(args, group, source, datasets, label):
    import torch
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory=group/(label+'_OFF_'+stamp)
    command=[sys.executable,'-u',str(ROOT/'repro/f2_train_entry.py'),
             '--dir_data',str(args.data_root),'--model','LFMNF2','--scale','4',
             '--test_only','--pre_train',str(source),'--data_test','+'.join(datasets),
             '--n_threads',str(args.workers),'--ext','img','--save_per_image_metrics',
             '--experiment_root',str(group),'--save',directory.name]
    if args.cpu: command+=['--cpu']
    launch(command,directory,config(args,command,test_only=True,checkpoint=str(source),
           checkpoint_sha256=digest),ROOT/'LFMN')
    rows=torch.load(directory/'per_image_metrics/epoch_0000.pt',map_location='cpu',weights_only=True)
    write_csv(directory/'per_image_metrics.csv',rows)
    means=[]
    for name,count in datasets.items():
        selected=[r for r in rows if r['dataset']==name]
        if len(selected)!=count or len({r['filename'] for r in selected})!=count:
            raise ValueError('Incorrect unique image count for '+name)
        means.append(dict(dataset=name,checkpoint=str(source),checkpoint_sha256=digest,
                          psnr=sum(r['psnr'] for r in selected)/count,
                          ssim=sum(r['ssim'] for r in selected)/count,
                          image_count=count,scale=4,self_ensemble=False))
    write_csv(directory/'benchmark_means.csv',means)
    write_json(group/(label+'_evaluation.json'),dict(directory=str(directory),
               checkpoint=str(source),checkpoint_sha256=digest))
    return rows,means


def smoke(args, group):
    directory=group/'F2_smoke_x4_seed1'
    for stop,resume in [(1,0),(2,1)]:
        command=train_command(args,directory,smoke=True,resume=resume,stop=stop)
        options=dict(smoke=True,resume_checkpoint=str(directory/'model/model_1.pt') if resume else None,
                     resume_start_epoch=resume)
        launch(command,directory,config(args,command,**options),ROOT/'LFMN',resume=bool(resume))
    if [int(r['epoch']) for r in csv_rows(directory/'metrics.csv')]!=[1,2]:
        raise AssertionError('Smoke resume timeline failed')
    log=(directory/'train_log.txt').read_text(encoding='utf-8')
    if 'Restored RNG/DataLoader state at epoch 1' not in log:
        raise AssertionError('RNG/checkpoint resume trace missing')
    expected={r['filename']:r for r in csv_rows(directory/'set5_per_image.csv') if r['epoch']=='2'}
    actual,_=evaluate(args,group,directory/'model/model_2.pt',{'Set5':5},'smoke_reload')
    for row in actual:
        for key in ('psnr','ssim'):
            if abs(row[key]-float(expected[row['filename']][key]))>1e-6:
                raise AssertionError('Strict reload changed '+key)
    print('GATE2 PASSED: real batch, validation, all checkpoint Set5, strict reload, resume',flush=True)
    write_json(group/'gate2.json',dict(passed=True,smoke_directory=str(directory),
               real_batch=True,set5_reload_equal=True,resume_epochs=[1,2]))


def checks(args, group):
    for label,script,options in [
        ('logging','check_training_run_logging.py',[]),
        ('summary','check_f2_summary.py',[]),
        ('structure','check_f2_budget.py',['--device','cpu' if args.cpu else 'cuda',
                                         '--output',str(group/'f2_structure_check.json')])]:
        directory=group/(label+'_check_x4_seed1')
        command=[sys.executable,'-u',str(ROOT/'repro'/script)]+options
        launch(command,directory,config(args,command,engineering_check=True),ROOT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'datasets')
    parser.add_argument('--output-root',type=Path,default=ROOT/'experiment/all_runs')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cpu',action='store_true',help='engineering smoke only')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--resume',type=Path)
    modes.add_argument('--eval-only',type=Path)
    args=parser.parse_args()
    args.data_root=args.data_root.resolve()
    if args.workers<1: raise ValueError('workers>=1 required; match actual B0 worker count')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:
        raise ValueError('F2 delivery is single process/single GPU')
    if args.cpu and not (args.check_only or args.smoke_only):
        raise ValueError('20e requires CUDA')
    # Dirty edits cannot silently change a official server run.
    if not (args.check_only or args.smoke_only) and git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Tracked source changes detected; use the committed F2 package')
    preflight(args,smoke=args.smoke_only or args.check_only)
    old=args.resume or args.eval_only
    if old:
        group=old.resolve()
        manifest=json.loads((group/'protocol.json').read_text(encoding='utf-8'))
        if (manifest['protocol']!=PROTOCOL or manifest['commit']!=git('rev-parse','HEAD') or
                manifest['source_sha256']!=source_hashes() or manifest['workers']!=args.workers or
                manifest['data_root']!=str(args.data_root)):
            raise ValueError('Resume/evaluation protocol, source, worker count or dataset path changed')
        if manifest.get('engineering_only'):
            raise ValueError('Do not resume a smoke group as a 20e experiment')
    else:
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        group=args.output_root.resolve()/('F2_x4_seed1_'+stamp)
        group.mkdir(parents=True,exist_ok=False)
        write_json(group/'protocol.json',dict(protocol=PROTOCOL,commit=git('rev-parse','HEAD'),
                   source_sha256=source_hashes(),workers=args.workers,data_root=str(args.data_root),
                   engineering_only=bool(args.check_only or args.smoke_only)))
        print('RUN GROUP: '+str(group),flush=True)
        checks(args,group)
        if args.check_only: return
        smoke(args,group)
        if args.smoke_only: return
    if not (group/'gate2.json').is_file(): raise ValueError('Gate2 smoke report missing')
    directory=group/'F2_train_x4_seed1'
    if not args.eval_only:
        epoch=0
        if directory.exists():
            rows=csv_rows(directory/'metrics.csv')
            epoch=int(rows[-1]['epoch']) if rows else 0
            if not epoch: raise ValueError('No complete checkpoint; preserve failed run and start a new group')
            saved=json.loads((directory/'config.json').read_text(encoding='utf-8'))
            import torch
            gpu=torch.cuda.get_device_name(0)
            if (saved['gpu_name']!=gpu or saved['torch_version']!=torch.__version__ or
                    saved['cuda_version']!=torch.version.cuda):
                raise ValueError('Resume environment differs from original run')
        if epoch<20:
            command=train_command(args,directory,resume=epoch)
            launch(command,directory,config(args,command,resume_checkpoint=str(directory/'model'/('model_{}.pt'.format(epoch))) if epoch else None,
                   resume_start_epoch=epoch),ROOT/'LFMN',resume=bool(epoch))
    source=directory/'model/model_20.pt'
    _,means=evaluate(args,group,source,BENCHMARKS,'epoch20')
    write_csv(group/'benchmark_epoch20.csv',means)
    subprocess.run([sys.executable,str(ROOT/'repro/summarize_f2_screen.py'),str(group)],check=True,cwd=ROOT)
    print('STOP AT EPOCH20. No B0 training, auto-continuation or shutdown.',flush=True)


if __name__=='__main__':
    main()
