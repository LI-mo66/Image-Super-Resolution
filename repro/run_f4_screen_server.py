"""Collect only F4: scratch, Cosine T150, fixed epoch20, no B0 training."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from source_provenance import provenance
from run_f1_screen_server import check_data,BENCHMARKS

PROTOCOL=dict(scale=4,seed=1,patch_size=256,batch_size=4,optimizer='ADAM',
    lr=2e-4,eta_min=1e-6,scheduler='cosine',scheduler_t_max=150,
    planned_epochs=150,stop_epoch=20,test_every=1000,data_range='1-800/801-900',
    loss='1*L1',self_ensemble=False,pretrained_checkpoint=None,weight_ema=False,
    tab_centroid_ema=True,precision='single',rgb_range=255,
    epoch_evaluation=['DIV2K'],checkpoint_selection='fixed epoch20')

def hashes():
    paths=list((ROOT/'LFMN').rglob('*.py'))+list((ROOT/'repro').glob('*f4*.py'))+[
        ROOT/'repro/run_f1_screen_server.py',ROOT/'repro/check_training_run_logging.py',
        ROOT/'repro/run_f4_nohup.sh']
    return {str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def config(args,command,**extra):
    import torch
    return dict(PROTOCOL,historical_B0_initial_shared_state_sha256='f6beb5ea471ae30df47e9d71b83fbee98155e62b583ad21c02c05b205465eb57',candidate_branch='codex/f4-spatial-residual-modulation',experiment_name='F4',model='LFMNF4',model_source='LFMN/model/lfmnf4.py',
        run_id=command[command.index('--save')+1] if '--save' in command else 'F4_engineering',
        start_time=datetime.datetime.now().astimezone().isoformat(),
        command=command,cwd=str(ROOT/'LFMN'),dataset='DIV2K',degradation='provided paired bicubic x4',
        data_root=str(args.data_root),workers=args.workers,augmentation=True,chop=False,
        data_fingerprint=getattr(args,'data_fingerprint',None),betas=[.9,.999],epsilon=1e-8,
        weight_decay=0,batches_per_epoch=1000,initialization='scratch',
        gpu_name='CPU' if args.cpu else torch.cuda.get_device_name(0),world_size=1,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        **provenance(ROOT),source_sha256=hashes(),baseline_reuse='PENDING_ARTIFACT_AUDIT_NO_RETRAIN',
        metric_protocol='DIV2K quantized RGB crop10; benchmark Y PSNR BT601-256 crop4 / MATLAB Y SSIM crop4',
        **extra)

def train_command(args,directory,smoke=False,resume=0):
    c=[sys.executable,'-u',str(ROOT/'repro/f4_train_entry.py'),'--dir_data',str(args.data_root),
       '--model','LFMNF4','--data_train','DIV2K','--data_test','DIV2K',
       '--data_range','1-1/801-801' if smoke else PROTOCOL['data_range'],
       '--scale','4','--patch_size','256','--batch_size','4','--seed','1',
       '--n_threads',str(args.workers),'--ext','img','--epochs',str((2 if resume else 1) if smoke else 20),
       '--test_every','1' if smoke else '1000','--lr','2e-4','--optimizer','ADAM',
       '--scheduler','cosine','--scheduler_t_max','150','--eta_min','1e-6','--loss','1*L1',
       '--print_every','1' if smoke else '100','--save_per_image_metrics',
       '--experiment_root',str(directory.parent),'--save',directory.name]
    if args.cpu:c+=['--cpu']
    if smoke:c+=['--max_train_batches','1']
    if resume:c+=['--load',directory.name,'--resume',str(resume)]
    return c

def smoke(args,group):
    directory=group/'F4_smoke_x4_seed1'
    for epoch in (0,1):
        command=train_command(args,directory,True,epoch)
        cfg=config(args,command,smoke=True,resume_start_epoch=epoch,
            resume_checkpoint=str(directory/'model/model_1.pt') if epoch else None)
        cfg.update(data_range='1-1/801-801',stop_epoch=epoch+1,test_every=1,max_train_batches=1,
                   checkpoint_selection='engineering boundary only')
        launch(command,directory,cfg,ROOT/'LFMN',resume=bool(epoch))
    with (directory/'metrics.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    assert [int(r['epoch']) for r in rows]==[1,2]
    assert 'Restored RNG/DataLoader state at epoch 1' in (directory/'train_log.txt').read_text(encoding='utf-8')

def evaluate(args,group):
    directory=group/('F4_benchmarks_epoch20_OFF_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    ckpt=group/'F4_x4_seed1/model/model_20.pt'
    command=[sys.executable,'-u',str(ROOT/'repro/f4_train_entry.py'),'--dir_data',str(args.data_root),
        '--model','LFMNF4','--scale','4','--test_only','--pre_train',str(ckpt),
        '--data_test','+'.join(BENCHMARKS),'--n_threads',str(args.workers),'--ext','img',
        '--save_per_image_metrics','--experiment_root',str(group),'--save',directory.name]
    launch(command,directory,config(args,command,test_only=True,checkpoint=str(ckpt)),ROOT/'LFMN')
    write_json(group/'evaluation_paths.json',dict(F4=str(directory)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,default=ROOT/'experiment/all_runs')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cpu',action='store_true',help='engineering/smoke only')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--resume',type=Path)
    args=p.parse_args();args.data_root=args.data_root.resolve()
    if args.workers<1 or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single process, workers >=1 required')
    if args.cpu and not (args.check_only or args.smoke_only):raise ValueError('formal20e requires CUDA')
    check_data(args.data_root)
    digest=hashlib.sha256()
    folders=[args.data_root/'DIV2K'/name for name in (
        'DIV2K_train_HR','DIV2K_train_LR_bicubic/X4','DIV2K_valid_HR','DIV2K_valid_LR_bicubic/X4')]
    folders += [args.data_root/'benchmark'/name/sub for name in BENCHMARKS for sub in ('HR','LR_bicubic/X4')]
    for folder in folders:
        for image in sorted(folder.glob('*.png')):
            digest.update(str(image.relative_to(args.data_root)).replace('\\','/').encode())
            with image.open('rb') as stream:
                file_digest=hashlib.sha256()
                for block in iter(lambda:stream.read(1024*1024),b''):file_digest.update(block)
            digest.update(file_digest.digest())
    args.data_fingerprint=digest.hexdigest()
    subprocess.run([sys.executable,str(ROOT/'repro/check_training_run_logging.py')],cwd=ROOT,check=True)
    if args.resume:
        group=args.resume.resolve();manifest=json.loads((group/'protocol.json').read_text(encoding='utf-8'))
        if manifest!=dict(protocol=PROTOCOL,workers=args.workers,data_root=str(args.data_root),
                          data_fingerprint=args.data_fingerprint,source_sha256=hashes()):
            raise ValueError('resume protocol/data/workers/source changed')
    else:
        group=args.output_root.resolve()/('F4_x4_seed1_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        group.mkdir(parents=True,exist_ok=False)
        write_json(group/'protocol.json',dict(protocol=PROTOCOL,workers=args.workers,
            data_root=str(args.data_root),data_fingerprint=args.data_fingerprint,source_sha256=hashes()))
    print('F4 GROUP:',group,flush=True)
    command=[sys.executable,'-u',str(ROOT/'repro/check_f4.py'),'--internal',
        '--device','cpu' if args.cpu else 'cuda','--data-root',str(args.data_root)]
    checkdir=group/('engineering_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    launch(command,checkdir,config(args,command,kind='engineering'),ROOT)
    if args.check_only:return
    if not args.resume:smoke(args,group)
    if args.smoke_only:return
    directory=group/'F4_x4_seed1';epoch=0
    if directory.exists():
        with (directory/'metrics.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        epoch=int(rows[-1]['epoch']) if rows else 0
        if not epoch:raise ValueError('failed before epoch1: preserve directory, start fresh')
    if epoch<20:
        command=train_command(args,directory,resume=epoch)
        launch(command,directory,config(args,command,resume_start_epoch=epoch,
            resume_checkpoint=str(directory/'model'/f'model_{epoch}.pt') if epoch else None),
            ROOT/'LFMN',resume=bool(epoch))
    elif epoch!=20:raise ValueError('epoch must not exceed20')
    evaluate(args,group)
    subprocess.run([sys.executable,str(ROOT/'repro/profile_f4_resources.py'),str(group)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'repro/summarize_f4_screen.py'),str(group)],cwd=ROOT,check=True)

if __name__=='__main__':main()
