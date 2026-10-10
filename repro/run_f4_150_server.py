"""Continue F4 epoch20 to total150 without modifying the original run."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from run_f4_screen_server import PROTOCOL,config,hashes,train_command as original_command,check_data,BENCHMARKS
from backfill_f4_set5 import dataset_fingerprint,digest,read_evaluation,aggregate
from f4_150_monitor import ensure_set5,update_table,preserve_rng

PINNED_PARENT_SHA='cafb8f9696c7799c117b8571fec274efb2eed165b54985af5d6db092b6427c08'


def safe_state(path):
    import torch
    import numpy as np
    from numpy._core.multiarray import _reconstruct
    with torch.serialization.safe_globals([_reconstruct,np.ndarray,np.dtype,type(np.dtype('uint32'))]):
        return torch.load(path,map_location='cpu',weights_only=True)


def snapshot(directory):
    result={}
    for p in sorted(Path(directory).rglob('*')):
        if p.is_symlink():raise ValueError('source run symlink not allowed')
        if p.is_file():result[str(p.relative_to(directory))]=digest(p)
    return result


def assert_parent_unchanged(group):
    m=json.loads((Path(group)/'continuation_manifest.json').read_text(encoding='utf-8'))
    if snapshot(Path(m['parent_group'])/'F4_x4_seed1')!=m['parent_snapshot']:
        raise ValueError('original parent artifacts changed; preserve all outputs')


def check_boundary(directory,epoch,steps_per_epoch=1000):
    import torch
    directory=Path(directory)
    for name in [f'model/model_{epoch}.pt','optimizer.pt','scheduler.pt','resume_state.pt',
                 'loss.pt','loss_log.pt','psnr_log.pt','ssim_log.pt','metrics.csv']:
        if not (directory/name).is_file():raise FileNotFoundError(directory/name)
    state=safe_state(directory/'resume_state.pt')
    optimizer=safe_state(directory/'optimizer.pt');scheduler=safe_state(directory/'scheduler.pt')
    if state['epoch']!=epoch or scheduler['last_epoch']!=epoch:
        raise ValueError('epoch/optimizer/scheduler/RNG boundary inconsistent; no guessed rollback')
    if scheduler['T_max']!=150 or scheduler['eta_min']!=1e-6 or scheduler['base_lrs']!=[2e-4]:
        raise ValueError('Cosine timeline must stay T_max150')
    expected=1e-6+(2e-4-1e-6)*(1+math.cos(math.pi*epoch/150))/2
    if any(abs(g['lr']-expected)>1e-12 for g in optimizer['param_groups']):
        raise ValueError('saved LR differs from original Cosine timeline')
    if {int(v['step']) for v in optimizer['state'].values()}!={epoch*steps_per_epoch}:
        raise ValueError('Adam update count differs from checkpoint epoch')
    for name in ['torch','cuda','loader','numpy','python','error_last']:
        if name not in state:raise ValueError('missing RNG/loader state: '+name)
    if safe_state(directory/'psnr_log.pt').shape!=(epoch,1,1):
        raise ValueError('preserve DIV2K-only history shape')
    if safe_state(directory/'ssim_log.pt').shape!=(epoch,1,1):raise ValueError('SSIM history shape')
    with (directory/'metrics.csv').open(encoding='utf-8') as f:curve=list(csv.DictReader(f))
    if [int(r['epoch']) for r in curve]!=list(range(1,epoch+1)):
        raise ValueError('metrics boundary differs from saved state')
    from model.lfmnf4 import Net
    with torch.random.fork_rng(devices=[]):
        n=Net(scale=4);weights=safe_state(directory/'model'/f'model_{epoch}.pt')
        n.load_state_dict(weights,strict=True)
        if not all(torch.isfinite(v).all() for v in weights.values() if v.is_floating_point()):
            raise FloatingPointError('checkpoint contains nonfinite weights')
    return curve


def training_command(args,directory,epoch,target=150,smoke=False):
    c=original_command(args,directory,smoke=smoke,resume=epoch)
    c[2]=str(ROOT/'repro/f4_150_train_entry.py')
    c[c.index('--epochs')+1]=str(target)
    return c


def continuation_config(args,c,parent,epoch,target=150):
    cfg=config(args,c,resume_start_epoch=epoch,resume_checkpoint=str(Path(c[c.index('--experiment_root')+1])/
        c[c.index('--save')+1]/'model'/f'model_{epoch}.pt'),
        training_mode='CONTINUATION',parent_run=str(parent),kind='F4_CONTINUE150',
        stop_epoch=target,checkpoint_selection=f'fixed epoch{target}',
        original_scratch_start=True,Set5_monitor='separate eval-only subprocess; never checkpoint selection',
        baseline_150_comparison='PENDING_NO_B0_TRAINING')
    return cfg


def preflight(args,parent):
    import torch
    source=Path(parent)/'F4_x4_seed1'
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    if cfg.get('status')!='completed' or cfg.get('model')!='LFMNF4':raise ValueError('parent not completed F4')
    for k,v in PROTOCOL.items():
        if cfg.get(k)!=v:raise ValueError('original protocol differs: '+k)
    if digest(source/'model/model_20.pt')!=PINNED_PARENT_SHA:raise ValueError('wrong parent epoch20 checkpoint')
    if Path(cfg['data_root']).resolve()!=args.data_root or cfg['workers']!=args.workers:
        raise ValueError('keep original data-root and workers')
    for name,h in cfg['source_sha256'].items():
        if name.startswith('LFMN/') or name=='repro/f4_train_entry.py':
            if digest(ROOT/name)!=h:raise ValueError('original training source changed: '+name)
    if not torch.cuda.is_available():raise ValueError('formal continuation requires CUDA')
    for k,actual in [('torch_version',str(torch.__version__)),('cuda_version',torch.version.cuda),
                     ('gpu_name',torch.cuda.get_device_name(0))]:
        if cfg[k]!=actual:raise ValueError('keep original runtime: '+k)
    if cfg['data_fingerprint']!=args.data_fingerprint:raise ValueError('dataset bytes changed')
    check_boundary(source,20)
    for e in range(1,21):
        if not (source/'model'/f'model_{e}.pt').is_file():raise FileNotFoundError('parent checkpoint '+str(e))
        aggregate(safe_state(source/'per_image_metrics'/f'epoch_{e:04d}.pt'),'DIV2K',100)
    return cfg


def final_evaluate(args,group):
    existing=group/'final_evaluation.json'
    if existing.exists():
        info=json.loads(existing.read_text(encoding='utf-8'))
        checkpoint=group/'F4_x4_seed1/model/model_150.pt'
        rows=read_evaluation(info['directory'],checkpoint)
        if info['checkpoint_sha256']!=digest(checkpoint):raise ValueError('existing final hash differs')
        for name,count in BENCHMARKS.items():aggregate(rows,name,count)
        return
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out=group/('benchmarks_epoch150_OFF_'+stamp)
    checkpoint=group/'F4_x4_seed1/model/model_150.pt'
    from backfill_f4_set5 import command
    c=command(args,checkpoint,out,list(BENCHMARKS))
    launch(c,out,config(args,c,test_only=True,checkpoint=str(checkpoint),stop_epoch=150,
        checkpoint_selection='fixed epoch150',kind='F4_150_FINAL_OFF'),ROOT/'LFMN')
    rows=read_evaluation(out,checkpoint)
    for name,count in BENCHMARKS.items():aggregate(rows,name,count)
    write_json(group/'final_evaluation.json',dict(directory=str(out),checkpoint_sha256=digest(checkpoint)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--parent-group',type=Path)
    mode.add_argument('--resume-run',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output-root',type=Path,default=ROOT/'experiment/all_runs')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();args.cpu=False;args.data_root=args.data_root.resolve()
    if args.workers<1:raise ValueError('workers >=1 required')
    import os
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('one GPU/process only')
    subprocess.run([sys.executable,str(ROOT/'repro/check_training_run_logging.py')],cwd=ROOT,check=True)
    check_data(args.data_root)
    print('Checking complete dataset fingerprint...',flush=True)
    args.data_fingerprint=dataset_fingerprint(args.data_root)
    if args.resume_run:
        group=args.resume_run.resolve()
        m=json.loads((group/'continuation_manifest.json').read_text(encoding='utf-8'))
        parent=Path(m['parent_group'])
        preflight(args,parent)
        if m['continuation_source_sha256']!=hashes():raise ValueError('continuation source changed')
        if m['workers']!=args.workers or m['data_fingerprint']!=args.data_fingerprint:
            raise ValueError('continuation runtime/data changed')
        directory=group/'F4_x4_seed1'
        with (directory/'metrics.csv').open(encoding='utf-8') as f:curve=list(csv.DictReader(f))
        epoch=int(curve[-1]['epoch']);check_boundary(directory,epoch)
        if not 20<=epoch<=150:raise ValueError('continuation must stay epochs20-150')
        assert_parent_unchanged(group)
    else:
        parent=args.parent_group.resolve();preflight(args,parent)
        if args.check_only:
            print('PREFLIGHT PASS: valid epoch20 full restoration; no training started',flush=True);return
        group=args.output_root.resolve()/('F4_continue150_x4_seed1_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        group.mkdir(parents=True,exist_ok=False)
        before=snapshot(parent/'F4_x4_seed1')
        directory=group/'F4_x4_seed1'
        shutil.copytree(parent/'F4_x4_seed1',directory)
        if snapshot(directory)!=before:raise ValueError('copied restoration assets differ')
        write_json(group/'continuation_manifest.json',dict(parent_group=str(parent),start_epoch=20,
            parent_snapshot=before,parent_checkpoint_sha256=PINNED_PARENT_SHA,
            continuation_source_sha256=hashes(),workers=args.workers,data_fingerprint=args.data_fingerprint,
            target_epoch=150,scheduler_t_max=150,budget='additional130; total150; no B0 training'))
        epoch=20
    print('F4 150 GROUP:',group,flush=True)
    pointer=ROOT/'experiment/nohup_launcher/F4_150_latest_group.txt'
    pointer.parent.mkdir(parents=True,exist_ok=True)
    pointer.write_text(str(group)+'\n',encoding='utf-8')
    if args.check_only:return
    if (group/'summary150.json').exists():
        from summarize_f4_150 import validate_completion
        validate_completion(group)
        print((group/'summary150.txt').read_text(encoding='utf-8'),flush=True);return
    with preserve_rng():
        for e in range(1,epoch+1):ensure_set5(group,e,args)
        update_table(group)
    if epoch<150:
        c=training_command(args,directory,epoch)
        launch(c,directory,continuation_config(args,c,parent,epoch),ROOT/'LFMN',resume=True)
    check_boundary(directory,150)
    assert_parent_unchanged(group)
    final_evaluate(args,group)
    subprocess.run([sys.executable,str(ROOT/'repro/profile_f4_150_resources.py'),str(group)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'repro/summarize_f4_150.py'),str(group)],cwd=ROOT,check=True)


if __name__=='__main__':main()
