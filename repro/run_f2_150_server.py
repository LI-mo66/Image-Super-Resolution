#!/usr/bin/env python3
"""B0/F2 paired independent scratch150; one GPU, original operations, full records."""
import argparse
from contextlib import contextmanager
import socket
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from f2_150_common import MODELS,BENCHMARKS,PROTOCOL,source_hashes,git,read_csv,write_csv,sha256_file,manifest,run_directory


def environment(cpu=False):
    import torch
    libs={}
    for package in ['numpy','Pillow','einops','opencv-python-headless','imageio','scipy','matplotlib','tqdm','scikit-image']:
        try:libs[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:libs[package]=None
    return dict(python=sys.version,torch_version=torch.__version__,cuda_version=torch.version.cuda,
                gpu_name='CPU' if cpu else torch.cuda.get_device_name(0),
                device='cpu' if cpu else 'cuda',world_size=1,precision='FP32',
                cudnn_version=torch.backends.cudnn.version(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision(),libraries=libs)


def prepare_data(root,smoke):
    from PIL import Image
    pairs=[]
    for split,indices in [('train',[1] if smoke else range(1,801)),('valid',[801] if smoke else range(801,901))]:
        for i in indices:
            name=str(i).zfill(4)
            pairs.append(('DIV2K_'+split,
                          root/'DIV2K'/('DIV2K_'+split+'_HR')/(name+'.png'),
                          root/'DIV2K'/('DIV2K_'+split+'_LR_bicubic')/'X4'/(name+'x4.png')))
    for name,count in ({'Set5':5} if smoke else BENCHMARKS).items():
        folder=root/'benchmark'/name;hr=sorted((folder/'HR').glob('*.png'))
        if len(hr)!=count:raise ValueError('{} expected {} HR PNG files, got {}'.format(name,count,len(hr)))
        pairs.extend((name,p,folder/'LR_bicubic'/'X4'/(p.stem+'x4.png')) for p in hr)
    records=[]
    for name,hr,lr in pairs:
        if not hr.is_file() or not lr.is_file():raise FileNotFoundError('Missing pair: {} / {}'.format(hr,lr))
        with Image.open(hr) as image:hs=image.size
        with Image.open(lr) as image:ls=image.size
        if any(h//4!=l for h,l in zip(hs,ls)):raise ValueError('x4 dimensions mismatch: '+str(hr))
        records.append(dict(dataset=name,hr=str(hr.relative_to(root)).replace('\\','/'),
                            lr=str(lr.relative_to(root)).replace('\\','/'),
                            hr_shape=list(hs),lr_shape=list(ls),
                            hr_sha256=sha256_file(hr),lr_sha256=sha256_file(lr)))
    digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    print('DATA PREFLIGHT: {} pairs, full content SHA256 {}'.format(len(records),digest),flush=True)
    return dict(data_root=str(root),image_pairs=len(records),fingerprint=digest,records=records)


def train_command(args,directory,label,smoke=False,resume=0,stop=150):
    command=[sys.executable,'-u',str(ROOT/'repro/f2_150_train_entry.py'),
             '--dir_data',str(args.data_root),'--model',MODELS[label],
             '--data_train','DIV2K','--data_test','DIV2K+Set5',
             '--data_range','1-1/801-801' if smoke else PROTOCOL['data_range'],
             '--scale','4','--patch_size','256','--batch_size','4','--n_threads',str(args.workers),
             '--ext','img','--epochs',str(stop),'--test_every','1' if smoke else '1000',
             '--lr','2e-4','--optimizer','ADAM','--scheduler','cosine','--scheduler_t_max','150',
             '--eta_min','1e-6','--loss','1*L1','--seed','1','--precision','single',
             '--print_every','1' if smoke else '100','--save_per_image_metrics',
             '--experiment_root',str(directory.parent),'--save',directory.name]
    if args.cpu:command+=['--cpu']
    if smoke:command+=['--max_train_batches','1']
    if resume:command+=['--load',directory.name,'--resume',str(resume)]
    return command


def config(args,group,label,command,**extra):
    m=manifest(group)
    save=command[command.index('--save')+1] if '--save' in command else label+'_engineering'
    result=dict(PROTOCOL,protocol=m['protocol'],commit=m['commit'],authorized_experiment='F2_B0_INDEPENDENT_COS150',
                run_id=save,experiment_name=label,model=MODELS.get(label,'engineering_check'),
                model_source='LFMN/model/lfmnf2.py' if label=='F2' else 'LFMN/model/lfmn.py',
                start_time=datetime.datetime.now().astimezone().isoformat(),status='prepared',
                epochs=150,steps_per_epoch=1000,learning_rate=2e-4,
                dataset='DIV2K',train_range='1-800',validation_range='801-900',
                degradation='provided bicubic X4 LR',command=command,cwd=str(ROOT/'LFMN'),
                git_commit=m['commit'],git_dirty=bool(git('status','--porcelain')),
                git_diff_summary=git('diff','--stat'),source_sha256=m['source_sha256'],
                data_root=str(args.data_root),workers=args.workers,dataset_fingerprint=m['dataset_fingerprint'],
                environment=m['environment'],gpu_name=m['environment']['gpu_name'],
                torch_version=m['environment']['torch_version'],cuda_version=m['environment']['cuda_version'],
                device=m['environment']['device'],world_size=1,ema_enabled=False,self_ensemble_enabled=False,
                resume_checkpoint=None,resume_start_epoch=0,
                metric_protocol='original utility.py; quantize255; DIV2K RGB PSNR/crop10; benchmark PSNR Y BT601-256/crop4, MATLAB SSIM',
                checkpoint_selection='primary fixed150; secondary independent DIV2K best, exact tie first',
)
    result.update(extra)
    if extra.get('smoke'):
        stop=int(command[command.index('--epochs')+1])
        result.update(epochs=stop,stop_epoch=stop,steps_per_epoch=1,test_every=1,
                      data_range='1-1/801-801',train_range='1-1',validation_range='801-801',max_train_batches=1)
    if extra.get('resume_checkpoint'):
        result['input_checkpoint_sha256']=sha256_file(extra['resume_checkpoint'])
        old=json.loads((Path(extra['resume_checkpoint']).parents[1]/'config.json').read_text(encoding='utf-8'))
        result['run_id']=old['run_id'];result['start_time']=old['start_time']
    return result


def checks(args,group):
    scripts=[('logging','check_training_run_logging.py',[]),
             ('structure','check_f2_budget.py',['--device','cpu' if args.cpu else 'cuda','--output',str(group/'structure_check.json')]),
             ('summary','check_f2_150_summary.py',[]),
             ('observation','check_f2_150_observation.py',[])]
    for label,script,options in scripts:
        directory=group/(label+'_check_x4_seed1_'+group.name.rsplit('_',1)[-1])
        command=[sys.executable,'-u',str(ROOT/'repro'/script)]+options
        launch(command,directory,config(args,group,label,command,engineering_check=True),ROOT)


def paired_batches(group,runs,expected_epochs):
    b0=read_csv(runs['B0']/'first_batch_hashes.csv');f2=read_csv(runs['F2']/'first_batch_hashes.csv')
    if [int(r['epoch']) for r in b0]!=list(range(1,expected_epochs+1)) or [int(r['epoch']) for r in f2]!=list(range(1,expected_epochs+1)):
        raise ValueError('Incomplete paired first-batch timeline')
    for a,b in zip(b0,f2):
        if a!=b:raise ValueError('B0/F2 first LR/HR batch differs at epoch '+a['epoch'])
    return True


def evaluate(args,group,label,source,datasets,selection,epoch):
    import torch
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    directory=group/(label+'_'+selection+'_OFF_'+stamp)
    digest=sha256_file(source)
    command=[sys.executable,'-u',str(ROOT/'repro/f2_150_train_entry.py'),
             '--dir_data',str(args.data_root),'--model',MODELS[label],'--scale','4',
             '--test_only','--pre_train',str(source),'--data_test','+'.join(datasets),
             '--n_threads',str(args.workers),'--ext','img','--precision','single',
             '--save_per_image_metrics','--experiment_root',str(group),'--save',directory.name]
    if args.cpu:command+=['--cpu']
    launch(command,directory,config(args,group,label,command,test_only=True,checkpoint=str(source),
           checkpoint_sha256=digest,selected_epoch=epoch,epoch=epoch,selection=selection),ROOT/'LFMN')
    rows=torch.load(directory/'per_image_metrics/epoch_0000.pt',map_location='cpu',weights_only=True)
    write_csv(directory/'per_image_metrics.csv',rows)
    means=[]
    for name,count in datasets.items():
        selected=[r for r in rows if r['dataset']==name]
        if len(selected)!=count or len({r['filename'] for r in selected})!=count:raise ValueError('Image count mismatch for '+name)
        means.append(dict(dataset=name,epoch=epoch,selection=selection,checkpoint=str(source),checkpoint_sha256=digest,
                          psnr=sum(r['psnr'] for r in selected)/count,ssim=sum(r['ssim'] for r in selected)/count,
                          image_count=count,scale=4,self_ensemble=False))
    write_csv(directory/'benchmark_means.csv',means)
    return dict(directory=directory.name,epoch=epoch,checkpoint_sha256=digest),rows


def smoke(args,group):
    m=manifest(group);runs={}
    expected=None
    for label in MODELS:
        directory=group/m['smoke_runs'][label];runs[label]=directory
        for stop,resume in [(1,0),(2,1)]:
            command=train_command(args,directory,label,smoke=True,resume=resume,stop=stop)
            launch(command,directory,config(args,group,label,command,smoke=True,
                   expected_initial_unchanged_state_sha256=expected,
                   resume_checkpoint=str(directory/'model/model_1.pt') if resume else None,resume_start_epoch=resume),
                   ROOT/'LFMN',resume=bool(resume))
            saved=json.loads((directory/'config.json').read_text(encoding='utf-8'))
            expected=saved['initial_unchanged_state_sha256']
        if [int(r['epoch']) for r in read_csv(directory/'metrics.csv')]!=[1,2]:raise ValueError('Smoke resume timeline mismatch')
        if 'Restored RNG/DataLoader state at epoch 1' not in (directory/'train_log.txt').read_text(encoding='utf-8'):
            raise ValueError('Smoke RNG resume trace missing')
        before={r['filename']:r for r in read_csv(directory/'set5_per_image.csv') if r['epoch']=='2'}
        _,actual=evaluate(args,group,label,directory/'model/model_2.pt',{'Set5':5},'smoke_reload',2)
        for row in actual:
            if any(abs(row[k]-float(before[row['filename']][k]))>1e-6 for k in ('psnr','ssim')):
                raise ValueError('Strict reload changes Set5 metric')
    paired_batches(group,runs,2)
    write_json(group/'gate2.json',dict(passed=True,smoke_runs=m['smoke_runs'],resume_epochs=[1,2],
               shared_initialization_equal=True,first_batch_hashes_equal=True,strict_reload_metrics_equal=True))
    print('GATE2 PASSED: both B0/F2 real batch, independent validation, Set5, resume and matched inputs',flush=True)


@contextmanager
def group_lock(group):
    path=group/'.runner_lock.json'
    token=dict(pid=os.getpid(),host=socket.gethostname(),started=datetime.datetime.now().astimezone().isoformat())
    payload=json.dumps(token,sort_keys=True)
    try:
        with path.open('x',encoding='utf-8') as stream:stream.write(payload)
    except FileExistsError:
        raise RuntimeError('Run group is locked; check the recorded PID before recovering a stale lock: '+str(path))
    try:yield
    finally:
        if path.exists() and path.read_text(encoding='utf-8')==payload:path.unlink()


def current_epoch(directory):
    if not directory.exists():return 0
    path=directory/'metrics.csv'
    rows=read_csv(path) if path.exists() else []
    if not rows:raise ValueError('Failed before first complete checkpoint; preserve and start new group')
    epochs=[int(r['epoch']) for r in rows]
    if epochs!=list(range(1,epochs[-1]+1)):raise ValueError('Broken metrics epoch timeline')
    if epochs[-1]>150:raise ValueError('Unregistered epoch exceeds150')
    return epochs[-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/Image-Super-Resolution/datasets'))
    parser.add_argument('--output-root',type=Path,default=ROOT/'experiment/all_runs')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cpu',action='store_true',help='engineering checks/smoke only')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--skip-resource-profile',action='store_true',help='record resources unknown, PSNR comparison still runs')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--resume',type=Path);modes.add_argument('--eval-only',type=Path)
    args=parser.parse_args();args.data_root=args.data_root.resolve()
    if args.workers!=4:raise ValueError('Registered independent150e protocol fixes workers=4')
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or int(os.environ.get('RANK','0'))!=0:
        raise ValueError('Registered150e package supports one process, one GPU only')
    engineering=args.check_only or args.smoke_only
    if args.cpu and not engineering:raise ValueError('Formal150e requires CUDA')
    if (args.resume or args.eval_only) and engineering:raise ValueError('Engineering runs cannot resume formal groups')
    if not engineering and git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Use the pinned clean code commit; do not edit source on server')
    env=environment(args.cpu);old=args.resume or args.eval_only
    if old:
        group=old.resolve();m=manifest(group)
        if (m['engineering_only'] or m['protocol']!=PROTOCOL or m['commit']!=git('rev-parse','HEAD') or
            m['source_sha256']!=source_hashes() or m['data_root']!=str(args.data_root) or m['workers']!=args.workers or m['environment']!=env):
            raise ValueError('Resume/evaluation commit, protocol, source, data path, workers or environment changed')
        data=prepare_data(args.data_root,False)
        if m['dataset_fingerprint']!=data['fingerprint']:raise ValueError('Dataset content changed')
    else:
        data=prepare_data(args.data_root,engineering)
        stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        group=args.output_root.resolve()/('F2_pair150_x4_seed1_'+stamp)
        group.mkdir(parents=True,exist_ok=False)
        m=dict(protocol=PROTOCOL,commit=git('rev-parse','HEAD'),source_sha256=source_hashes(),
               data_root=str(args.data_root),workers=args.workers,environment=env,
               dataset_fingerprint=data['fingerprint'],engineering_only=bool(engineering),
               runs={k:k+'_train_x4_seed1_'+stamp for k in MODELS},
               smoke_runs={k:k+'_smoke_x4_seed1_'+stamp for k in MODELS})
        write_json(group/'protocol.json',m);write_json(group/'dataset_manifest.json',data)
        print('RUN GROUP: '+str(group),flush=True)
    with group_lock(group):
        if not old:
            checks(args,group)
            if args.check_only:return
            smoke(args,group)
            if args.smoke_only:return
        if not (group/'gate2.json').is_file() or not json.loads((group/'gate2.json').read_text())['passed']:
            raise ValueError('Verified paired Gate2 evidence missing')
        if not args.eval_only:
            expected=None
            for label in MODELS:
                directory=run_directory(group,label);epoch=current_epoch(directory)
                if label=='F2':
                    b0=json.loads((run_directory(group,'B0')/'config.json').read_text(encoding='utf-8'))
                    expected=b0['initial_unchanged_state_sha256']
                if epoch<150:
                    command=train_command(args,directory,label,resume=epoch)
                    launch(command,directory,config(args,group,label,command,
                           expected_initial_unchanged_state_sha256=expected,
                           resume_checkpoint=str(directory/'model'/('model_{}.pt'.format(epoch))) if epoch else None,
                           resume_start_epoch=epoch),ROOT/'LFMN',resume=bool(epoch))
        runs={k:run_directory(group,k) for k in MODELS}
        paired_batches(group,runs,150)
        evaluations={}
        for label,directory in runs.items():
            valid=read_csv(directory/'validation_per_checkpoint.csv')
            if [int(r['epoch']) for r in valid]!=list(range(1,151)):raise ValueError('Incomplete independent validation timeline')
            best=int(max(valid,key=lambda r:float(r['validation_psnr']))['epoch'])
            final,_=evaluate(args,group,label,directory/'model/model_150.pt',BENCHMARKS,'final150',150)
            if best==150:secondary=dict(final)
            else:secondary,_=evaluate(args,group,label,directory/'model'/('model_{}.pt'.format(best)),BENCHMARKS,'best_div2k',best)
            evaluations[label]={'final150':final,'best_div2k':secondary}
            write_json(group/'evaluation_paths.json',evaluations)
        if not args.skip_resource_profile:
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            command=[sys.executable,'-u',str(ROOT/'repro/profile_f2_150_resources.py'),str(group)]
            try:launch(command,group/('resource_check_x4_seed1_'+stamp),config(args,group,'resources',command,test_only=True),ROOT)
            except subprocess.CalledProcessError as error:
                write_json(group/'resource_status.json',dict(status='failed',exit_code=error.returncode,
                           note='PSNR artifacts preserved; efficiency unknown, see managed resource log'))
                print('Resource measurement failed; preserving PSNR results and reporting resource unknown',flush=True)
        else:write_json(group/'resource_status.json',dict(status='skipped',note='Efficiency not measured'))
        subprocess.run([sys.executable,str(ROOT/'repro/summarize_f2_150.py'),str(group)],cwd=ROOT,check=True)
        print('COMPLETED B0/F2 independent150. No automatic1000e, extra seeds or shutdown.',flush=True)


if __name__=='__main__':main()
