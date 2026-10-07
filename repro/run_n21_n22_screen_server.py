"""One dedicated GPU queue: N21 -> N22 -> new B0. Never resumes implicitly."""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
# The queue controller should not own a CUDA context. In containers NVML may
# expose host PIDs rather than namespace PIDs; GPU metadata is queried via CLI.
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from model.lfmn import Net as Baseline
from summarize_n21_n22_screen import summarize

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()

def write(path,payload):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def environment():
    modules=('numpy','einops','PIL','imageio','skimage','cv2','matplotlib','tqdm')
    gpu=hardware()
    return {'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':gpu['name'],'driver':gpu['driver'],
            'python':sys.version,'executable':sys.executable,
            'libraries':{name:str(getattr(importlib.import_module(name),'__version__','unknown')) for name in modules}}

def hardware():
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=name,memory.total,driver_version',
                                 '--format=csv,noheader,nounits'],text=True).strip()
    name,memory,driver=[part.strip() for part in raw.split(',')]
    return {'name':name,'total_mib':int(memory),'driver':driver}

def code_hashes():
    paths=sorted(list((ROOT/'LFMN').rglob('*.py'))+list((ROOT/'repro').glob('*n21*.py'))
                 +list((ROOT/'repro').glob('*n22*.py'))+[ROOT/'repro/screen_train_entry.py'])
    return {str(p.relative_to(ROOT)):sha(p) for p in paths if '__pycache__' not in p.parts}

def data_hashes(data,protocol):
    records={}
    for prefix,ids in [('train',protocol['training_ids']),('valid',protocol['validation_ids'])]:
        for index in ids:
            for relative in [f'DIV2K/DIV2K_{prefix}_HR/{index:04d}.png',
                             f'DIV2K/DIV2K_{prefix}_LR_bicubic/X4/{index:04d}x4.png']:
                path=data/relative
                if not path.is_file():
                    raise FileNotFoundError(f'Missing data: {path}')
                records[relative]=sha(path)
    return records

def other_gpu_pids():
    response=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(s.strip() and not s.strip().isdigit() for s in response.splitlines()):
        raise RuntimeError('Cannot establish whether other GPU jobs exist')
    return sorted({int(s.strip()) for s in response.splitlines() if s.strip().isdigit()}-{os.getpid()})

def command(args,protocol,group):
    model={'n21':'lfmn_n21','n22':'lfmn_n22','b0':'LFMN'}[group]
    cmd=[sys.executable,str(ROOT/'repro/screen_train_entry.py'), '--model',model,
         '--dir_data',str(args.data_root),'--data_train','DIV2K','--data_test','DIV2K',
         '--data_range',protocol['expected_config']['data_range'],'--scale','4',
         '--patch_size','256','--batch_size','4','--n_threads',str(protocol['threads']),
         '--ext','img','--epochs',str(protocol['epochs']),'--test_every','1000',
         '--lr','2e-4','--scheduler','cosine','--scheduler_t_max','150','--eta_min','1e-6',
         '--optimizer','ADAM','--epsilon','1e-8','--weight_decay','0','--loss','1*L1',
         '--seed','1','--precision','single','--print_every','100',
         '--save_models','--save_per_image_metrics','--save',str(args.output/group)]
    if protocol['smoke']:
        cmd+=['--max_train_batches','1']
    return cmd

def run(cmd,cwd,log):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(cmd,cwd=cwd,env={**os.environ,'PYTHONUNBUFFERED':'1'},
                                 stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            stream.write(line)
            stream.flush()
        rc=process.wait()
        os.fsync(stream.fileno())
    if rc:
        raise subprocess.CalledProcessError(rc,cmd)

def protocol(smoke):
    threads=0 if smoke else min(4,max(1,(os.cpu_count() or 1)//2))
    ranges='1-2/859-859' if smoke else '1-800/801-900'
    return {'smoke':smoke,'epochs':1 if smoke else 40,'updates_per_epoch':1 if smoke else 1000,
        'threads':threads,'training_ids':[1,2] if smoke else list(range(1,801)),
        'validation_ids':[859] if smoke else list(range(801,901)),
        'scheduler_horizon':150,'baseline_source':'b198de7','baseline_reuse':False,
        'order':['n21','n22','b0'],'initialization':'scratch, common baseline weights and RNG verified',
        'metric':'Full DIV2K images: quantized RGB PSNR border10; quantized Y SSIM border4; no x8/chop',
        'expected_config':{'data_range':ranges,'scale':'[4]','patch_size':256,'batch_size':4,
            'epochs':1 if smoke else 40,'test_every':1000,'seed':1,'ext':'img','rgb_range':255,
            'n_threads':threads,'lr':0.0002,'scheduler':'cosine','scheduler_t_max':150,'eta_min':1e-6,
            'optimizer':'ADAM','betas':'(0.9, 0.999)','weight_decay':0.0,'epsilon':1e-8,
            'loss':'1*L1','precision':'single','pre_train':'','load':'','resume':0,
            'no_augment':'False','self_ensemble':'False','chop':'False',
            'max_train_batches':1 if smoke else 0,'rgcrd_mode':'off'}}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--prepared',action='store_true')
    ap.add_argument('--smoke-test',action='store_true',help='Local integration only: 1 update/group, never shuts down')
    ap.add_argument('--shutdown-on-success',action='store_true')
    ap.add_argument('--dedicated-instance',action='store_true',help='Confirm no other work needs this instance')
    args=ap.parse_args()
    args.data_root=args.data_root.resolve()
    args.output=args.output.resolve()
    if args.prepare_only and args.prepared:
        ap.error('prepare-only and prepared conflict')
    if args.shutdown_on_success and (args.smoke_test or not args.dedicated_instance or sys.platform!='linux'):
        ap.error('Shutdown requires Linux, a non-smoke run, and --dedicated-instance')
    if args.shutdown_on_success and (not Path('/usr/bin/shutdown').is_file() or not os.access('/usr/bin/shutdown',os.X_OK)):
        raise RuntimeError('Executable AutoDL shutdown helper missing; queue not started')
    if args.data_root==args.output or args.data_root in args.output.parents or args.output in args.data_root.parents:
        ap.error('Data and results must be disjoint')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    if os.environ.get('CUDA_VISIBLE_DEVICES','0')!='0':
        raise RuntimeError('This framework requires GPU0')
    if not args.smoke_test:
        if git('status','--porcelain','--untracked-files=normal'):
            raise RuntimeError('Tracked source is dirty')
        if other_gpu_pids():
            raise RuntimeError('Other GPU jobs detected; use a dedicated instance')
        if hardware()['total_mib']<16*1024:
            raise RuntimeError('Server handoff requires >=16GiB VRAM; recommend RTX4090 24GB')
        if shutil.disk_usage(args.output.parent if args.output.parent.exists() else ROOT).free<5*1024**3:
            raise RuntimeError('Need >=5GiB free result storage (datasets not included)')
    p=protocol(args.smoke_test)
    env=environment()
    commit=git('rev-parse','HEAD')
    hashes=code_hashes()
    print('Auditing data paths and hashes',flush=True)
    data=data_hashes(args.data_root,p)
    manifest_path=args.output/'manifest.json'
    if args.prepared:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for key,value in [('commit',commit),('source_hashes',hashes),('data_hashes',data),('protocol',p),
                          ('data_root',str(args.data_root)),('output',str(args.output)),('environment',env)]:
            if manifest.get(key)!=value:
                raise ValueError(f'Prepared provenance changed: {key}')
        if manifest['status']!='PREPARED':
            raise ValueError('Not a prepared, unstarted output')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        manifest={'commit':commit,'source_hashes':hashes,'data_hashes':data,'protocol':p,
            'data_root':str(args.data_root),'output':str(args.output),'status':'PREPARING',
            'started_utc':now(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':env['gpu'],'environment':env,'group_exit_codes':{}}
        write(manifest_path,manifest)
        # Local checker weights generated here: no historical tar/checkpoint required.
        torch.manual_seed(1)
        init=args.output/'checks_initial_b0.pt'
        torch.save(Baseline(scale=4).state_dict(),init)
        for group in ('n21','n22'):
            run([sys.executable,str(ROOT/f'repro/check_{group}.py'),'--checkpoint',str(init),
                 '--data',str(args.data_root/'DIV2K'),'--output',str(args.output/f'check_{group}')],
                ROOT,args.output/f'{group}_preflight.log')
        manifest['status']='PREPARED'
        manifest['prepared_utc']=now()
        write(manifest_path,manifest)
    if args.prepare_only:
        print('PREPARED: no screening training or shutdown started',flush=True)
        return
    try:
        manifest['status']='RUNNING'
        manifest['shutdown_requested']=args.shutdown_on_success
        write(manifest_path,manifest)
        for group in p['order']:
            if (args.output/group).exists():
                raise FileExistsError(f'{group} exists; refusing overwrite/resume')
            started=time.monotonic()
            run(command(args,p,group),ROOT/'LFMN',args.output/f'{group}_console.log')
            manifest['group_exit_codes'][group]=0
            manifest.setdefault('group_wall_seconds',{})[group]=time.monotonic()-started
            write(manifest_path,manifest)
            if hashes!=code_hashes():
                raise RuntimeError('Source changed during queue')
        if data!=data_hashes(args.data_root,p):
            raise RuntimeError('Data changed during queue')
        result=summarize(args.output)
        print(json.dumps(result['comparisons'],indent=2),flush=True)
        manifest['status']='COMPLETE'
        manifest['completed_utc']=now()
        write(manifest_path,manifest)
        if args.shutdown_on_success:
            # AutoDL container power-off helper; deliberately no generic systemd flags.
            if other_gpu_pids():
                manifest['status']='COMPLETE_SHUTDOWN_BLOCKED_OTHER_GPU_JOB'
                write(manifest_path,manifest)
                raise RuntimeError('Results complete; other GPU jobs prevent shutdown')
            if not Path('/usr/bin/shutdown').is_file():
                raise RuntimeError('AutoDL shutdown helper missing')
            manifest['status']='COMPLETE_SHUTDOWN_REQUESTED'
            write(manifest_path,manifest)
            os.sync()
            print('All three groups and integrity checks complete. Requesting AutoDL shutdown.',flush=True)
            subprocess.run(['/usr/bin/shutdown'],check=True)
    except BaseException as error:
        if manifest['status']=='RUNNING':
            manifest['status']='FAILED_NO_SHUTDOWN'
        elif manifest['status']=='COMPLETE_SHUTDOWN_REQUESTED':
            manifest['status']='COMPLETE_SHUTDOWN_FAILED'
        manifest['error']=repr(error)
        manifest['ended_utc']=now()
        write(manifest_path,manifest)
        raise

if __name__=='__main__':
    main()
