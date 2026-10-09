"""Read-only B0/F3 artifact audit, frozen diagnostics and Set5 backfill. No training."""
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
sys.path.insert(0,str(ROOT/'repro'))
BENCHMARKS={'Set5':5,'Set14':14,'B100':100,'Urban100':100,'Manga109':109}
F3_SHA='b5ca4291221406a1050e5b8595db8e921945490eb67ed0819162636fb478b986'
B0_SHA='4dc41417fe4c5224d4943dd9beb3f506f4f0a0b5c17de595c1e0db235829cd94'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def locate(root,label,expected,explicit=None):
    candidates=[explicit/'config.json'] if explicit else sorted(root.rglob(label+'_x4_seed1/config.json'))
    found=[]
    for path in candidates:
        checkpoint=path.parent/'model/model_20.pt'
        if checkpoint.is_file() and sha(checkpoint)==expected:found.append(path.parent.resolve())
    if not found:raise FileNotFoundError(label+' epoch20 SHA not found; specify --'+label.lower()+'-run')
    print(label+' matching copies: '+str(len(found)),flush=True)
    return found[-1]

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))

def compare_rows(b0,f3):
    import numpy as np
    a={(r['dataset'],r['filename']):r for r in b0}
    b={(r['dataset'],r['filename']):r for r in f3}
    if len(a)!=len(b0) or len(b)!=len(f3) or set(a)!=set(b):raise ValueError('pairing/duplicate filenames')
    rng=np.random.default_rng(1);report={};paired=[]
    for dataset in sorted({k[0] for k in a}):
        keys=sorted(k for k in a if k[0]==dataset)
        delta=np.array([float(b[k]['psnr'])-float(a[k]['psnr']) for k in keys])
        for k,value in zip(keys,delta):paired.append(dict(dataset=k[0],filename=k[1],
            b0_psnr=float(a[k]['psnr']),f3_psnr=float(b[k]['psnr']),delta_psnr=float(value),
            delta_ssim=float(b[k]['ssim'])-float(a[k]['ssim'])))
        if not np.isfinite(delta).all():raise ValueError('nonfinite PSNR')
        report[dataset]=dict(n=len(keys),mean=float(delta.mean()),median=float(np.median(delta)),
            positive=int((delta>0).sum()),ci95=np.quantile(
                delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[.025,.975]).tolist(),
            worst=keys[int(delta.argmin())][1],best=keys[int(delta.argmax())][1])
    return report,paired

def benchmark_rows(run,label):
    import torch
    group=run.parent;paths=group/'evaluation_paths.json'
    candidates=[]
    if paths.is_file() and label in load_json(paths):candidates.append(Path(load_json(paths)[label]))
    candidates+=sorted(group.glob(label+'_benchmarks_epoch20_OFF*'),reverse=True)
    for directory in candidates:
        file=directory/'per_image_metrics/epoch_0000.pt'
        if file.is_file():
            rows=torch.load(file,map_location='cpu',weights_only=True)
            for name,count in BENCHMARKS.items():
                selected=[r for r in rows if r['dataset']==name and r['scale']==4]
                if len(selected)!=count or len({r['filename'] for r in selected})!=count:
                    raise ValueError('benchmark incomplete '+name)
            return rows
    raise FileNotFoundError(label+' benchmark outputs missing')

def models(b0,f3,device):
    import torch
    from model.lfmn import Net as B0
    from model.lfmnf3 import Net as F3
    result=[]
    for cls,path in [(B0,b0),(F3,f3)]:
        net=cls(scale=4).to(device).eval()
        net.load_state_dict(torch.load(path/'model/model_20.pt',map_location=device,weights_only=True),strict=True)
        result.append(net)
    return result

def image(path,device):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as im:array=np.array(im.convert('RGB')).copy()
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().to(device)

def score(sr,hr,benchmark=False):
    from types import SimpleNamespace
    import utility
    dataset=SimpleNamespace(dataset=SimpleNamespace(benchmark=True)) if benchmark else None
    return utility.calc_psnr(utility.quantize(sr,255),hr,4,255,dataset)

def backfill_set5(run,net,data_root,output,epochs=range(1,21)):
    import torch
    import utility
    rows=[]
    folder=data_root/'benchmark/Set5'
    hr_files=sorted((folder/'HR').glob('*.png'))
    if len(hr_files)!=5:raise ValueError('Set5 requires five images')
    with torch.inference_mode():
        for epoch in epochs:
            checkpoint=run/'model'/f'model_{epoch}.pt'
            net.load_state_dict(torch.load(checkpoint,map_location=next(net.parameters()).device,weights_only=True),strict=True)
            per=[]
            for hp in hr_files:
                device=next(net.parameters()).device
                x=image(folder/'LR_bicubic/X4'/(hp.stem+'x4.png'),device);hr=image(hp,device)
                sr=utility.quantize(net(x),255)
                per.append(dict(epoch=epoch,filename=hp.stem,psnr=score(sr,hr,True),
                    ssim=utility.calc_ssim(sr,hr,4,255)))
            rows+=per
            print('F3_SET5 epoch={} PSNR={:.9f} SSIM={:.9f} OFF checkpoint_SHA={}'.format(epoch,
                sum(r['psnr'] for r in per)/5,sum(r['ssim'] for r in per)/5,sha(checkpoint)),flush=True)
    write_csv(output/'f3_set5_per_epoch_per_image.csv',rows)
    return [dict(epoch=e,psnr=sum(r['psnr'] for r in rows if r['epoch']==e)/5,
                 ssim=sum(r['ssim'] for r in rows if r['epoch']==e)/5) for e in epochs]

def frozen_diagnosis(b0,f3,data_root):
    import torch
    from torch.nn import functional as F
    samples=[]
    before={k:v.detach().cpu().clone() for k,v in f3.state_dict().items()}
    device=next(f3.parameters()).device
    with torch.inference_mode():
        for index in [801,834,867,900]:
            base=data_root/'DIV2K';x=image(base/'DIV2K_valid_LR_bicubic/X4'/f'{index:04d}x4.png',device)
            hr=image(base/'DIV2K_valid_HR'/f'{index:04d}.png',device)
            top=(x.shape[-2]-64)//2;left=(x.shape[-1]-64)//2
            x=x[:,:,top:top+64,left:left+64];hr=hr[:,:,top*4:(top+64)*4,left*4:(left+64)*4]
            stages=[];handles=[]
            def hook(i):
                def capture(module,inputs,outputs):
                    fs,current=inputs
                    u=F.leaky_relu(module.dw(F.leaky_relu(module.reduce(fs),.1)),.1)
                    z=F.layer_norm(current.permute(0,2,3,1),(48,),eps=1e-5)
                    correction=module.current_correction(z.permute(0,3,1,2).contiguous())
                    beta,gamma=outputs
                    rms=lambda t:float(t.square().mean().sqrt())
                    stages.append(dict(stage=i+1,u_rms=rms(u),correction_rms=rms(correction),
                        correction_ratio=rms(correction)/max(rms(u),1e-12),beta_mean=float(beta.mean()),
                        beta_saturation=float(((beta<.01)|(beta>.99)).float().mean()),gamma_rms=rms(gamma)))
                return capture
            for i,module in enumerate(f3.sfmls):handles.append(module.register_forward_hook(hook(i)))
            try:on=score(f3(x),hr)
            finally:
                for h in handles:h.remove()
            bypass=[]
            for module in f3.sfmls:
                module.enabled=False
                try:bypass.append(score(f3(x),hr)-on)
                finally:module.enabled=True
            for module in f3.sfmls:module.enabled=False
            try:all_off=score(f3(x),hr)-on
            finally:
                for module in f3.sfmls:module.enabled=True
            samples.append(dict(image=f'{index:04d}',crop_LR=[top,left,64,64],
                f3_on=on,b0=score(b0(x),hr),all_correction_OFF_minus_ON=all_off,
                per_stage_OFF_minus_ON=bypass,stages=stages))
            print('FROZEN_DIAGNOSTIC',json.dumps({k:v for k,v in samples[-1].items() if k!='stages'}),flush=True)
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in f3.state_dict().items())
    return dict(scope='exploratory fixed 4 center crops; quantized RGB crop10; not full validation or architecture gains',
                state_unchanged=True,samples=samples)

def profile(nets):
    import torch
    import statistics
    import time
    from torch.utils.flop_counter import FlopCounterMode
    if next(nets[0].parameters()).device.type!='cuda':return dict(status='SKIPPED_CPU')
    result={}
    for h,w in [(64,64),(128,128),(96,160)]:
        for label,net in zip(['B0','F3'],nets):
            torch.cuda.empty_cache();x=torch.zeros(1,3,h,w,device='cuda');times=[]
            with torch.inference_mode():
                for _ in range(10):net(x)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                for _ in range(30):
                    start=time.perf_counter();sr=net(x);torch.cuda.synchronize()
                    times.append((time.perf_counter()-start)*1000);del sr
                allocated=torch.cuda.max_memory_allocated()/1024**2
                reserved=torch.cuda.max_memory_reserved()/1024**2
            # Some torch2.5 counters return zero in inference_mode. Count in no_grad.
            with torch.no_grad(),FlopCounterMode(display=False) as counter:net(x)
            count=counter.get_total_flops()
            result[f'{label}_LR{h}x{w}']=dict(parameters=sum(p.numel() for p in net.parameters()),
                median_ms=statistics.median(times),p90_ms=sorted(times)[26],peak_allocated_MiB=allocated,
                peak_reserved_MiB=reserved,counted_flops=count if count>0 else None,
                count_coverage='supported conv/matmul/attention; excludes norm/sort/index/elementwise etc',raw_ms=times)
    return result

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-root',type=Path,default=Path('/root/autodl-tmp'))
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--b0-run',type=Path);p.add_argument('--f3-run',type=Path)
    p.add_argument('--expected-b0-sha',default=B0_SHA);p.add_argument('--expected-f3-sha',default=F3_SHA)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--backfill-set5',action='store_true')
    p.add_argument('--diagnose',action='store_true');p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    import torch
    torch.set_num_threads(4)
    b0=locate(args.search_root,'B0',args.expected_b0_sha,args.b0_run)
    f3=locate(args.search_root,'F3',args.expected_f3_sha,args.f3_run)
    output=ROOT/'experiment/audits'/('F3_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    fields=['scale','seed','patch_size','batch_size','optimizer','lr','eta_min','scheduler_t_max',
        'planned_epochs','stop_epoch','test_every','data_range','loss','self_ensemble','precision','rgb_range',
        'workers','gpu_name','torch_version','cuda_version','initial_shared_state_sha256',
        'data_fingerprint','git_commit','git_dirty','metric_protocol','source_sha256','resolved_arguments']
    configs={label:{k:load_json(run/'config.json').get(k) for k in fields} for label,run in [('B0',b0),('F3',f3)]}
    report=dict(paths=dict(B0=str(b0),F3=str(f3)),checkpoint_sha=dict(B0=sha(b0/'model/model_20.pt'),
        F3=sha(f3/'model/model_20.pt')),configs=configs,current_environment=dict(torch=torch.__version__,
        cuda=torch.version.cuda,device=args.device,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,threads=torch.get_num_threads()))
    for label,run in [('B0',b0),('F3',f3)]:
        snapshots={}
        scheduler=run/'scheduler.pt'
        if scheduler.is_file():
            state=torch.load(scheduler,map_location='cpu',weights_only=True)
            snapshots['scheduler']={k:state.get(k) for k in ['T_max','eta_min','last_epoch','base_lrs','_last_lr']}
        optimizer=run/'optimizer.pt'
        if optimizer.is_file():
            state=torch.load(optimizer,map_location='cpu',weights_only=True)
            steps=sorted({float(v['step']) for v in state['state'].values() if 'step' in v})
            snapshots['optimizer']=dict(unique_steps=steps,parameter_states=len(state['state']),
                groups=[{k:g.get(k) for k in ['lr','betas','eps','weight_decay']} for g in state['param_groups']])
        report.setdefault('training_state',{})[label]=snapshots
    bm,bpairs=compare_rows(benchmark_rows(b0,'B0'),benchmark_rows(f3,'F3'))
    report['benchmark_comparison']=bm;write_csv(output/'benchmark_pairs.csv',bpairs)
    curves={label:csv_rows(run/'metrics.csv') for label,run in [('B0',b0),('F3',f3)]}
    report['curves']=curves
    pairs=[]
    for epoch in range(1,21):
        a=torch.load(b0/'per_image_metrics'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
        b=torch.load(f3/'per_image_metrics'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
        stats,rows=compare_rows(a,b)
        if len(rows)!=100:raise ValueError('DIV2K requires 100 paired images')
        pairs.extend([dict(epoch=epoch,**r) for r in rows])
        report.setdefault('validation_by_epoch',{})[epoch]=stats['DIV2K']
    write_csv(output/'validation_pairs.csv',pairs)
    nets=models(b0,f3,args.device) if args.backfill_set5 or args.diagnose or args.profile else None
    if args.diagnose:report['frozen_diagnosis']=frozen_diagnosis(*nets,args.data_root)
    if args.profile:report['matched_resources']=profile(nets)
    if args.backfill_set5:report['set5_by_epoch']=backfill_set5(f3,nets[1],args.data_root,output)
    (output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    compact=dict(report)
    compact['configs']={label:{k:v for k,v in cfg.items() if k not in ['source_sha256','resolved_arguments']} for label,cfg in configs.items()}
    bs=configs['B0']['source_sha256'] or {};fs=configs['F3']['source_sha256'] or {}
    compact['common_source_hash_equal']={k:bs[k]==fs[k] if k in fs else None
        for k in bs if k.startswith('LFMN/') and 'lfmnf1' not in k}
    compact['validation_by_epoch']=report['validation_by_epoch']
    compact.pop('frozen_diagnosis',None)
    if 'frozen_diagnosis' in report:
        compact['frozen_diagnosis']=dict(scope=report['frozen_diagnosis']['scope'],state_unchanged=True,
            samples=[{k:v for k,v in r.items() if k!='stages'} for r in report['frozen_diagnosis']['samples']],
            mean_stage_correction_ratio=[sum(r['stages'][i]['correction_ratio'] for r in report['frozen_diagnosis']['samples'])/4 for i in range(8)])
    if 'matched_resources' in compact:
        compact['matched_resources']={k:{a:b for a,b in v.items() if a!='raw_ms'} for k,v in compact['matched_resources'].items() if isinstance(v,dict)}
    text=json.dumps(compact,indent=2,ensure_ascii=False)
    (output/'paste_report.txt').write_text(text,encoding='utf-8')
    print('\n===== PASTE REPORT BEGIN =====\n'+text+'\n===== PASTE REPORT END =====',flush=True)
    print('AUDIT OUTPUT:',output,flush=True)

if __name__=='__main__':main()
