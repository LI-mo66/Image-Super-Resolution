"""Frozen F4 epoch150 gate identity interventions; zero optimizer steps."""
import argparse
from contextlib import contextmanager
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from source_provenance import provenance

SHA150='9ec79bee0889ba68501e2bd7c84fae903b1a066e5339207296adfcd7112de149'
IDS=[801+99*i//23 for i in range(24)]
MODES=['original','all_identity']+[f'stage{i}_identity' for i in range(1,9)]
FIELDS=['psnr_rgb','psnr_y','ssim_y','quant_rgb_mse','quant_y_mse','raw_rgb_mse']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(t):
    h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
    h.update(t.detach().cpu().contiguous().numpy().tobytes());return h.hexdigest()


def state_digest(net):
    h=hashlib.sha256()
    for k,v in sorted(net.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def csv_write(path,rows):
    if not rows:return
    temp=Path(path).with_suffix('.csv.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)


@contextmanager
def gate_control(net,mode,statistics=False):
    import torch
    if mode not in MODES:raise ValueError('unknown mode')
    if statistics and mode!='original':raise ValueError('statistics only for original')
    gates=list(net.residual_gates)
    if len(gates)!=8:raise ValueError('eight gates required')
    saved=[g.enabled for g in gates];counts=[0]*8;handles=[];stats={}
    off=set(range(8)) if mode=='all_identity' else {int(mode[5])-1} if mode.startswith('stage') else set()
    def projection_hook(i):
        def hook(module,inputs,logits):
            g=1+torch.tanh(logits)
            stats[i]=dict(stage=i+1,gate_mean=float(g.mean()),gate_std=float(g.std(unbiased=False)),
                          fraction_below_005=float((g<.05).float().mean()),fraction_above_195=float((g>1.95).float().mean()))
        return hook
    def gate_hook(i):
        def hook(module,inputs,output):
            counts[i]+=1
            if not module.enabled and output is not inputs[1]:raise AssertionError('identity must return R, not zero')
            if statistics:
                stats[i].update(residual_rms=float(inputs[1].square().mean().sqrt()),
                                modulated_residual_rms=float(output.square().mean().sqrt()))
        return hook
    try:
        for i,g in enumerate(gates):
            g.enabled=i not in off
            handles.append(g.register_forward_hook(gate_hook(i)))
            if statistics:handles.append(g.projection[-1].register_forward_hook(projection_hook(i)))
        yield counts,stats
    finally:
        for h in handles:h.remove()
        for g,enabled in zip(gates,saved):g.enabled=enabled


def wrapper(net,device):
    import torch
    from model import Model
    m=Model.__new__(Model);torch.nn.Module.__init__(m)
    m.model=net;m.chop=False;m.self_ensemble=False;m.precision='single';m.device=torch.device(device);m.n_GPUs=1;m.eval()
    m.forward_x8=lambda *a,**k:(_ for _ in ()).throw(AssertionError('x8 entered'))
    calls=[0];original=net.forward
    def counted(x):calls[0]+=1;return original(x)
    net.forward=counted
    return m,calls


def metrics(sr,hr):
    import utility
    from types import SimpleNamespace
    q=utility.quantize(sr,255)
    raw=(sr-hr)/255;diff=(q-hr)/255
    rgb=float(diff[...,10:-10,10:-10].square().mean())
    coeff=diff.new_tensor([65.738,129.057,25.064]).view(1,3,1,1)/256
    y=float((diff*coeff).sum(1)[...,4:-4,4:-4].square().mean())
    out=dict(psnr_rgb=utility.calc_psnr(q,hr,4,255),
             psnr_y=utility.calc_psnr(q,hr,4,255,dataset=SimpleNamespace(dataset=SimpleNamespace(benchmark=True))),
             ssim_y=float(utility.calc_ssim(q,hr,4,255)),quant_rgb_mse=rgb,quant_y_mse=y,
             raw_rgb_mse=float(raw[...,10:-10,10:-10].square().mean()))
    if not all(math.isfinite(v) for v in out.values()):raise FloatingPointError('nonfinite metrics')
    return out


def paired(original,changed):
    import numpy as np
    a={r['filename']:r for r in original};b={r['filename']:r for r in changed}
    if set(a)!=set(b) or len(a)!=len(original) or len(b)!=len(changed):raise ValueError('paired IDs/duplicates')
    rng=np.random.default_rng(20261011);indices=rng.integers(0,len(a),size=(20000,len(a)))
    report={}
    for field in FIELDS:
        d=np.array([b[k][field]-a[k][field] for k in sorted(a)])
        report[field]=dict(mean_delta=float(d.mean()),median_delta=float(np.median(d)),
            positive=int((d>0).sum()),negative=int((d<0).sum()),total=len(d),
            image_bootstrap95=np.quantile(d[indices].mean(1),[.025,.975]).tolist())
    return report


def read_pair(root,index,device):
    import imageio.v2 as imageio
    from data.common import set_channel,np2Tensor
    lr=root/'DIV2K/DIV2K_valid_LR_bicubic/X4'/f'{index:04d}x4.png'
    hr=root/'DIV2K/DIV2K_valid_HR'/f'{index:04d}.png'
    a,b=set_channel(imageio.imread(lr),imageio.imread(hr),n_channels=3)
    h,w=a.shape[:2];b=b[:h*4,:w*4]
    if a.shape[2]!=3 or b.shape!=(h*4,w*4,3):raise ValueError('RGB x4 pairing failed')
    x,y=np2Tensor(a,b,rgb_range=255)
    return x.unsqueeze(0).to(device),y.unsqueeze(0).to(device)


def verify(net,m,calls):
    import torch
    device=next(net.parameters()).device;before=state_digest(net)
    with torch.inference_mode():
        for shape in [(1,3,32,32),(1,3,33,47)]:
            x=torch.linspace(0,255,math.prod(shape),device=device).reshape(shape)
            ref=m(x,0)
            for mode in MODES:
                n=calls[0]
                with gate_control(net,mode,statistics=mode=='original') as (counts,stats):actual=m(x,0)
                if calls[0]-n!=1 or counts!=[1]*8 or not torch.isfinite(actual).all():raise AssertionError('actual forward/hook count')
                if mode=='original' and not torch.equal(ref,actual):raise AssertionError('original hook changes output')
                if not all(g.enabled for g in net.residual_gates):raise AssertionError('flags not restored')
    if before!=state_digest(net):raise AssertionError('verification changed state')
    return before


def child(args,out):
    import torch
    from model.lfmnf4 import Net
    torch.set_num_threads(4)
    net=Net(scale=4).to(args.device).eval()
    net.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=True),strict=True)
    net.requires_grad_(False)
    if not all(bool(tab.initted) for tab,lrsa in net.blocks):raise ValueError('trained TAB buffers required')
    m,calls=wrapper(net,args.device);before=verify(net,m,calls);rows=[];stages=[];started=time.monotonic()
    reference_path=args.checkpoint.parents[1]/'per_image_metrics/epoch_0150.pt'
    reference=torch.load(reference_path,map_location='cpu',weights_only=True) if reference_path.exists() else []
    reference={r['filename']:r for r in reference if r['dataset']=='DIV2K'}
    input_manifest=[]
    first_result=None;start_calls=calls[0]
    with torch.inference_mode():
        for position,index in enumerate(IDS):
            x,hr=read_pair(args.data_root,index,args.device);split='A' if position%2==0 else 'B'
            input_hash=(tensor_digest(x),tensor_digest(hr))
            input_manifest.append(dict(filename=f'{index:04d}',split=split,lr_shape=list(x.shape),hr_shape=list(hr.shape),dtype=str(x.dtype),lr_tensor_sha256=input_hash[0],hr_tensor_sha256=input_hash[1]))
            for mode in MODES:
                n=calls[0]
                with gate_control(net,mode,statistics=mode=='original') as (counts,stats):sr=m(x,0)
                if counts!=[1]*8 or calls[0]-n!=1 or sr.shape!=hr.shape or not torch.isfinite(sr).all():raise AssertionError('shape/OFF/count')
                if not all(g.enabled for g in net.residual_gates):raise AssertionError('flags not restored')
                score=metrics(sr,hr)
                if mode=='original':
                    if position==0:first_result=sr.detach().clone()
                    old=reference.get(f'{index:04d}')
                    if old and (abs(old['psnr']-score['psnr_rgb'])>1e-4 or abs(old['ssim']-score['ssim_y'])>1e-4):
                        raise ValueError('original differs from epoch150 reference; inspect data/runtime')
                    stages += [dict(filename=f'{index:04d}',split=split,**stats[i]) for i in range(8)]
                rows.append(dict(filename=f'{index:04d}',split=split,mode=mode,**score))
                print('GATE_PROBE {} image={} {}/24 RGB_PSNR={:.9f} Y_PSNR={:.9f}'.format(mode,index,position+1,score['psnr_rgb'],score['psnr_y']),flush=True)
            if input_hash!=(tensor_digest(x),tensor_digest(hr)):raise AssertionError('inputs changed')
            write_json(out/'input_tensors.json',input_manifest)
            csv_write(out/'per_image_metrics.csv',rows);csv_write(out/'original_stage_statistics.csv',stages)
            write_json(out/'progress.json',dict(completed_images=position+1,formal_forwards=calls[0]-start_calls))
            if state_digest(net)!=before:raise AssertionError('model/buffers changed')
        formal_calls=calls[0]-start_calls
        x,_=read_pair(args.data_root,IDS[0],args.device)
        if not torch.equal(first_result,m(x,0)):raise AssertionError('original repeat differs after interventions')
    if formal_calls!=240 or state_digest(net)!=before or sha(args.checkpoint)!=SHA150:
        raise AssertionError('state/checkpoint/count integrity')
    report=dict(status='COMPLETE_F4_GATE_DIAGNOSTIC',checkpoint_sha256=SHA150,ids=IDS,modes=MODES,
        formal_forwards=formal_calls,optimizer_steps=0,state_unchanged=True,checkpoint_unchanged=True,
        flags_restored=all(g.enabled for g in net.residual_gates),elapsed_seconds=time.monotonic()-started,
        reference_images_checked=len([i for i in IDS if f'{i:04d}' in reference]),
        results={},limits=['identity-original is frozen-checkpoint intervention, not B0 comparison',
        'images previously evaluated; split A/B are descriptive, not fresh blind tests',
        '9 intervention CIs uncorrected for multiple comparisons; not training-seed confidence',
        'stage intervention changes downstream features/routing; effects are not additive'])
    for split in ['all','A','B']:
        selected=[r for r in rows if split=='all' or r['split']==split]
        original=[r for r in selected if r['mode']=='original']
        report['results'][split]=dict(absolute={mode:{k:sum(r[k] for r in selected if r['mode']==mode)/len(original) for k in FIELDS} for mode in MODES},
            versus_original={mode:paired(original,[r for r in selected if r['mode']==mode]) for mode in MODES[1:]})
    write_json(out/'gate_report.json',report)
    lines=['F4 epoch150 gate identity probe | OFF | 24 full DIV2K images | no training',
           'Delta = intervention - original; positive PSNR means bypass improved this checkpoint.']
    for split in ['all','A','B']:
        for mode,d in report['results'][split]['versus_original'].items():
            z=d['psnr_rgb'];lines.append('{} {} deltaRGB={:+.9f} median={:+.9f} wins={}/{} CI={} deltaY={:+.9f} deltaSSIM={:+.9f} deltaQuantMSE={:+.9g} deltaRawMSE={:+.9g}'.format(
                split,mode,z['mean_delta'],z['median_delta'],z['positive'],z['total'],z['image_bootstrap95'],d['psnr_y']['mean_delta'],d['ssim_y']['mean_delta'],d['quant_rgb_mse']['mean_delta'],d['raw_rgb_mse']['mean_delta']))
    text='\n'.join(lines);(out/'paste_report.txt').write_text(text,encoding='utf-8');print('\n'+text,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--internal',action='store_true')
    args=p.parse_args();args.checkpoint=args.checkpoint.resolve();args.data_root=args.data_root.resolve()
    if sha(args.checkpoint)!=SHA150:raise ValueError('only audited F4 fixed epoch150 allowed')
    cfgpath=args.checkpoint.parents[1]/'config.json'
    cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    for name in ['LFMN/model/lfmn.py','LFMN/model/lfmnf4.py','LFMN/utility.py','LFMN/data/common.py']:
        if sha(ROOT/name)!=cfg['source_sha256'][name]:raise ValueError('original model/metric source changed: '+name)
    if args.internal:
        import os
        if not os.environ.get('LFMN_MANAGED_RUN'):raise ValueError('internal requires captured launcher')
        child(args,Path(os.environ['LFMN_MANAGED_RUN']));return
    import torch
    h=hashlib.sha256();pair_records=[]
    for index in IDS:
        for folder,name in [('DIV2K_valid_HR',f'{index:04d}.png'),('DIV2K_valid_LR_bicubic/X4',f'{index:04d}x4.png')]:
            path=args.data_root/'DIV2K'/folder/name;d=sha(path);h.update((folder+'/'+name).encode());h.update(bytes.fromhex(d))
            pair_records.append(dict(path=str(path),sha256=d))
    out=ROOT/'experiment/gate_probes'/('F4_epoch150_seed1_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    config=dict(kind='frozen F4 gate intervention; no optimizer',checkpoint=str(args.checkpoint),checkpoint_sha256=SHA150,
        ids=IDS,modes=MODES,data_root=str(args.data_root),data_pair_sha256=h.hexdigest(),pair_records=pair_records,
        self_ensemble=False,chop=False,precision='FP32',optimizer_steps=0,
        metric='RGB quant PSNR/MSE crop10; Y quant PSNR/MSE BT601-256 crop4; project Y SSIM crop4; raw RGB MSE crop10 /255',
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        **provenance(ROOT),source_sha256={n:sha(ROOT/n) for n in ['repro/probe_f4_gates.py','LFMN/model/lfmn.py','LFMN/model/lfmnf4.py','LFMN/utility.py','LFMN/data/common.py']})
    print('GATE OUTPUT:',out,flush=True)
    c=[sys.executable,'-u',str(Path(__file__).resolve()),'--checkpoint',str(args.checkpoint),
       '--data-root',str(args.data_root),'--device',args.device,'--internal']
    launch(c,out,config,ROOT)
    print('GATE OUTPUT:',out,flush=True)


if __name__=='__main__':main()
