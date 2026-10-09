"""Frozen spatial-information dependence. No training and no mode selection."""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'));sys.path.insert(0,str(ROOT/'repro'))
from collect_modulation_audit import F3_SHA,locate,sha,image,score,write_csv
from probe_modulation_amplitude import SELECTION,TEST,EXCLUDED,state_digest,aggregate,paired
MODES=['original','spatial_mean','shuffle17','shuffle29']


def transform(output,mode,image_id,stage):
    import torch
    if mode=='original':return output
    if mode=='spatial_mean':return output.mean((-2,-1),keepdim=True).expand_as(output).contiguous()
    if mode not in ['shuffle17','shuffle29']:raise ValueError('unknown spatial mode')
    seed=17 if mode=='shuffle17' else 29
    generator=torch.Generator(device='cpu')
    generator.manual_seed(seed+image_id*1009+stage*1000003)
    count=output.shape[-2]*output.shape[-1]
    permutation=torch.randperm(count,generator=generator,device='cpu').to(output.device)
    return output.flatten(2).index_select(2,permutation).reshape_as(output).contiguous()


@contextmanager
def spatial_control(net,mode,image_id):
    handles=[];calls=[0];stats=[]
    def hook(stage):
        def capture(module,inputs,output):
            calls[0]+=1
            if mode=='original':
                energy=float(output.square().mean())
                mean_energy=float(output.mean((-2,-1),keepdim=True).square().mean())
                stats.append(dict(filename=f'{image_id:04d}',stage=stage+1,
                    correction_rms=energy**.5,spatial_mean_rms=mean_energy**.5,
                    spatial_mean_energy_fraction=mean_energy/max(energy,1e-30)))
            return transform(output,mode,image_id,stage)
        return capture
    try:
        for stage,module in enumerate(net.sfmls):
            handles.append(module.current_correction.register_forward_hook(hook(stage)))
        yield calls,stats
    finally:
        for handle in handles:handle.remove()


def verify(net):
    import torch
    device=next(net.parameters()).device
    x=torch.linspace(0,255,3*32*32,device=device).reshape(1,3,32,32)
    before=state_digest(net)
    with torch.inference_mode():
        original=net(x)
        with spatial_control(net,'original',1) as (calls,stats):
            actual=net(x)
            assert calls[0]==8 and len(stats)==8 and torch.equal(actual,original)
        for mode in MODES[1:]:
            with spatial_control(net,mode,1) as (calls,stats):
                actual=net(x);assert torch.isfinite(actual).all() and calls[0]==8
    assert before==state_digest(net)
    return before


def evaluate(net,data,ids,mode,split):
    import torch
    import utility
    rows=[];stages=[];device=next(net.parameters()).device
    with torch.inference_mode():
        for index in ids:
            root=data/'DIV2K'
            x=image(root/'DIV2K_valid_LR_bicubic/X4'/f'{index:04d}x4.png',device)
            hr=image(root/'DIV2K_valid_HR'/f'{index:04d}.png',device)
            h,w=x.shape[-2:];hr=hr[:,:,:4*h,:4*w]
            with spatial_control(net,mode,index) as (calls,stats):sr=net(x)
            if calls[0]!=8 or sr.shape!=hr.shape or not torch.isfinite(sr).all():
                raise ValueError('shape/finite/hook count failed')
            stages+=stats
            raw_mse=float(((sr-hr)/255.)[:,:,10:-10,10:-10].square().mean())
            quantized=utility.quantize(sr,255)
            rows.append(dict(split=split,mode=mode,filename=f'{index:04d}',psnr=score(quantized,hr),
                ssim=utility.calc_ssim(quantized,hr,4,255),raw_rgb_mse=raw_mse))
            print('SPATIAL_PROBE {} {} {}/{} image={} PSNR={:.9f}'.format(
                split,mode,len(rows),len(ids),index,rows[-1]['psnr']),flush=True)
    return rows,stages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-root',type=Path,default=Path('/root/autodl-tmp'))
    p.add_argument('--f3-run',type=Path);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    import torch
    import numpy as np
    from model.lfmnf3 import Net
    torch.set_num_threads(4)
    run=locate(args.search_root,'F3',F3_SHA,args.f3_run)
    checkpoint=run/'model/model_20.pt';checkpoint_sha=sha(checkpoint)
    output=ROOT/'experiment/spatial_probes'/('F3_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    source=[ROOT/'LFMN/model/lfmn.py',ROOT/'LFMN/model/lfmnf3.py',ROOT/'LFMN/utility.py',
            Path(__file__),ROOT/'repro/probe_modulation_amplitude.py',ROOT/'repro/collect_modulation_audit.py']
    digest=hashlib.sha256()
    for index in sorted(SELECTION+TEST):
        for folder,name in [('DIV2K_valid_HR',f'{index:04d}.png'),('DIV2K_valid_LR_bicubic/X4',f'{index:04d}x4.png')]:
            digest.update((folder+'/'+name).encode())
            digest.update(bytes.fromhex(sha(args.data_root/'DIV2K'/folder/name)))
    manifest=dict(kind='frozen spatial dependence; zero optimizer steps',checkpoint_sha256=checkpoint_sha,
        checkpoint=str(checkpoint),modes=MODES,selection_ids=SELECTION,test_ids=TEST,excluded_ids=sorted(EXCLUDED),
        split_scope='previously evaluated48/48; no fresh blind test and no mode selection',
        metric='full image quantized RGB PSNR crop10; project Y SSIM crop4; raw RGB MSE crop10 normalized255',
        self_ensemble=False,chop=False,precision='FP32',device=args.device,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source},data_pair_sha256=digest.hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('SPATIAL OUTPUT:',output,flush=True)
    net=Net(scale=4).to(args.device).eval()
    net.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True),strict=True)
    before=verify(net);started=time.monotonic();tables={};all_rows=[];all_stages=[]
    for split,ids in [('selection',SELECTION),('test',TEST)]:
        tables[split]={}
        for mode in MODES:
            rows,stages=evaluate(net,args.data_root,ids,mode,split)
            tables[split][mode]=rows;all_rows+=rows;all_stages+=stages
            write_csv(output/'per_image_metrics.csv',all_rows)
            write_csv(output/'original_stage_statistics.csv',all_stages)
    if checkpoint_sha!=sha(checkpoint) or before!=state_digest(net):raise AssertionError('state/checkpoint changed')
    summary={}
    for split in tables:
        summary[split]=dict(absolute={mode:aggregate(tables[split][mode]) for mode in MODES},
            versus_original={mode:paired(tables[split]['original'],tables[split][mode]) for mode in MODES[1:]})
    stage_summary=[]
    for stage in range(1,9):
        selected=[r for r in all_stages if r['stage']==stage]
        stage_summary.append(dict(stage=stage,
            mean_spatial_mean_energy_fraction=float(np.mean([r['spatial_mean_energy_fraction'] for r in selected])),
            median_spatial_mean_energy_fraction=float(np.median([r['spatial_mean_energy_fraction'] for r in selected])),
            mean_correction_rms=float(np.mean([r['correction_rms'] for r in selected]))))
    report=dict(manifest=manifest,results=summary,original_stage_summary=stage_summary,
        state_unchanged=True,checkpoint_unchanged=True,optimizer_steps=0,elapsed_seconds=time.monotonic()-started,
        decision='DIAGNOSTIC_ONLY_NO_AUTOMATIC_PROMOTION',
        limits=['mean intervention changes variance/amplitude and remains image-dependent',
                'shuffle destroys spatial coherence as well as alignment; later-stage inputs change',
                'permutation seeds are not training seeds; images/splits were previously evaluated',
                'checkpoint dependence does not prove architecture improvement or method-family failure'])
    text=json.dumps(report,indent=2)
    (output/'spatial_report.json').write_text(text,encoding='utf-8')
    (output/'paste_report.txt').write_text(text,encoding='utf-8')
    print('\n===== SPATIAL REPORT BEGIN =====\n'+text+'\n===== SPATIAL REPORT END =====',flush=True)


if __name__=='__main__':main()
