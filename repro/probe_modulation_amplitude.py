"""Pre-registered frozen F3 amplitude probe. No optimizer, no training."""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
sys.path.insert(0,str(ROOT/'repro'))
from collect_modulation_audit import F3_SHA,locate,sha,image,score,write_csv
GRID=[0.,.5,.75,1.,1.25]
EXCLUDED={801,834,867,900}
SELECTION=[i for i in range(801,901) if i%2 and i not in EXCLUDED]
TEST=[i for i in range(801,901) if not i%2 and i not in EXCLUDED]


@contextmanager
def scaled_correction(net,value):
    handles=[];calls=[0]
    def hook(module,inputs,output):
        calls[0]+=1
        return output if value==1 else output*value
    try:
        for module in net.sfmls:handles.append(module.current_correction.register_forward_hook(hook))
        yield calls
    finally:
        for handle in handles:handle.remove()


def state_digest(net):
    digest=hashlib.sha256()
    for key,tensor in sorted(net.state_dict().items()):
        digest.update(key.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_hooks(net):
    import torch
    device=next(net.parameters()).device
    # Fixed image without consuming the training/data RNG stream.
    x=torch.linspace(0,255,3*32*32,device=device).reshape(1,3,32,32)
    before=state_digest(net)
    with torch.inference_mode():
        original=net(x)
        with scaled_correction(net,1.) as calls:
            actual=net(x)
            assert calls[0]==8 and torch.equal(original,actual),'lambda1 identity failure'
        for m in net.sfmls:m.enabled=False
        try:disabled=net(x)
        finally:
            for m in net.sfmls:m.enabled=True
        with scaled_correction(net,0.) as calls:
            actual=net(x)
            assert calls[0]==8 and torch.equal(disabled,actual),'lambda0 bypass mismatch'
    assert before==state_digest(net),'persistent state changed'
    return before


def evaluate(net,data,ids,value,split):
    import torch
    import utility
    device=next(net.parameters()).device;rows=[]
    with torch.inference_mode(),scaled_correction(net,value) as calls:
        for index in ids:
            base=data/'DIV2K'
            x=image(base/'DIV2K_valid_LR_bicubic/X4'/f'{index:04d}x4.png',device)
            hr=image(base/'DIV2K_valid_HR'/f'{index:04d}.png',device)
            h,w=x.shape[-2:];hr=hr[:,:,:h*4,:w*4]
            sr=net(x)
            if sr.shape!=hr.shape or not torch.isfinite(sr).all():raise ValueError('shape/nonfinite SR')
            mse=float(((sr-hr)/255.)[:,:,10:-10,10:-10].square().mean())
            quantized=utility.quantize(sr,255)
            row=dict(split=split,lambda_value=value,filename=f'{index:04d}',
                psnr=score(quantized,hr),ssim=utility.calc_ssim(quantized,hr,4,255),raw_rgb_mse=mse)
            rows.append(row)
            print('PROBE {} lambda={} {}/{} image={} PSNR={:.9f}'.format(
                split,value,len(rows),len(ids),index,row['psnr']),flush=True)
        if calls[0]!=8*len(ids):raise ValueError('actual correction call count mismatch')
    return rows


def aggregate(rows):
    import numpy as np
    return {key:float(np.mean([r[key] for r in rows])) for key in ['psnr','ssim','raw_rgb_mse']}


def select_lambda(tables):
    reference=aggregate(tables[1.])['psnr']
    best=max(GRID,key=lambda v:aggregate(tables[v])['psnr'])
    return 1. if aggregate(tables[best])['psnr']-reference<=1e-6 else best


def paired(reference,candidate):
    import numpy as np
    a={r['filename']:r for r in reference};b={r['filename']:r for r in candidate}
    if len(a)!=len(reference) or len(b)!=len(candidate) or set(a)!=set(b):raise ValueError('pair mismatch')
    keys=sorted(a)
    delta=np.array([b[k]['psnr']-a[k]['psnr'] for k in keys])
    rng=np.random.default_rng(1)
    return dict(n=len(keys),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
        positive=int((delta>0).sum()),positive_fraction=float((delta>0).mean()),
        ci95=np.quantile(delta[rng.integers(len(keys),size=(10000,len(keys)))].mean(1),[.025,.975]).tolist(),
        mean_ssim_delta=float(np.mean([b[k]['ssim']-a[k]['ssim'] for k in keys])),
        mean_raw_mse_delta=float(np.mean([b[k]['raw_rgb_mse']-a[k]['raw_rgb_mse'] for k in keys])))


def decision(value,stats):
    passed=(stats['n']==48 and value!=1 and stats['mean_delta']>=.005 and stats['median_delta']>0 and
            stats['positive_fraction']>=.6 and stats['ci95'][0]>0 and stats['mean_ssim_delta']>=-1e-4)
    return 'PROBE_SUPPORTS_FURTHER_INVESTIGATION' if passed else 'PROBE_INCONCLUSIVE'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search-root',type=Path,default=Path('/root/autodl-tmp'))
    p.add_argument('--f3-run',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    import torch
    from model.lfmnf3 import Net
    torch.set_num_threads(4)
    run=locate(args.search_root,'F3',F3_SHA,args.f3_run)
    checkpoint=run/'model/model_20.pt';checkpoint_sha=sha(checkpoint)
    output=ROOT/'experiment/amplitude_probes'/('F3_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    source_paths=[ROOT/'LFMN/model/lfmn.py',ROOT/'LFMN/model/lfmnf3.py',ROOT/'LFMN/utility.py',Path(__file__)]
    manifest=dict(kind='frozen amplitude probe, zero optimizer steps',checkpoint_sha256=checkpoint_sha,
        checkpoint=str(checkpoint),grid=GRID,selection_ids=SELECTION,test_ids=TEST,excluded_ids=sorted(EXCLUDED),
        split_scope='48 selection /48 test; previously evaluated DIV2K, not fresh blind test',
        metric='full image quantized RGB PSNR crop10; project Y SSIM crop4; raw RGB MSE crop10 normalized255',
        self_ensemble=False,chop=False,device=args.device,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in source_paths})
    digest=hashlib.sha256()
    for index in sorted(SELECTION+TEST):
        for folder,name in [('DIV2K_valid_HR',f'{index:04d}.png'),('DIV2K_valid_LR_bicubic/X4',f'{index:04d}x4.png')]:
            path=args.data_root/'DIV2K'/folder/name
            digest.update((folder+'/'+name).encode());digest.update(bytes.fromhex(sha(path)))
    manifest['data_pair_sha256']=digest.hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PROBE OUTPUT:',output,flush=True)
    net=Net(scale=4).to(args.device).eval()
    net.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True),strict=True)
    before=verify_hooks(net);started=time.monotonic()
    selection={1.:evaluate(net,args.data_root,SELECTION,1.,'selection')}
    test_reference=evaluate(net,args.data_root,TEST,1.,'test_reference')
    all_rows=selection[1.]+test_reference
    write_csv(output/'per_image_metrics.csv',all_rows)
    for value in GRID:
        if value==1.:continue
        selection[value]=evaluate(net,args.data_root,SELECTION,value,'selection')
        all_rows+=selection[value];write_csv(output/'per_image_metrics.csv',all_rows)
    chosen=select_lambda(selection)
    locked=dict(selected_lambda=chosen,selection_metrics={str(v):aggregate(selection[v]) for v in GRID},
        selection_rule='highest selection mean PSNR; within1e-6 of reference retain1; test never used to select')
    (output/'locked_selection.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')
    print('LOCKED SELECTED LAMBDA:',chosen,flush=True)
    test_candidate=test_reference if chosen==1 else evaluate(net,args.data_root,TEST,chosen,'test_selected')
    if chosen!=1:all_rows+=test_candidate;write_csv(output/'per_image_metrics.csv',all_rows)
    test_stats=paired(test_reference,test_candidate)
    if before!=state_digest(net) or checkpoint_sha!=sha(checkpoint):raise AssertionError('state/checkpoint changed')
    report=dict(manifest=manifest,**locked,test_comparison=test_stats,
        test_reference=aggregate(test_reference),test_selected=aggregate(test_candidate),
        state_unchanged=True,checkpoint_unchanged=True,optimizer_steps=0,
        elapsed_seconds=time.monotonic()-started,decision=decision(chosen,test_stats),
        interpretation='checkpoint dependence only; not architecture gain, not proof of generalization, no automatic training')
    text=json.dumps(report,indent=2)
    (output/'probe_report.json').write_text(text,encoding='utf-8')
    (output/'paste_report.txt').write_text(text,encoding='utf-8')
    print('\n===== AMPLITUDE REPORT BEGIN =====\n'+text+'\n===== AMPLITUDE REPORT END =====',flush=True)


if __name__=='__main__':main()
