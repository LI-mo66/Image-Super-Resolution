"""Bounded paired local LFMN adaptation; standard KD diagnostic, not new architecture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'LFMN'))
from model.lfmn import Net
from rgcrd.teacher import SwinIRTeacher
from utility import calc_ssim


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch(root, rng, hashes, count=2, validation=None):
    pairs = []
    metadata = []
    for slot in range(count):
        index = int(rng.integers(1,801)) if validation is None else validation
        prefix = 'train' if validation is None else 'valid'
        lp = root/f'DIV2K_{prefix}_LR_bicubic'/'X4'/f'{index:04d}x4.png'
        hp = root/f'DIV2K_{prefix}_HR'/f'{index:04d}.png'
        for path in (lp,hp):
            if str(path) not in hashes:
                hashes[str(path)] = sha(path)
        with Image.open(lp) as im:
            lr = np.array(im.convert('RGB'))
        with Image.open(hp) as im:
            hr = np.array(im.convert('RGB'))
        assert hr.shape[:2] == (lr.shape[0]*4,lr.shape[1]*4)
        if validation is None:
            y = int(rng.integers(0,lr.shape[0]-64+1))
            x = int(rng.integers(0,lr.shape[1]-64+1))
            flips = rng.integers(0,2,3).tolist()
        else:
            y,x = (lr.shape[0]-64)//2,(lr.shape[1]-64)//2
            flips = [0,0,0]
        lr,hr = lr[y:y+64,x:x+64],hr[y*4:(y+64)*4,x*4:(x+64)*4]
        def augment(z):
            if flips[0]: z=z[:,::-1]
            if flips[1]: z=z[::-1]
            if flips[2]: z=z.transpose(1,0,2)
            return torch.from_numpy(z.copy()).permute(2,0,1).float()
        pairs.append((augment(lr),augment(hr)))
        metadata.append({'image':index,'y':y,'x':x,'augmentation':flips})
    a = torch.stack([p[0] for p in pairs])
    b = torch.stack([p[1] for p in pairs])
    fingerprint = hashlib.sha256(a.numpy().tobytes()+b.numpy().tobytes()).hexdigest()
    return a.cuda(),b.cuda(),metadata,fingerprint


def evaluate(net, root, hashes):
    net.eval()
    rows = []
    with torch.inference_mode():
        for index in range(843,859):
            lr,hr,_,_ = batch(root,np.random.default_rng(0),hashes,1,index)
            sr=net(lr)
            assert sr.shape == hr.shape and torch.isfinite(sr).all()
            error=((sr[...,4:-4,4:-4]-hr[...,4:-4,4:-4])/255.).square().mean().item()
            rows.append({'image':index,'psnr':float(-10*np.log10(error)),
                         'ssim':float(calc_ssim(sr,hr,4,255.))})
    return rows


def optimizer(net):
    return torch.optim.Adam(net.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=0)


def update(net, opt, lr, hr, teacher_output=None):
    net.train()
    opt.zero_grad(set_to_none=True)
    sr=net(lr)
    l1=F.l1_loss(sr,hr)
    kd=F.l1_loss(sr,teacher_output) if teacher_output is not None else sr.new_zeros(())
    objective=l1+.1*kd
    assert torch.isfinite(sr).all() and torch.isfinite(objective)
    objective.backward()
    gradients=[p.grad for p in net.parameters() if p.grad is not None]
    assert len(gradients)==sum(1 for p in net.parameters() if p.requires_grad)
    assert all(torch.isfinite(g).all() for g in gradients)
    norm=float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients)).item())
    assert norm>0
    opt.step()
    return {'l1':float(l1.item()),'kd':float(kd.item()),'gradient_norm':norm}


def checks(payload, teacher, args):
    a,b=Net(scale=4).cuda().eval(),Net(scale=4).cuda().eval()
    a.load_state_dict(payload,strict=True)
    b.load_state_dict(payload,strict=True)
    lr,hr,_,_=batch(args.data_root,np.random.default_rng(1900),{},2)
    with torch.inference_mode():
        assert torch.equal(a(lr),b(lr))
        t,_=teacher(lr)
        torch.testing.assert_close(t,teacher.teacher(lr/255.)*255.,atol=.001,rtol=1e-5)
    t=t.clone()
    before=a.last_conv.weight.detach().clone()
    basic=update(a,optimizer(a),lr,hr)
    guided=update(b,optimizer(b),lr,hr,t)
    assert not torch.equal(before,a.last_conv.weight)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'smoke_reload.pt'
    torch.save(b.state_dict(),path)
    a.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True)
    a.eval();b.eval()
    with torch.inference_mode():
        assert torch.equal(a(lr),b(lr))
        assert torch.isfinite(a(lr[:1])).all()
    return {'passed':True,'initial_exact':True,'reload_exact':True,'teacher_direct_match':True,
            'real_batch':'B2 LR64 HR256 FP32','l1_step':basic,'kd_step':guided,
            'parameters':sum(p.numel() for p in a.parameters()),'source_sha256':sha(Path(__file__))}


def summary(baseline, basic, guided):
    assert [r['image'] for r in baseline]==[r['image'] for r in basic]==[r['image'] for r in guided]
    d=np.array([k['psnr']-b['psnr'] for k,b in zip(guided,basic)])
    start=np.array([k['psnr']-b['psnr'] for k,b in zip(guided,baseline)])
    ss=np.array([k['ssim']-b['ssim'] for k,b in zip(guided,basic)])
    rng=np.random.default_rng(2001)
    ci=np.quantile(d[rng.integers(0,16,(10000,16))].mean(1),[.025,.975])
    gates={'kd_minus_l1_ge_001':bool(d.mean()>=.01),'wins_ge_12':bool((d>0).sum()>=12),
           'ci_lower_positive':bool(ci[0]>0),'ssim_nonnegative':bool(ss.mean()>=0),
           'kd_above_start':bool(start.mean()>0)}
    return {'kd_minus_l1':float(d.mean()),'kd_minus_start':float(start.mean()),
            'l1_minus_start':float(np.mean([b['psnr']-s['psnr'] for b,s in zip(basic,baseline)])),
            'ssim_delta':float(ss.mean()),'wins':int((d>0).sum()),'bootstrap95':ci.tolist(),
            'gates':gates,'decision':'PASS_LOCAL_EXPLORATORY' if all(gates.values()) else 'NO_GO'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--teacher-checkpoint',type=Path,required=True)
    parser.add_argument('--teacher-repo',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Approved local CUDA protocol required')
    checkpoint_hash=sha(args.checkpoint)
    assert checkpoint_hash=='e428004505ec01364f60a0812c879b6ffd1fc08d013e48ffe81a22904927023b'
    assert sha(args.teacher_checkpoint)=='129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a'
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if (args.output/'report.json').exists(): raise RuntimeError('Refuse to overwrite a completed run')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1902)
    rng_state=torch.get_rng_state(); cuda_state=torch.cuda.get_rng_state_all()
    teacher=SwinIRTeacher(SimpleNamespace(rgcrd_teacher_repo=str(args.teacher_repo),
        rgcrd_teacher_checkpoint=str(args.teacher_checkpoint),rgcrd_teacher_amp=False),torch.device('cuda'))
    torch.set_rng_state(rng_state);torch.cuda.set_rng_state_all(cuda_state)
    check=checks(payload,teacher,args)
    (args.output/'check.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(check),flush=True)
    if args.check_only:return
    basic,guided=Net(scale=4).cuda(),Net(scale=4).cuda()
    basic.load_state_dict(payload,strict=True);guided.load_state_dict(payload,strict=True)
    hashes={}
    start_metrics=evaluate(basic,args.data_root,hashes)
    opt_b,opt_k=optimizer(basic),optimizer(guided)
    torch.manual_seed(1902)
    rng=np.random.default_rng(1902)
    # Both models receive precisely the same materialized batch. No teacher
    # construction or model RNG operation influences data sampling.
    started=time.monotonic()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    with (args.output/'train.jsonl').open('w',encoding='utf-8') as logfile:
        for step in range(1,201):
            if time.monotonic()-started>1800:raise RuntimeError('Approved 30-minute training budget exhausted')
            lr,hr,metadata,fingerprint=batch(args.data_root,rng,hashes)
            with torch.inference_mode():
                target,_=teacher(lr)
                assert torch.isfinite(target).all()
                # Standard loss backward must not receive inference tensors.
                target=target.clone()
            target=target.clone()
            stats={}
            order=('b','k') if step%2 else ('k','b')
            for group in order:
                stats[group]=update(basic,opt_b,lr,hr) if group=='b' else update(guided,opt_k,lr,hr,target)
            row={'step':step,'batch_sha256':fingerprint,'batch':metadata,'stats':stats,
                 'seconds':time.monotonic()-started}
            logfile.write(json.dumps(row)+'\n');logfile.flush()
            if step%10==0:print(json.dumps(row),flush=True)
    basic_metrics=evaluate(basic,args.data_root,hashes)
    guided_metrics=evaluate(guided,args.data_root,hashes)
    torch.save({'model':basic.state_dict(),'optimizer':opt_b.state_dict(),'steps':200},args.output/'b_l1.pt')
    torch.save({'model':guided.state_dict(),'optimizer':opt_k.state_dict(),'steps':200},args.output/'k_kd.pt')
    assert sha(args.checkpoint)==checkpoint_hash
    result={'start':start_metrics,'l1':basic_metrics,'kd':guided_metrics,'summary':summary(start_metrics,basic_metrics,guided_metrics),
            'checkpoint_sha256':checkpoint_hash,'teacher_sha256':sha(args.teacher_checkpoint),'source_commit':commit,
            'script_sha256':sha(Path(__file__)),'baseline_source_sha256':sha(ROOT/'LFMN/model/lfmn.py'),
            'data_sha256':hashes,'steps_per_group':200,'seconds':time.monotonic()-started,
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'seed':1902,
            'metric':'RGB float PSNR HR border4; common Y-quantized SSIM border4; fixed center LR64 crops',
            'limitations':'Single seed crop local fine-tuning, matched extra training; not scratch, architecture improvement or benchmark claim.'}
    (args.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':
    main()
