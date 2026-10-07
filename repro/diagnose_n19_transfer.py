"""Frozen-backbone readout calibration diagnostic, HR control included."""
import argparse
import hashlib
import json
from pathlib import Path
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair(root, index, train, device):
    prefix = 'train' if train else 'valid'
    lp = root/f'DIV2K_{prefix}_LR_bicubic'/'X4'/f'{index:04d}x4.png'
    hp = root/f'DIV2K_{prefix}_HR'/f'{index:04d}.png'
    with Image.open(lp) as im:
        lr = np.array(im.convert('RGB'))
    with Image.open(hp) as im:
        hr = np.array(im.convert('RGB'))
    y, x = (lr.shape[0]-64)//2, (lr.shape[1]-64)//2
    def tensor(z):
        return torch.from_numpy(z.copy()).permute(2, 0, 1)[None].float().to(device)
    return tensor(lr[y:y+64,x:x+64]), tensor(hr[y*4:(y+64)*4,x*4:(x+64)*4]), {str(lp):sha(lp),str(hp):sha(hp)}


def mse(prediction, hr):
    assert torch.isfinite(prediction).all() and prediction.shape == hr.shape
    return ((prediction[...,4:-4,4:-4]-hr[...,4:-4,4:-4])/255.).square().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--teacher-checkpoint',type=Path,required=True)
    parser.add_argument('--teacher-repo',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(scale=4).to(device).eval().requires_grad_(False)
    checkpoint_hash = sha(args.checkpoint)
    teacher_hash = sha(args.teacher_checkpoint)
    assert teacher_hash == '129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a'
    net.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True),strict=True)
    teacher = SwinIRTeacher(SimpleNamespace(rgcrd_teacher_repo=str(args.teacher_repo),
                             rgcrd_teacher_checkpoint=str(args.teacher_checkpoint),rgcrd_teacher_amp=False),device)
    saved = {}
    def capture(module, arguments):
        saved['features'] = arguments[0].detach()
    handle = net.last_conv.register_forward_pre_hook(capture)
    xtx = torch.zeros((433,433),dtype=torch.float64,device=device)
    xty = torch.zeros((433,6),dtype=torch.float64,device=device)
    count, hashes = 0, {}
    fit_rows = []
    try:
        with torch.inference_mode():
            for index in range(1,17):
                lr, hr, h = pair(args.data_root,index,True,device)
                hashes.update(h)
                student = net(lr)
                u = saved['features']
                target, _ = teacher(lr)
                assert torch.isfinite(target).all()
                if index == 1:
                    direct = teacher.teacher(lr/255.)*255.
                    torch.testing.assert_close(target,direct,atol=.001,rtol=1e-5)
                patches = F.unfold(u,3,padding=1)[0].T
                grid = torch.arange(256,device=device)
                yy, xx = torch.meshgrid(grid,grid,indexing='ij')
                valid = ((yy>=8)&(yy<248)&(xx>=8)&(xx<248)).flatten().nonzero()[:,0]
                generator = torch.Generator(device='cpu').manual_seed(1900+index)
                ids = valid[torch.randperm(len(valid),generator=generator)[:4096].to(device)]
                x = patches[ids].double()
                x = torch.cat([x,torch.ones((len(x),1),device=device,dtype=torch.float64)],1)
                targets = torch.cat([(target-student)[0].permute(1,2,0).reshape(-1,3)[ids],
                                     (hr-student)[0].permute(1,2,0).reshape(-1,3)[ids]],1).double()
                xtx += x.T@x
                xty += x.T@targets
                count += len(x)
                fit_rows.append({'image':index,'baseline_mse':mse(student,hr),'teacher_mse':mse(target,hr)})
                print(f'Fit image {index}/16',flush=True)
            # Center and standardize sufficient statistics; bias is unpenalized.
            moment = xtx/count
            cross = xty/count
            mean = moment[:-1,-1]
            scale = (moment.diag()[:-1]-mean.square()).clamp_min(1e-8).sqrt()
            covariance = (moment[:-1,:-1]-mean[:,None]*mean[None,:])/(scale[:,None]*scale[None,:])
            rhs = (cross[:-1]-mean[:,None]*cross[-1:])/scale[:,None]
            standardized = torch.linalg.solve(covariance+torch.eye(432,device=device,dtype=torch.float64)*.01,rhs)
            weights = standardized/scale[:,None]
            biases = cross[-1]-(mean[:,None]*weights).sum(0)
            kernels = {'teacher_readout':weights[:,:3].T.reshape(3,48,3,3).float(),
                       'hr_control':weights[:,3:].T.reshape(3,48,3,3).float()}
            bias = {'teacher_readout':biases[:3].float(),'hr_control':biases[3:].float()}
            calibration = []
            for index in range(17,25):
                lr, hr, h = pair(args.data_root,index,True,device)
                hashes.update(h)
                student = net(lr)
                u = saved['features']
                base = mse(student,hr)
                for mode in kernels:
                    correction = F.conv2d(u,kernels[mode],bias[mode],padding=1)
                    for alpha in (0.,.25,.5,1.):
                        value = mse(student+alpha*correction,hr)
                        calibration.append({'image':index,'mode':mode,'alpha':alpha,'delta_psnr':float(10*np.log10(base/value))})
                print(f'Calibration image {index}',flush=True)
            selected = {mode:max((0.,.25,.5,1.),key=lambda alpha:np.mean([r['delta_psnr'] for r in calibration if r['mode']==mode and r['alpha']==alpha])) for mode in kernels}
            original_weight = net.last_conv.weight.detach().clone()
            original_bias = net.last_conv.bias.detach().clone()
            rows = []
            for index in range(827,843):
                lr, hr, h = pair(args.data_root,index,False,device)
                hashes.update(h)
                student = net(lr)
                u = saved['features']
                target,_ = teacher(lr)
                base = mse(student,hr)
                row = {'image':index,'baseline_psnr':float(-10*np.log10(base)),
                       'strong_teacher_delta':float(10*np.log10(base/mse(target,hr)))}
                for mode in kernels:
                    correction = F.conv2d(u,kernels[mode],bias[mode],padding=1)
                    value = student+selected[mode]*correction
                    row[mode+'_delta'] = float(10*np.log10(base/mse(value,hr)))
                    if index == 827:
                        # Verify merging introduces no architectural overhead.
                        base_image = F.interpolate(lr,scale_factor=4,mode='bilinear',align_corners=False)
                        folded = F.conv2d(u,original_weight+.5*kernels[mode],
                                          original_bias+.5*bias[mode],padding=1)+base_image
                        torch.testing.assert_close(student+.5*correction,folded,atol=.002,rtol=1e-5)
                row['teacher_minus_hr_control'] = row['teacher_readout_delta']-row['hr_control_delta']
                rows.append(row)
                print(json.dumps(row),flush=True)
    finally:
        handle.remove()
    assert sha(args.checkpoint)==checkpoint_hash
    assert torch.equal(net.last_conv.weight,original_weight) and torch.equal(net.last_conv.bias,original_bias)
    summary = {}
    rng = np.random.default_rng(1901)
    for mode in kernels:
        delta = np.array([r[mode+'_delta'] for r in rows])
        boot = delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
        ci = np.quantile(boot,[.025,.975]).tolist()
        summary[mode] = {'alpha':selected[mode],'mean_delta':float(delta.mean()),'wins':int((delta>0).sum()),
                         'bootstrap95':ci,'pass':bool(delta.mean()>0 and (delta>0).sum()>=12 and ci[0]>0)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'kernels':{k:v.cpu() for k,v in kernels.items()},'bias':{k:v.cpu() for k,v in bias.items()},
                'alpha':selected,'baseline_sha256':checkpoint_hash},args.output.parent/'readout_calibration.pt')
    result = {'checkpoint_sha256':checkpoint_hash,'teacher_sha256':teacher_hash,'checkpoint_unchanged':True,
              'source_sha256':sha(ROOT/'LFMN/model/lfmn.py'),'data_sha256':hashes,'fit':fit_rows,'calibration':calibration,
              'rows':rows,'summary':summary,'parameters':sum(p.numel() for p in net.parameters()),
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
              'seconds':time.time()-started,'fold_check_pass':True,
              'limitations':'Additional readout fitting, crop metrics, no full-network retraining or architectural innovation claim; teacher has different training budget.'}
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
