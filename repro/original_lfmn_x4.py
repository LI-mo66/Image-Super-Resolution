"""Original binary LFMN x4 scratch baseline. No pretrained/TTA/model modification.

The python backend is explicitly restricted to engineering smoke diagnostics.
Epoch checkpoints are transactional; latest is the source of truth for CSV/report.
"""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from types import SimpleNamespace

from lfmn_baseline_logging import RunLog, atomic_json

ROOT = Path(__file__).resolve().parents[1]
SETS = {'Set5': 5, 'Set14': 14, 'B100': 100, 'Urban100': 100, 'Manga109': 109}
BINARY_SHA = 'cb147a6d540e384e6233a23c28dbdc54ee1421af0f6f40dece5e32817fcc37ee'
PAPER = dict(zip(SETS, [(32.64, .9004), (28.95, .7893), (27.80, .7438),
                        (26.91, .8087), (31.45, .9193)]))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return 'unknown (standalone delivery without Git metadata)'


def code_fingerprint():
    return {p: sha(ROOT / p) for p in ['repro/original_lfmn_x4.py',
            'repro/lfmn_baseline_logging.py','repro/preflight_original_lfmn_x4.py',
            'repro/check_original_lfmn_x4.py','repro/check_lfmn_baseline_logging.py', 'LFMN/model/common.py',
            'LFMN/model/lfmn.py', 'LFMN/model/lfmn.cpython-39-x86_64-linux-gnu.so',
            'LFMN/data/common.py']}


def protocol(args, manifest):
    return dict(model='LFMN', backend=args.backend, scale=4, n_stage=8,
                model_source='hehesjtu/LFMN@4fe1614ef24f03bee81d53f5d7ec7df91606d93b',
                epochs=1000, steps_per_epoch=1000, batch_size=8, patch_size=256,
                seed=args.seed, rgb_range=255, loss='1*L1', optimizer='Adam',
                learning_rate=2e-4, betas=[.9, .999], epsilon=1e-8, weight_decay=0,
                scheduler=dict(type='MultiStepLR', milestones=[200,400,600,800,1000], gamma=.5),
                dataset='DIV2K', train_range='1-800', validation_range=dict(SETS),
                degradation='provided bicubic x4; do not resize online',
                augmentation='paired horizontal/vertical flip and transpose p=.5',
                pretrained_checkpoint=None, self_ensemble_enabled=False, ema_enabled=False,
                chop_enabled=False, amp_enabled=False, tf32_enabled=False,
                metric_protocol='official PSNR float32 Y coefficients/256, quantize SR, shave4; '
                                'SSIM float64 BT.601 Y, Gaussian11 sigma1.5 valid, shave4',
                selection='equal mean of five dataset mean PSNR; ties keep earliest epoch',
                data_fingerprint=manifest['fingerprint'], source_fingerprint=code_fingerprint(),
                workers=args.workers, smoke=args.smoke, smoke_steps=args.smoke_steps,
                smoke_eval_limit=args.smoke_eval_limit)


def scan_data(root, check_pixels=True):
    import imageio.v2 as imageio
    import numpy as np
    root = Path(root).resolve()
    groups = {}
    manifest = []
    for name, count in {'DIV2K':800, **SETS}.items():
        base = root / ('DIV2K' if name == 'DIV2K' else 'benchmark/' + name)
        if name == 'Manga109' and not base.exists():
            base = root / 'benchmark/manga109'
        hrdir = base / ('DIV2K_train_HR' if name == 'DIV2K' else 'HR')
        lrdir = base / ('DIV2K_train_LR_bicubic/X4' if name == 'DIV2K' else 'LR_bicubic/X4')
        hrfiles = sorted(hrdir.glob('*.png'))
        if len(hrfiles) != count:
            raise ValueError('%s needs exactly %d PNG HR images, found %d at %s' %
                             (name, count, len(hrfiles), hrdir))
        if name == 'DIV2K' and [p.stem for p in hrfiles] != ['%04d' % i for i in range(1,801)]:
            raise ValueError('DIV2K IDs must be 0001..0800')
        pairs = []
        expected_lr = set()
        for hp in hrfiles:
            lp = lrdir / (hp.stem + 'x4.png')
            expected_lr.add(lp.name)
            if not lp.is_file():
                raise FileNotFoundError(lp)
            if check_pixels:
                h, l = imageio.imread(hp), imageio.imread(lp)
                for a in [h,l]:
                    if a.dtype != np.uint8 or a.ndim not in (2,3) or (a.ndim == 3 and a.shape[2] not in (1,3)):
                        raise ValueError('Only uint8 grayscale/RGB accepted: %s' % hp)
                if h.shape[0] < l.shape[0]*4 or h.shape[1] < l.shape[1]*4 or \
                        h.shape[0]-l.shape[0]*4 > 3 or h.shape[1]-l.shape[1]*4 > 3:
                    raise ValueError('Misaligned x4 pair: %s %s' % (hp,lp))
                if name == 'DIV2K' and (min(l.shape[:2]) < 64 or h.ndim != 3 or l.ndim != 3):
                    raise ValueError('DIV2K training requires RGB and LR >=64')
                if min(l.shape[:2])*4-8 < 11:
                    raise ValueError('Image too small for valid SSIM')
            pairs.append((str(lp),str(hp)))
            manifest.extend([dict(path=str(p.relative_to(root)).replace('\\','/'), sha256=sha(p))
                             for p in [lp,hp]])
            if len(pairs) % 100 == 0:
                print('DATA CHECK %s decoding/hash %d/%d' % (name,len(pairs),count),flush=True)
        if {p.name for p in lrdir.glob('*.png')} != expected_lr:
            raise ValueError('Unexpected LR files in %s' % lrdir)
        groups[name] = pairs
        print('DATA CHECK %s: %d complete x4 pairs' % (name,len(pairs)), flush=True)
    digest = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    return groups, dict(root=str(root), files=manifest, fingerprint=digest)


def build_model(backend):
    import torch
    sys.path.insert(0, str(ROOT / 'LFMN'))
    if backend == 'official':
        if platform.system() != 'Linux' or sys.version_info[:2] != (3,9) or platform.machine() not in ('x86_64','AMD64'):
            raise RuntimeError('Original binary requires Linux x86_64 Python 3.9. No automatic fallback.')
        path = ROOT / 'LFMN/model/lfmn.cpython-39-x86_64-linux-gnu.so'
        if sha(path) != BINARY_SHA:
            raise RuntimeError('Original binary SHA256 mismatch')
        common_bytes = (ROOT/'LFMN/model/common.py').read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(common_bytes).hexdigest() != '81fc84ac6e8e19991e8129d6825d51be6f1720ba418f344ea0f4af9d8eef43c6':
            raise RuntimeError('Original model/common.py SHA256 mismatch')
    else:
        path = ROOT / 'LFMN/model/lfmn.py'
    # Explicit file prevents import preference from silently substituting the binary/source.
    spec = importlib.util.spec_from_file_location('model.lfmn', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = SimpleNamespace(scale=[4], n_colors=3, rgb_range=255, n_feats=48,
                           n_resblocks=8, res_scale=1, pre_train='', precision='single')
    if hasattr(module,'make_model'):
        net = module.make_model(args)
    else:
        import inspect
        net = None
        for name in ('LFMN','Model','Net'):
            cls = getattr(module,name,None)
            if inspect.isclass(cls) and issubclass(cls,torch.nn.Module):
                net = cls(args)
                break
        if net is None:
            raise RuntimeError('Official factory unavailable; inspect binary API, do not substitute architecture')
    print('MODEL source=%s parameters=%d random initialization; pretrained=None' %
          (path,sum(p.numel() for p in net.parameters())), flush=True)
    return net


def make_train_dataset(pairs):
    import torch
    from data import common
    import imageio.v2 as imageio

    class TrainDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8000  # official repeat = (8 * 1000) // 800 = 10

        def __getitem__(self, index):
            lp,hp = pairs[index % 800]
            lr,hr = imageio.imread(lp),imageio.imread(hp)
            lr,hr = common.get_patch(lr,hr,patch_size=256,scale=4)
            lr,hr = common.augment(lr,hr)
            return tuple(common.np2Tensor(lr,hr,rgb_range=255))

    return TrainDataset()


def metrics(sr, hr):
    import torch
    import numpy as np
    import cv2
    sr = sr.clamp(0,255).round()
    diff = (sr-hr)/255
    convert = diff.new_tensor([65.738,129.057,25.064]).view(1,3,1,1)/256
    diff = (diff * convert).sum(1)
    mse = diff[...,4:-4,4:-4].pow(2).mean().item()
    psnr = -10*math.log10(max(mse,1e-30))
    def y(t):
        a = t.detach().float().clamp(0,255).round()[0,:,4:-4,4:-4]
        a = a.cpu().numpy().astype(np.float64).transpose(1,2,0)
        return np.dot(a/255,[65.481,128.553,24.966])+16
    a,b = y(sr),y(hr)
    w = cv2.getGaussianKernel(11,1.5)
    w = np.outer(w,w.transpose())
    filt = lambda t: cv2.filter2D(t,-1,w)[5:-5,5:-5]
    ma,mb = filt(a),filt(b)
    va,vb = filt(a*a)-ma*ma, filt(b*b)-mb*mb
    cov = filt(a*b)-ma*mb
    c1,c2 = (.01*255)**2,(.03*255)**2
    ssim = (((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))).mean()
    if not math.isfinite(psnr) or not math.isfinite(ssim):
        raise RuntimeError('Nonfinite validation metric')
    return psnr,float(ssim)


def evaluate(net, groups, device, limit=0):
    import torch
    import imageio.v2 as imageio
    from data import common
    result, per_image = {},[]
    net.eval()
    with torch.inference_mode():
        for name in SETS:
            values = []
            pairs = groups[name][:limit] if limit else groups[name]
            for i,(lp,hp) in enumerate(pairs,1):
                lr,hr = common.set_channel(imageio.imread(lp),imageio.imread(hp),n_channels=3)
                lr,hr = common.np2Tensor(lr,hr,rgb_range=255)
                lr,hr = lr.unsqueeze(0).to(device),hr.unsqueeze(0).to(device)
                hr = hr[...,:lr.shape[-2]*4,:lr.shape[-1]*4]
                sr = net(lr)  # one full-image call; no flips, chops, resizing, or TTA
                if sr.shape != hr.shape or not torch.isfinite(sr).all():
                    raise RuntimeError('Invalid model output for %s' % lp)
                p,s = metrics(sr,hr)
                values.append((p,s))
                per_image.append(dict(dataset=name,image=Path(hp).name,psnr=p,ssim=s))
                if i % 10 == 0 or i == len(pairs):
                    print('VALIDATE %s %d/%d' % (name,i,len(pairs)),flush=True)
            result[name] = dict(psnr=sum(v[0] for v in values)/len(values),
                                ssim=sum(v[1] for v in values)/len(values),count=len(values))
    return result,per_image


def rng_state(generator):
    import torch
    import numpy as np
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state,generator):
    import torch
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
    generator.set_state(state['loader'])


def atomic_torch(path,value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        torch.save(value,f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def load_checkpoint(path):
    import torch
    # Only locally generated, trusted full-state checkpoints, never external pretrained files.
    return torch.load(path,map_location='cpu',weights_only=False)


def write_history(run, history):
    path = Path(run)/'metrics.csv'
    columns = ['epoch','global_step','train_loss','learning_rate','next_learning_rate',
               'train_seconds','validation_seconds','epoch_seconds','elapsed_seconds','selection_psnr','is_best']
    for name in SETS:
        columns += [name+'_'+k for k in ('psnr','ssim','best_psnr','best_psnr_epoch','best_ssim','best_ssim_epoch')]
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows({k:r.get(k,'') for k in columns} for r in history)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)  # rebuild from committed checkpoint; crash cannot create duplicate epochs


def summary(run, state):
    history = state['history']
    if not history:
        return
    best = state['best_snapshot']
    report = dict(complete=state['epoch']==1000,engineering_smoke=state['protocol']['smoke'],
                  best_epoch=best['epoch'],final_epoch=state['epoch'],
                  selected_checkpoint_metrics=best['metrics'],final_metrics=state['metrics'],
                  independent_dataset_bests=state['bests'],elapsed_seconds=state['elapsed'],
                  paper_x4=PAPER,protocol=state['protocol'],
                  note='Paper TTA setting and exact training batch/iteration count unspecified; '
                       'official demo enables x8; this run never uses x8. Best is one epoch, never spliced.')
    atomic_json(Path(run)/'summary.json',report)
    lines = ['# LFMN ×4 从零训练结果', '', '统一 best Epoch: %d；final Epoch: %d；累计训练+验证 %.2f 小时。' %
             (best['epoch'],state['epoch'],state['elapsed']/3600), '',
             '|数据集|统一 best PSNR/SSIM|final PSNR/SSIM|论文|best ΔPSNR|独立 PSNR best Epoch|',
             '|---|---|---|---|---|---|']
    for name in SETS:
        b,f,p = best['metrics'][name],state['metrics'][name],PAPER[name]
        lines.append('|%s|%.6f/%.6f|%.6f/%.6f|%.2f/%.4f|%+.6f|%d|' %
                     (name,b['psnr'],b['ssim'],f['psnr'],f['ssim'],p[0],p[1],
                      b['psnr']-p[0],state['bests'][name]['psnr_epoch']))
    lines += ['',report['note'], '', '烟雾测试结果仅验证工程链路，不是论文性能复现。' if report['engineering_smoke']
              else '五数据集均参与统一选模；这些指标不是未参与选模的盲测结果。']
    (Path(run)/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(14,8))
    for ax,name in zip(axes.flat,SETS):
        ax.plot([r['epoch'] for r in history],[r[name+'_psnr'] for r in history])
        ax.axvline(best['epoch'],color='gray',linestyle='--')
        ax.set(title=name,xlabel='Epoch',ylabel='PSNR (dB)')
        ax.grid(True)
    axes.flat[-1].axis('off')
    fig.tight_layout()
    fig.savefig(Path(run)/'psnr_curves.png',dpi=160)
    plt.close(fig)


def train(args,log):
    import torch
    import numpy as np
    if int(os.environ.get('WORLD_SIZE','1')) != 1:
        raise RuntimeError('Only single GPU/process supported; DDP launch rejected')
    if args.backend != 'official' and not args.smoke:
        raise RuntimeError('Python reimplementation is diagnostic-only, forbidden for formal baseline')
    groups,manifest = scan_data(args.data)
    proto = protocol(args,manifest)
    fingerprint = hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for selected device')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    net = build_model(args.backend).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    # Dataset class is local: Linux fork only; main process is used for Windows diagnostics.
    worker_options = {'multiprocessing_context':'fork'} if args.workers else {}
    loader = torch.utils.data.DataLoader(make_train_dataset(groups['DIV2K']),batch_size=8,
                shuffle=True,num_workers=args.workers,pin_memory=device.type=='cuda',
                generator=generator,persistent_workers=False,**worker_options)
    opt = torch.optim.Adam(net.parameters(),lr=2e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,[200,400,600,800,1000],gamma=.5)
    checkpoint_dir = log.path/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    state = dict(epoch=0,global_step=0,history=[],bests={},best_snapshot=None,elapsed=0,
                 protocol=proto,fingerprint=fingerprint)
    if args.resume:
        state = load_checkpoint(checkpoint_dir/'latest.pt')
        if state['fingerprint'] != fingerprint:
            raise RuntimeError('Resume protocol/data/source mismatch; refusing incompatible baseline')
        net.load_state_dict(state['model'],strict=True)
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'],generator)
        print('RESUME checkpoint=%s SHA256=%s start_epoch=%d optimizer/scheduler/RNG/loader restored' %
              (checkpoint_dir/'latest.pt',sha(checkpoint_dir/'latest.pt'),state['epoch']+1),flush=True)
        if state['best_snapshot']:
            atomic_torch(checkpoint_dir/'best.pt',state['best_snapshot'])
        write_history(log.path,state['history'])
    atomic_json(log.path/'data_manifest.json',manifest)
    log.config.update(proto)
    log.config.update(git_commit=git('rev-parse','HEAD'),git_dirty=bool(git('status','--porcelain')),
                      git_diff_summary=git('diff','--stat'),command=sys.argv,cwd=os.getcwd(),
                      gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                      device=str(device),world_size=1,torch_version=torch.__version__,
                      python_version=sys.version,protocol_fingerprint=fingerprint,
                      resume_checkpoint=str(checkpoint_dir/'latest.pt') if args.resume else None,
                      resume_start_epoch=state['epoch']+1,preflight_receipt=args.preflight_receipt)
    log.save()
    write_history(log.path,state['history'])
    print(json.dumps(log.config,ensure_ascii=False,indent=2),flush=True)
    if not args.smoke:
        if not args.preflight_receipt:
            raise RuntimeError('Formal training requires matching passed server preflight receipt')
        receipt = json.loads(Path(args.preflight_receipt).read_text(encoding='utf-8'))
        receipt_proto = dict(proto)
        receipt_proto.update(smoke=True,smoke_steps=0,smoke_eval_limit=0)
        receipt_fp = hashlib.sha256(json.dumps(receipt_proto,sort_keys=True).encode()).hexdigest()
        if receipt.get('status') != 'passed' or receipt.get('fingerprint') != receipt_fp or \
                receipt.get('gpu_name') != log.config['gpu_name'] or receipt.get('torch_version') != torch.__version__:
            raise RuntimeError('Preflight receipt incompatible with this data/source/protocol/GPU/PyTorch')
    last = args.stop_after or (2 if args.smoke else 1000)
    if not 1 <= last <= (2 if args.smoke else 1000):
        raise ValueError('stop-after outside fixed training budget')
    for epoch in range(state['epoch']+1,last+1):
        start = time.perf_counter()
        net.train()
        lr = opt.param_groups[0]['lr']
        losses = []
        steps = args.smoke_steps if args.smoke and args.smoke_steps else 1000
        print('EPOCH %d/1000 lr=%.10g steps=%d batch=8 HRpatch=256' % (epoch,lr,steps),flush=True)
        for step,(x,y) in enumerate(loader,1):
            opt.zero_grad(set_to_none=True)
            output = net(x.to(device,non_blocking=True))
            loss = torch.nn.functional.l1_loss(output,y.to(device,non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            if (step == 1 or step % 100 == 0) and not all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()):
                raise RuntimeError('Nonfinite gradient')
            opt.step()
            losses.append(loss.item())
            if step % 100 == 0 or step == steps:
                print('TRAIN epoch=%d iteration=%d/%d loss=%.8f LR=%.10g' %
                      (epoch,step,steps,sum(losses)/len(losses),lr),flush=True)
            if step == steps:
                break
        if len(losses) != steps:
            raise RuntimeError('Unexpected number of optimizer updates')
        scheduler.step()
        training_seconds = time.perf_counter()-start
        validation_start = time.perf_counter()
        scores,images = evaluate(net,groups,device,args.smoke_eval_limit if args.smoke else 0)
        validation_seconds = time.perf_counter()-validation_start
        elapsed = state['elapsed']+time.perf_counter()-start
        bests = copy.deepcopy(state['bests'])
        for name,v in scores.items():
            b = bests.setdefault(name,dict(psnr=-1e30,ssim=-1e30))
            for metric in ['psnr','ssim']:
                if v[metric] > b[metric]:
                    b[metric],b[metric+'_epoch'] = v[metric],epoch
            print('%s current %.8f/%.8f best PSNR %.8f@%d best SSIM %.8f@%d' %
                  (name,v['psnr'],v['ssim'],b['psnr'],b['psnr_epoch'],b['ssim'],b['ssim_epoch']),flush=True)
        selection = sum(v['psnr'] for v in scores.values())/5
        is_best = state['best_snapshot'] is None or selection > state['best_snapshot']['selection_psnr']
        row = dict(epoch=epoch,global_step=state['global_step']+steps,train_loss=sum(losses)/len(losses),
                   learning_rate=lr,next_learning_rate=opt.param_groups[0]['lr'],train_seconds=training_seconds,
                   validation_seconds=validation_seconds,epoch_seconds=training_seconds+validation_seconds,
                   elapsed_seconds=elapsed,selection_psnr=selection,is_best=is_best)
        for name in SETS:
            row.update({name+'_'+k:v for k,v in scores[name].items() if k != 'count'})
            row.update({name+'_best_'+k:v for k,v in bests[name].items()})
        payload = dict(epoch=epoch,global_step=row['global_step'],model=net.state_dict(),
                       optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(generator),
                       history=state['history']+[row],bests=bests,metrics=scores,per_image_metrics=images,
                       elapsed=elapsed,selection_psnr=selection,protocol=proto,fingerprint=fingerprint)
        # Full best snapshot includes optimizer/scheduler/RNG. No recursive snapshots.
        payload['best_snapshot'] = copy.deepcopy(payload) if is_best else state['best_snapshot']
        atomic_torch(checkpoint_dir/'latest.pt',payload)
        if is_best:
            atomic_torch(checkpoint_dir/'best.pt',payload['best_snapshot'])
        state = payload
        write_history(log.path,state['history'])
        atomic_json(log.path/('per_image_epoch_%04d.json' % epoch),images)
        summary(log.path,state)
        log.config.update(last_completed_epoch=epoch,final_metrics=scores,best_metrics=state['best_snapshot']['metrics'],
                          best_epoch=state['best_snapshot']['epoch'],elapsed_seconds=elapsed)
        log.save()
        print('CHECKPOINT committed epoch=%d global_step=%d best_epoch=%d elapsed=%.2fs' %
              (epoch,row['global_step'],state['best_snapshot']['epoch'],elapsed),flush=True)
    if state['epoch'] == (2 if args.smoke else 1000):
        atomic_torch(checkpoint_dir/'final.pt',state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',default=str(ROOT/'experiment/all_runs'))
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--backend',choices=['official','python'],default='official')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-steps',type=int,default=0,help='0=full 1000 updates; smoke diagnostics only')
    p.add_argument('--smoke-eval-limit',type=int,default=0,help='0=complete five datasets; diagnostics only')
    p.add_argument('--stop-after',type=int,default=0,help='Stop at committed epoch, preserving original schedule')
    p.add_argument('--resume',help='Own run directory; always restores committed latest.pt')
    p.add_argument('--preflight-receipt')
    p.add_argument('--check-data',action='store_true')
    args = p.parse_args()
    if args.smoke_steps < 0 or args.smoke_steps > 1000 or args.smoke_eval_limit < 0 or args.workers < 0:
        p.error('Invalid diagnostic limits')
    if not args.smoke and (args.smoke_steps or args.smoke_eval_limit):
        p.error('Diagnostic limits are forbidden for formal training')
    if platform.system() == 'Windows' and args.workers:
        p.error('Windows diagnostic mode requires --workers 0')
    log = RunLog(args.output,args.seed,args.resume,config={'experiment_name':'LFMN_original_baseline'})
    try:
        if args.check_data:
            _,manifest = scan_data(args.data)
            atomic_json(log.path/'data_manifest.json',manifest)
        else:
            train(args,log)
    except BaseException as error:
        log.finish(error)
        raise SystemExit(130 if isinstance(error,KeyboardInterrupt) else 1)
    else:
        log.finish()


if __name__ == '__main__':
    main()
