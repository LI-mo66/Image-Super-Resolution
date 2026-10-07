"""Bounded engineering checks only; this is not a PSNR screening run."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as Baseline
from utility import calc_ssim

ID = 'n22'
Candidate = importlib.import_module('model.lfmn_' + ID).Net

def pair(data, index, validation=False):
    prefix = 'valid' if validation else 'train'
    with Image.open(data / f'DIV2K_{prefix}_LR_bicubic/X4/{index:04d}x4.png') as im:
        a = np.array(im.convert('RGB'))
    with Image.open(data / f'DIV2K_{prefix}_HR/{index:04d}.png') as im:
        b = np.array(im.convert('RGB'))
    assert b.shape[:2] == (4*a.shape[0], 4*a.shape[1])
    y, x = (a.shape[0]-64)//2, (a.shape[1]-64)//2
    a, b = a[y:y+64,x:x+64], b[y*4:(y+64)*4,x*4:(x+64)*4]
    return tuple(torch.from_numpy(z.copy()).permute(2,0,1).float().unsqueeze(0) for z in (a,b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True, help='DIV2K directory')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(2122)
    baseline = Baseline(scale=4)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(2122)
    candidate = Candidate(scale=4)
    assert torch.equal(expected_rng, torch.get_rng_state()), 'Candidate consumes extra data RNG'
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    baseline.load_state_dict(state, strict=True)
    missing, unexpected = candidate.load_state_dict(state, strict=False)
    added = set(candidate.state_dict()) - set(baseline.state_dict())
    assert set(missing) == added and not unexpected, (missing, unexpected)
    assert all(candidate.state_dict()[k].shape == v.shape for k,v in state.items())
    baseline, candidate = baseline.to(device).eval(), candidate.to(device).eval()
    report = {'id':ID, 'device':device, 'torch':torch.__version__,
              'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'baseline_params':sum(p.numel() for p in baseline.parameters()),
              'candidate_params':sum(p.numel() for p in candidate.parameters()),
              'new_keys':sorted(added), 'shapes':[], 'performance_claim':False}
    with torch.inference_mode():
        for b,h,w in ((1,32,32),(2,33,47),(1,48,65)):
            x = torch.rand(b,3,h,w,device=device)*255
            expected = baseline(x)
            candidate.set_enabled(False)
            disabled = candidate(x)
            assert torch.equal(expected, disabled), 'Disabled path differs'
            candidate.set_enabled(True)
            actual = candidate(x)
            delta = (expected-actual).abs().max().item()
            assert actual.shape == (b,3,h*4,w*4) and torch.isfinite(actual).all()
            assert delta < 1e-4, delta
            report['shapes'].append({'input':[b,3,h,w],'zero_gain_max_error':delta})
        for name,p in candidate.named_parameters():
            if name.endswith(('inheritance_gain','prior_gain')):
                p.fill_(0.4)
        x = torch.rand(2,3,33,47,device=device)*255
        batch = candidate(x)
        singles = torch.cat([candidate(x[i:i+1]) for i in range(2)])
        batch_delta = (batch-singles).abs().max().item()
        assert torch.isfinite(batch).all() and torch.isfinite(singles).all()
        report['eval_batch_separation_max_error'] = batch_delta
        report['baseline_batch_separation_max_error'] = (baseline(x)-torch.cat([baseline(x[i:i+1]) for i in range(2)])).abs().max().item()
        report['anchors_initialized_for_batch_check'] = all(bool(block[0].initted.item()) for block in candidate.blocks)
        original_priors = candidate.detail_priors(x)
        changed = x.clone()
        changed[1] = 255 - changed[1]
        assert torch.equal(original_priors[0],candidate.detail_priors(changed)[0])
        report['prior_image_isolation'] = True
        report['active_effect_max_error'] = (batch-baseline(x)).abs().max().item()
        assert report['active_effect_max_error'] > 1e-5
        if ID == 'n22':
            for val in (0.,128.,255.):
                priors = candidate.detail_priors(torch.full((2,3,33,47),val,device=device))
                assert priors.shape == (2,4,33,47)
                assert torch.isfinite(priors).all() and priors.min() >= 0 and priors.max() <= 1
        if device == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                mixed = candidate(x)
            assert torch.isfinite(mixed).all()
            report['autocast_forward_finite'] = True
    # Reload initial weights, and reset candidate-only gains before smoke updates.
    candidate.load_state_dict(state, strict=False)
    with torch.no_grad():
        for name,p in candidate.named_parameters():
            if name.endswith(('inheritance_gain','prior_gain')):
                p.zero_()
    pairs = [pair(args.data,i) for i in (1,2)]
    lr = torch.cat([p[0] for p in pairs]).to(device)
    hr = torch.cat([p[1] for p in pairs]).to(device)
    opt = torch.optim.Adam(candidate.parameters(),lr=1e-4)
    new_params = {k:p for k,p in candidate.named_parameters() if k not in dict(baseline.named_parameters())}
    report['smoke_steps'] = []
    candidate.train()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        out = candidate(lr)
        loss = F.l1_loss(out,hr)
        assert torch.isfinite(out).all() and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in candidate.parameters())
        norms = {k:0. if p.grad is None else p.grad.abs().max().item() for k,p in new_params.items()}
        report['smoke_steps'].append({'step':step+1,'l1':loss.item(),'new_gradient_max':norms})
        if step == 0:
            assert all(v > 0 for k,v in norms.items() if k.endswith(('inheritance_gain','prior_gain')))
        if step == 2:
            assert all(v > 0 for v in norms.values()), 'Candidate parameter is inactive'
        opt.step()
    candidate.eval()
    vl, vh = pair(args.data,859,True)
    vl,vh = vl.to(device),vh.to(device)
    with torch.inference_mode():
        y = candidate(vl)
        mse = ((y[...,4:-4,4:-4]-vh[...,4:-4,4:-4])/255).square().mean().item()
        report['validation_smoke'] = {'image':859,'region':'center LR64, not full benchmark',
            'rgb_float_psnr_border4':float(-10*np.log10(mse)),
            'y_quantized_ssim_border4':float(calc_ssim(y,vh,4,255))}
    checkpoint = args.output / 'smoke.pt'
    torch.save(candidate.state_dict(),checkpoint)
    reloaded = Candidate(scale=4).to(device).eval()
    reloaded.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    with torch.inference_mode():
        z = reloaded(vl)
    assert torch.equal(y,z), 'Save/reload changes prediction'
    report['strict_reload_max_error'] = (y-z).abs().max().item()
    if device == 'cuda':
        reloaded.train()
        reloaded.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            amp_loss = F.l1_loss(reloaded(lr),hr)
        scaler = torch.amp.GradScaler('cuda', init_scale=128.)
        scaler.scale(amp_loss).backward()
        assert torch.isfinite(amp_loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in reloaded.parameters())
        report['autocast_scaled_backward_finite'] = True
    report['status'] = 'PASS'
    (args.output/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('new_keys','smoke_steps')},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
