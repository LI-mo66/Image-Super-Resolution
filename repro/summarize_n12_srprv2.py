#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch


def load_curve(run, name):
    value = torch.load(run / name, map_location='cpu', weights_only=True).float()
    return value[:, 0, 0].numpy() if value.ndim == 3 else value.flatten().numpy()

def main():
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); p.add_argument('--baseline-reference',type=Path,required=True); a=p.parse_args()
    cand=a.run/'srprv2'; b0=a.baseline_reference
    cfg=dict(line.split(': ',1) for line in (cand/'config.txt').read_text().splitlines() if ': ' in line)
    expected={'model':'LFMNSRPRV2','epochs':'20','data_range':'1-800/801-900','scale':'[4]','patch_size':'256','batch_size':'4','seed':'1','lr':'0.0002','scheduler':'cosine','scheduler_t_max':'150','eta_min':'1e-06','loss':'1*L1','pre_train':''}
    for k,v in expected.items():
        if cfg.get(k)!=v: raise ValueError(f'{k}: expected {v!r}, got {cfg.get(k)!r}')
    cp,cs=load_curve(cand,'psnr_log.pt'),load_curve(cand,'ssim_log.pt'); bp,bs=load_curve(b0,'psnr_log.pt'),load_curve(b0,'ssim_log.pt')
    print('| epoch | SRPRv2 PSNR | B0 PSNR | delta | SRPRv2 SSIM | B0 SSIM | delta |')
    print('|---:|---:|---:|---:|---:|---:|---:|')
    for i in range(min(20,len(cp),len(bp))): print(f'| {i+1} | {cp[i]:.6f} | {bp[i]:.6f} | {cp[i]-bp[i]:+.6f} | {cs[i]:.6f} | {bs[i]:.6f} | {cs[i]-bs[i]:+.6f} |')
    print(f'\nfinal_delta={cp[19]-bp[19]:+.6f}')
    print(f'last5_delta={(cp[15:20]-bp[15:20]).mean():+.6f}')
    print(f'positive_epoch_ratio={((cp[:20]-bp[:20])>0).mean():.3f}')

if __name__=='__main__': main()
