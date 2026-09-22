#!/usr/bin/env python3
"""N12/SRPRv2 matched Conv2d profile."""
import argparse, json, statistics, time, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as BaselineNet
from model.lfmnsrprv2 import Net as SRPRv2Net

def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
def count(model): return sum(p.numel() for p in model.parameters())
def profile(model, sample, warmup, repeats, device):
    hooks=[]; macs=0
    def hook(layer, inputs, output):
        nonlocal macs
        out=output[0] if isinstance(output, tuple) else output
        b,c,h,w=out.shape
        macs += b*c*h*w*(layer.in_channels//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1]
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d): hooks.append(m.register_forward_hook(hook))
    with torch.inference_mode():
        for _ in range(warmup): model(sample)
        sync(device)
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        times=[]
        for _ in range(repeats):
            sync(device); t=time.perf_counter(); model(sample); sync(device)
            times.append((time.perf_counter()-t)*1000)
    for h in hooks: h.remove()
    peak={'allocated_mib':None,'reserved_mib':None}
    if device.type=='cuda': peak={'allocated_mib':torch.cuda.max_memory_allocated(device)/2**20,'reserved_mib':torch.cuda.max_memory_reserved(device)/2**20}
    ordered=sorted(times)
    return {'parameters':count(model),'macs':macs,'flops':2*macs,'mac_definition':'1 MAC = multiply-accumulate; FLOPs = 2 * MAC','median_ms':statistics.median(times),'p95_ms':ordered[max(0,int(.95*len(ordered))-1)],**peak}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--size',type=int,default=64); p.add_argument('--warmup',type=int,default=10); p.add_argument('--repeats',type=int,default=50); p.add_argument('--output',type=Path); a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); sample=torch.rand(1,3,a.size,a.size,device=device)
    rows=[]
    for name, cls in [('LFMN',BaselineNet),('SRPRv2',SRPRv2Net)]:
        model=cls(scale=4).to(device).eval(); row={'model':name,'input_shape':[1,3,a.size,a.size],'device':str(device),'precision':'float32',**profile(model,sample,a.warmup,a.repeats,device)}; rows.append(row); del model
        if device.type=='cuda': torch.cuda.empty_cache()
    payload={'profiler':'same Conv2d hook for both models','rows':rows}; text=json.dumps(payload,indent=2); print(text)
    if a.output: a.output.write_text(text+'\n',encoding='utf-8')
if __name__=='__main__': main()
