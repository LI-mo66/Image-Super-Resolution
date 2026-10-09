"""Absolute F3 OFF resources; no baseline training or relative efficiency claim."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))

def main():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from model.lfmnf3 import Net
    group=Path(sys.argv[1]).resolve()
    path=group/'F3_x4_seed1/model/model_20.pt'
    n=Net(scale=4).cuda().eval()
    n.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True)
    result=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,precision='FP32',
        batch=1,self_ensemble=False,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in n.parameters()),warmup=10,repeats=50,
        flops_convention='supported PyTorch FlopCounter operations, multiply-add=2; excludes LN/sort/index/pooling/interpolation/elementwise',
        relative_efficiency='UNKNOWN_BASELINE_PENDING',results={})
    with torch.inference_mode():
        for h,w in [(64,64),(128,128),(96,160)]:
            torch.cuda.empty_cache();x=torch.zeros(1,3,h,w,device='cuda')
            for _ in range(10):n(x)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(50):
                start=time.perf_counter();y=n(x);torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
                if not torch.isfinite(y).all():raise FloatingPointError('profile output')
                del y
            peak=torch.cuda.max_memory_allocated()/1024**2
            reserved=torch.cuda.max_memory_reserved()/1024**2
            with FlopCounterMode(display=False) as fc:n(x)
            result['results'][f'LR{h}x{w}']=dict(counted_flops=fc.get_total_flops(),
                median_ms=statistics.median(times),p90_ms=sorted(times)[44],
                peak_allocated_mib=peak,peak_reserved_mib=reserved,raw_ms=times)
    target=group/'resources.json'
    if target.exists():
        target=group/('resources_'+str(time.time_ns())+'.json')
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('F3 resources:',target,flush=True)

if __name__=='__main__':main()
