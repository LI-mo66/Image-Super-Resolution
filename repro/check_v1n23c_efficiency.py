"""Frozen same-device inference benchmark; no V1/candidate optimization."""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))


def main():
    import torch
    from model.lfmnsrprv2 import Net as V1
    ap = argparse.ArgumentParser()
    ap.add_argument('--group',required=True,choices=('n23c',))
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to((ROOT/'experiment/all_runs').resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(1)
    x = torch.rand(1,3,128,128,device='cuda')*255
    rows = []
    for label, factory in [('v1',V1),(args.group,importlib.import_module('model.lfmn_v1'+args.group).Net)]:
        torch.manual_seed(1)
        net = factory(scale=4).cuda().eval()
        params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            for _ in range(2):
                net(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                y = net(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-start)
                if y.shape != (1,3,512,512) or not torch.isfinite(y).all():
                    raise RuntimeError('Invalid benchmark forward')
                del y
            rows.append(dict(model=label,params=params,median_seconds=statistics.median(times),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        del net
        torch.cuda.empty_cache()
    report = dict(purpose='FROZEN_RANDOM_INIT_INFERENCE_ONLY',shape=[1,3,128,128],warmup=2,repeats=5,
        torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),precision='FP32 default backends',
        rows=rows,latency_ratio=rows[1]['median_seconds']/rows[0]['median_seconds'],
        flops='UNMEASURED',warning='Tile/device-specific; not trained/full-image deployment guarantee')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
