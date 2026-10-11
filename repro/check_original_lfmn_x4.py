"""Meaningful engineering checks, never performance claims."""
import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys
import torch
import numpy as np
from original_lfmn_x4 import ROOT, metrics, rng_state, restore_rng, atomic_torch, load_checkpoint, write_history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--infrastructure-only',action='store_true')
    args = p.parse_args()
    sys.path.insert(0,str(ROOT/'LFMN'))
    import utility
    from types import SimpleNamespace
    loader = SimpleNamespace(dataset=SimpleNamespace(benchmark=True))
    torch.manual_seed(20)
    for shape in [(1,3,45,51),(1,3,256,256)]:
        sr = torch.rand(shape)*280-10
        hr = (torch.rand(shape)*255).round()
        p1,s1 = metrics(sr,hr)
        q = utility.quantize(sr,255)
        assert p1 == utility.calc_psnr(q,hr,4,255,loader)
        assert abs(s1-utility.calc_ssim(q,hr,4,255,loader)) < 1e-12
    print('PASS exact official PSNR / SSIM equivalence (odd rectangular and patch images)')
    # Check full RNG roundtrip and identical next optimizer update with a small probe.
    # This isolates infrastructure; it is not a replacement LFMN training run.
    root = ROOT/'experiment/baseline_checks/checkpoint_probe'
    root.mkdir(parents=True,exist_ok=True)
    generator = torch.Generator().manual_seed(7)
    net = torch.nn.Conv2d(3,3,1)
    opt = torch.optim.Adam(net.parameters(),lr=2e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt,[200,400,600,800,1000],gamma=.5)
    def step():
        x = torch.rand(1,3,8,8)
        y = torch.rand_like(x)
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(net(x),y)
        loss.backward()
        opt.step()
        sched.step()
        return loss.item(),copy.deepcopy(net.state_dict())
    step()
    saved = dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(generator))
    atomic_torch(root/'probe.pt',saved)
    expected = step()
    expected_draws = (random.random(),np.random.rand(),torch.rand(2),torch.rand(2,generator=generator))
    restored = load_checkpoint(root/'probe.pt')
    net.load_state_dict(restored['model'],strict=True)
    opt.load_state_dict(restored['optimizer'])
    sched.load_state_dict(restored['scheduler'])
    restore_rng(restored['rng'],generator)
    actual = step()
    actual_draws = (random.random(),np.random.rand(),torch.rand(2),torch.rand(2,generator=generator))
    assert actual[0] == expected[0]
    assert all(torch.equal(actual[1][k],expected[1][k]) for k in expected[1])
    assert actual_draws[:2] == expected_draws[:2]
    assert all(torch.equal(a,b) for a,b in zip(actual_draws[2:],expected_draws[2:]))
    write_history(root,[dict(epoch=1,global_step=1000),dict(epoch=2,global_step=2000)])
    write_history(root,[dict(epoch=1,global_step=1000),dict(epoch=2,global_step=2000)])
    rows = list(csv.DictReader((root/'metrics.csv').open()))
    assert [r['epoch'] for r in rows] == ['1','2']
    print('PASS full-state reload produces identical next loss/update/RNG; idempotent CSV recovery')


if __name__ == '__main__':
    main()
