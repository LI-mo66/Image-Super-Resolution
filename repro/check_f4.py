"""F4 engineering checks only; no baseline optimizer steps."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--internal', action='store_true')
    args = parser.parse_args()
    if not args.internal or not os.environ.get('LFMN_MANAGED_RUN'):
        raise RuntimeError('Use run_f4_screen_server.py --check-only for captured checks')
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from model.lfmn import Net as B0
    from model.lfmnf4 import Net
    from model import Model
    torch.set_num_threads(4)
    torch.manual_seed(1)
    baseline = B0(scale=4).eval()
    state = torch.get_rng_state().clone()
    torch.manual_seed(1)
    candidate = Net(scale=4).eval()
    assert torch.equal(state, torch.get_rng_state()), 'shared RNG changed'
    common = {k: v for k, v in candidate.state_dict().items() if 'residual_gates.' not in k}
    assert set(common) == set(baseline.state_dict())
    probe = Net(scale=4)
    incompatible = probe.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {k for k in probe.state_dict() if 'residual_gates.' in k}
    assert not any('current_correction' in k for k in candidate.state_dict())
    assert all(torch.equal(v, common[k]) for k, v in baseline.state_dict().items())
    assert sum(p.numel() for p in candidate.parameters()) == 766859
    baseline, candidate = baseline.to(args.device), candidate.to(args.device)
    with torch.no_grad():
        for shape in [(1, 3, 32, 32), (2, 3, 33, 47), (1, 3, 48, 65)]:
            x = torch.rand(shape, device=args.device) * 255
            expected, actual = baseline(x), candidate(x)
            assert torch.equal(expected, actual), (shape, (expected-actual).abs().max())
            assert torch.isfinite(actual).all()
    def read(path):
        return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2,0,1).float()
    pairs = []
    for index in [1, 2, 3, 4]:
        base = args.data_root / 'DIV2K'
        pairs.append((read(base/'DIV2K_train_LR_bicubic'/'X4'/f'{index:04d}x4.png')[:, :64, :64],
                      read(base/'DIV2K_train_HR'/f'{index:04d}.png')[:, :256, :256]))
    x = torch.stack([p[0] for p in pairs]).to(args.device)
    hr = torch.stack([p[1] for p in pairs]).to(args.device)
    candidate.train()
    counts = [0] * 8
    handles = [gate.register_forward_hook(lambda m, a, o, i=i: counts.__setitem__(i, counts[i]+1))
               for i, gate in enumerate(candidate.residual_gates)]
    optimizer = torch.optim.Adam(candidate.parameters(), lr=2e-4)
    new = {k: p for k,p in candidate.named_parameters() if 'residual_gates.' in k}
    previous = {k: p.detach().clone() for k,p in new.items()}
    losses=[]
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(candidate(x), hr)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.values())
        active = [k for k,p in new.items() if p.grad.abs().max() > 0]
        if step == 0:
            assert all('.3.' in k for k in active) and len(active)==16, active
        if step == 2:
            assert len(active)==len(new), active
        optimizer.step()
        losses.append(loss.item())
        print('ENGINEERING_STEP', step+1, 'L1', loss.item(), 'active_new_tensors',len(active),flush=True)
    assert all(not torch.equal(p,previous[k]) for k,p in new.items())
    assert counts == [3]*8, counts
    for handle in handles: handle.remove()
    candidate.eval()
    # Trained nonzero gates must also restore B0 mapping when bypassed.
    baseline.load_state_dict({k:v for k,v in candidate.state_dict().items() if 'residual_gates.' not in k}, strict=True)
    for gate in candidate.residual_gates: gate.enabled=False
    with torch.no_grad(): assert torch.equal(baseline(x[:1]),candidate(x[:1]))
    for gate in candidate.residual_gates: gate.enabled=True
    with torch.inference_mode():
        expected=candidate(x[:1])
        buffer=io.BytesIO();torch.save(candidate.state_dict(),buffer);buffer.seek(0)
        restored=Net(scale=4).to(args.device).eval()
        restored.load_state_dict(torch.load(buffer,map_location=args.device,weights_only=True),strict=True)
        assert torch.equal(expected,restored(x[:1]))
        wrapper=Model.__new__(Model);torch.nn.Module.__init__(wrapper)
        wrapper.model=restored;wrapper.chop=False;wrapper.self_ensemble=False
        wrapper.precision='single';wrapper.device=torch.device(args.device);wrapper.n_GPUs=1;wrapper.eval()
        count=[0];original=restored.forward
        def counted(t):count[0]+=1;return original(t)
        restored.forward=counted
        wrapper.forward_x8=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('x8 entered'))
        wrapper(x[:1],0)
        assert count[0]==1
    directory=Path(os.environ['LFMN_MANAGED_RUN'])
    with (directory/'metrics.csv').open('a',newline='',encoding='utf-8') as stream:
        for index, value in enumerate(losses,1):
            csv.writer(stream).writerow([index,2e-4,value,'','','','',''])
    (directory/'checks.json').write_text(json.dumps(dict(status='PASS',parameters=766859,
        baseline_optimizer_steps=0,candidate_engineering_steps=3,ensemble_OFF_forward_calls=1,
        scope='engineering only, not PSNR evidence'),indent=2),encoding='utf-8')
    print('F4 CHECK PASS: zero mapping/RNG/shapes/real batch4/gradient startup/update/strict reload/actual OFF',flush=True)


if __name__=='__main__':main()
