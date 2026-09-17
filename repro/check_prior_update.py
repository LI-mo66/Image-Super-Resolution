#!/usr/bin/env python3
"""Server-only identity, gradient, causal-path and serialization checks."""
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as Baseline
from model.lfmnpriorupdate import Net


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base, net = Baseline(scale=4), Net(scale=4)
    weights = torch.load(ROOT / 'LFMN/model/scale4_model_939.pt',
                         map_location='cpu', weights_only=True)
    base.load_state_dict(weights, strict=True)
    result = net.load_state_dict(weights, strict=False)
    require(set(result.missing_keys) == {
        k for k in net.state_dict() if k.startswith('prior_update.')
    } and not result.unexpected_keys, 'checkpoint mismatch')
    added = sum(p.numel() for p in net.prior_update.parameters())
    require(added == 1016, 'unexpected parameter count')
    base, net = base.to(device).eval(), net.to(device).eval()
    x = torch.rand(1, 3, 32, 32, device=device) * 255
    with torch.no_grad():
        require(torch.equal(base(x), net(x)), 'initial output not identical')
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    target = torch.rand(1, 3, 128, 128, device=device) * 255
    # Eval mode retains autograd while keeping TAB centroid buffers fixed.
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        F.l1_loss(net(x), target).backward()
        names = ('project.weight',) if step == 0 else ('reduce.weight', 'spatial.weight')
        for name, p in net.prior_update.named_parameters():
            require(p.grad is not None and torch.isfinite(p.grad).all().item(),
                    'missing/nonfinite gradient: ' + name)
            if name in names:
                require(p.grad.abs().sum().item() > 0, 'zero gradient: ' + name)
        optimizer.step()

    priors = []
    handles = [m.register_forward_pre_hook(
        lambda module, inputs: priors.append(inputs[0].detach().clone())
    ) for m in net.sfmls]
    with torch.no_grad():
        updated_output = net(x)
    for h in handles:
        h.remove()
    require(all(torch.equal(priors[0], p) for p in priors[:4]), 'early prior changed')
    require(all(torch.equal(priors[4], p) for p in priors[4:]), 'late prior not shared')
    require(not torch.equal(priors[0], priors[4]), 'update not reaching late stages')
    handle = net.prior_update.register_forward_pre_hook(
        lambda module, inputs: (inputs[0], inputs[1] + 1.0)
    )
    with torch.no_grad():
        perturbed = net(x)
    handle.remove()
    require(not torch.equal(updated_output, perturbed), 'stage-four state has no effect')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'model.pt'
        torch.save(net.state_dict(), path)
        restored = Net(scale=4).to(device).eval()
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            require(torch.equal(updated_output, restored(x)), 'reload changed output')
    print('device:', device)
    print('initial max output difference: 0.0')
    print('added parameters:', added)
    print('first/second-step gradients, stage-4 dependency, stage-5–8 routing, reload: OK')
    print('Prior-update validation passed (not a performance result)')


if __name__ == '__main__':
    main()
