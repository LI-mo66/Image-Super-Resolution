#!/usr/bin/env python3
"""Validate A1 anchored prior-evolution wiring without training."""
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net as Baseline
from model.lfmnpriorproxy import Net


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = Baseline(scale=4).to(device).eval()
    net = Net(scale=4).to(device).eval()
    base_state = base.state_dict()
    load_result = net.load_state_dict(base_state, strict=False)
    require(
        set(load_result.missing_keys) == {
            key for key in net.state_dict() if key.startswith('prior_update.')
        } and not load_result.unexpected_keys,
        'baseline-to-candidate checkpoint mapping mismatch',
    )
    require(sum(p.numel() for p in net.prior_update.parameters()) == 1016,
            'unexpected prior-update parameter count')

    x = torch.rand(1, 3, 32, 32, device=device) * 255
    with torch.no_grad():
        base_out = base(x)
        net.update_enabled = False
        disabled_out = net(x)
    require(torch.equal(base_out, disabled_out), 'disabled candidate differs from baseline')

    captured = []
    handles = [module.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    ) for module in net.sfmls]
    net.update_enabled = True
    with torch.no_grad():
        enabled_out = net(x)
    for handle in handles:
        handle.remove()
    require(len(captured) == 8, 'unexpected number of SFML calls')
    require(all(torch.equal(captured[0], value) for value in captured[:4]),
            'early stages do not share fs0')
    require(all(torch.equal(captured[4], value) for value in captured[4:]),
            'late stages do not share fs1')
    require(torch.equal(captured[0], captured[4]),
            'zero-start update is not initially anchored')
    require(torch.equal(disabled_out, enabled_out),
            'zero-start enabled path differs from disabled baseline')

    with torch.no_grad():
        net.prior_update.project.weight.normal_(mean=0.0, std=1e-3)
        net.prior_update.project.bias.normal_(mean=0.0, std=1e-3)
    captured.clear()
    with torch.no_grad():
        enabled_out = net(x)
    handles = [module.register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone())
    ) for module in net.sfmls]
    with torch.no_grad():
        enabled_out = net(x)
    for handle in handles:
        handle.remove()
    require(not torch.equal(captured[0], captured[4]),
            'nonzero update does not reach late stages')
    require(not torch.equal(disabled_out, enabled_out),
            'nonzero update does not affect output')

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)
    target = torch.rand_like(enabled_out)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.l1_loss(net(x), target)
        loss.backward()
        for name, parameter in net.prior_update.named_parameters():
            require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                    f'missing/nonfinite gradient: {name}')
            if step == 0 and name.startswith('project.'):
                require(parameter.grad.abs().sum().item() > 0,
                        f'zero first-step project gradient: {name}')
        optimizer.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'model.pt'
        torch.save(net.state_dict(), path)
        restored = Net(scale=4).to(device).eval()
        restored.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            require(torch.equal(net(x), restored(x)), 'reload changed output')

    print('device:', device)
    print('added parameters: 1016')
    print('disabled identity, stage routing, causal dependency, gradients, reload: OK')
    print('Prior-proxy validation passed (not a performance result)')


if __name__ == '__main__':
    main()
