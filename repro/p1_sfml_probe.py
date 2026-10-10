"""Fixed-checkpoint SFML diagnostics; hooks never train or update model state."""
import weakref

import torch

VARIANTS = ['p0', 'beta_minus', 'beta_plus', 'gamma_minus', 'gamma_plus']
_BASELINES = weakref.WeakKeyDictionary()
_ACTIVE = weakref.WeakKeyDictionary()
_EPS = 1e-12
_QUANTILE_MAX_VALUES = 262144


def _summary(x):
    x = x.detach().float().reshape(-1).cpu()
    if not x.numel():
        return {'count': 0}
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite SFML diagnostic tensor')
    # Quantile has an input-size limit, and full-image sorting is expensive.
    # Only quantiles are approximated; moments below use every region element.
    if x.numel() > _QUANTILE_MAX_VALUES:
        indices = torch.linspace(0, x.numel() - 1, _QUANTILE_MAX_VALUES,
                                 dtype=torch.float64).round().long()
        sample = x[indices]
    else:
        sample = x
    q = torch.quantile(sample, torch.tensor([0.05, 0.5, 0.95]))
    return dict(count=x.numel(), mean=x.mean().item(), std=x.std(unbiased=False).item(),
                rms=x.square().mean().sqrt().item(), p05=q[0].item(),
                p50=q[1].item(), p95=q[2].item(), abs_p95=torch.quantile(sample.abs(), 0.95).item(),
                quantile_sample_count=sample.numel(), quantiles_approximate=sample.numel() != x.numel(),
                quantile_sampling='deterministic evenly spaced flattened region/channel elements')


def _region_stats(beta, gamma, prev, mask):
    b, g, p = (t[0, :, mask].detach().float() for t in (beta, gamma, prev))
    if not p.numel():
        return {'pixels': 0}
    db = (b - 1) * p
    total = db + g
    bs, gs, ps = _summary(b), _summary(g), _summary(p)
    ds, ts = _summary(db), _summary(total)
    dot = (db * g).mean().item()
    return dict(pixels=int(mask.sum().item()), beta=bs, gamma=gs, prev=ps,
                beta_below_005=(b < .05).float().mean().item(),
                beta_above_095=(b > .95).float().mean().item(),
                multiplicative_delta=ds, total_delta=ts,
                multiplicative_to_prev_rms=ds['rms'] / (ps['rms'] + _EPS),
                gamma_to_prev_rms=gs['rms'] / (ps['rms'] + _EPS),
                total_to_prev_rms=ts['rms'] / (ps['rms'] + _EPS),
                beta_gamma_mean_dot=dot,
                beta_gamma_cosine=dot / (ds['rms'] * gs['rms'] + _EPS))


class Session:
    def __init__(self, net, lr, masks, variant):
        if variant not in VARIANTS:
            raise ValueError(f'Unknown P1 variant: {variant}')
        if any(m.training for m in net.modules()):
            raise ValueError('P1 requires every model module in eval mode')
        if net in _ACTIVE:
            raise RuntimeError('Close previous P1 session before installing another')
        if lr.ndim != 4 or lr.shape[:2] != (1, 3):
            raise ValueError('Expected B=1 RGB input')
        self.net, self.variant = net, variant
        self.lr = lr.detach().cpu().clone()
        self.handles, self.stages, self.prev = [], {}, None
        self.expected_fm = None
        self.closed = False
        self.routes_seen = set()
        self.buffers = {k: v.detach().cpu().clone() for k, v in net.named_buffers()}
        self.masks = {'all': torch.ones(lr.shape[-2:], dtype=torch.bool)}
        for key, mask in masks.items():
            if key == 'all' or mask.dtype != torch.bool or tuple(mask.shape) != tuple(lr.shape[-2:]):
                raise ValueError('Masks must be boolean H,W; all is reserved')
            self.masks[key] = mask.detach().cpu().clone()
        if variant == 'p0':
            _BASELINES[net] = {'lr': self.lr.clone(), 'routes': {}}
        else:
            base = _BASELINES.get(net)
            if base is None or not torch.equal(base['lr'], self.lr):
                raise ValueError('Run p0 for this exact image before P1 variants')
            if len(base['routes']) != len(net.sfmls):
                raise ValueError('P0 routes are incomplete')
        _ACTIVE[net] = self
        try:
            self.handles.append(net.first_conv.register_forward_hook(self._prev_hook))
            for i in range(len(net.sfmls)):
                self.handles.append(net.sfmls[i].register_forward_hook(self._sfml_hook(i)))
                self.handles.append(net.blocks[i][0].register_forward_pre_hook(self._fm_hook(i)))
                self.handles.append(net.blocks[i][0].iasa_attn.register_forward_pre_hook(self._route_hook(i)))
                self.handles.append(net.esas[i].register_forward_hook(self._prev_hook))
        except Exception:
            self.close()
            raise

    def _prev_hook(self, module, args, out):
        self.prev = out.detach()

    def _sfml_hook(self, i):
        def hook(module, args, out):
            beta, gamma = out
            if self.prev is None or beta.shape != gamma.shape or beta.shape != self.prev.shape:
                raise RuntimeError('SFML/prev shape or cache mismatch')
            stage = {'stage': i, 'shape': list(beta.shape)}
            if self.variant == 'p0':
                # Move each current-stage tensor once, avoiding full-region
                # advanced-indexing copies and statistical scratch space on GPU.
                bc, gc, pc = (t.detach().float().cpu() for t in (beta, gamma, self.prev))
                stage['regions'] = {k: _region_stats(bc, gc, pc, m)
                                    for k, m in self.masks.items()}
                bp, gp = beta, gamma
            else:
                alpha = -.02 if self.variant.endswith('minus') else .02
                bp = 1 + (1 + alpha) * (beta - 1) if self.variant.startswith('beta') else beta
                gp = (1 + alpha) * gamma if self.variant.startswith('gamma') else gamma
            stage['beta_outside_01_fraction'] = ((bp < 0) | (bp > 1)).float().mean().item()
            self.stages[i] = stage
            self.expected_fm = bp.detach() * self.prev + gp.detach()
            # Returning None for p0 preserves the original tuple and tensor identities.
            return None if self.variant == 'p0' else (bp, gp)
        return hook

    def _fm_hook(self, i):
        def hook(module, args):
            actual = args[0]
            expected = self.expected_fm
            if expected is None or actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(f'Stage {i}: FM does not equal beta*prev+gamma')
            self.expected_fm = None
        return hook

    def _route_hook(self, i):
        def hook(module, args):
            idx = args[1].detach().cpu().clone()
            self.routes_seen.add(i)
            if self.variant == 'p0':
                _BASELINES[self.net]['routes'][i] = idx
                self.stages[i]['sorting_permutation_changed_fraction'] = 0.0
            else:
                base = _BASELINES[self.net]['routes'][i]
                if idx.shape != base.shape:
                    raise RuntimeError('P1 route shape changed')
                self.stages[i]['sorting_permutation_changed_fraction'] = (idx != base).float().mean().item()
        return hook

    def finish(self, sr, hr, metrics):
        if self.closed:
            raise RuntimeError('Session already closed')
        if len(self.stages) != len(self.net.sfmls) or len(self.routes_seen) != len(self.net.sfmls):
            raise RuntimeError('Incomplete P1 forward')
        if self.expected_fm is not None or not torch.isfinite(sr).all():
            raise RuntimeError('Incomplete or nonfinite output')
        current = dict(self.net.named_buffers())
        if current.keys() != self.buffers.keys() or any(
                not torch.equal(current[k].detach().cpu(), v) for k, v in self.buffers.items()):
            raise RuntimeError('Model buffers changed during P1 eval')
        return dict(variant=self.variant, alpha=0.0 if self.variant == 'p0' else
                    (-.02 if self.variant.endswith('minus') else .02),
                    sr_shape=list(sr.shape), hr_shape=list(hr.shape), metrics=metrics,
                    stages=[self.stages[i] for i in sorted(self.stages)],
                    route_note='Sorting permutation changes; not clustering membership changes',
                    evidence_scope='Fixed-checkpoint sensitivity only; no trained structural gain')

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if _ACTIVE.get(self.net) is self:
            del _ACTIVE[self.net]
        self.prev, self.expected_fm = None, None
        self.masks.clear()
        self.closed = True


def install(net, lr, masks, variant):
    return Session(net, lr, masks, variant)


def engineering_self_test():
    """CPU toy-interface checks only, not a research/model performance run."""
    import json
    import torch.nn as nn

    class SFML(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x), x * .1

    class IASA(nn.Module):
        def forward(self, x, idx, k, v):
            return x

    class TAB(nn.Module):
        def __init__(self):
            super().__init__()
            self.iasa_attn = IASA()

        def forward(self, x):
            idx = x.flatten(2).mean(1).argsort(1).unsqueeze(-1)
            self.iasa_attn(x, idx, None, None)
            return x

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.first_conv = nn.Identity()
            self.sfmls = nn.ModuleList([SFML(), SFML()])
            self.blocks = nn.ModuleList([nn.ModuleList([TAB()]) for _ in range(2)])
            self.esas = nn.ModuleList([nn.Identity(), nn.Identity()])
            self.register_buffer('sentinel', torch.tensor(1.0))

        def forward(self, x):
            prev = self.first_conv(x)
            for i in range(2):
                beta, gamma = self.sfmls[i](x)
                prev = self.esas[i](self.blocks[i][0](beta * prev + gamma))
            return prev

    net = Toy().eval()
    for shape in [(32, 32), (33, 35), (35, 33)]:
        lr = torch.linspace(-1, 1, 3 * shape[0] * shape[1]).reshape(1, 3, *shape)
        expected = net(lr)
        masks = {'flat': torch.zeros(shape, dtype=torch.bool),
                 'edge': torch.ones(shape, dtype=torch.bool)}
        for variant in VARIANTS:
            session = install(net, lr, masks, variant)
            try:
                sr = net(lr)
                json.dumps(session.finish(sr, expected, {}), allow_nan=False)
                if variant == 'p0':
                    assert torch.equal(sr, expected)
            finally:
                session.close()
            assert torch.equal(net(lr), expected)
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in net.modules())
    return {'passed': True, 'device': 'cpu', 'scope': 'toy interface only', 'shapes': 3}


self_check = engineering_self_test


if __name__ == '__main__':
    print(engineering_self_test())
