"""Distinguish actual startup from explicitly measured sub-ULP Adam updates."""
import torch


def audit_updates(net, new, before, optimizer):
    norm_weights = {name+'.weight' for name,module in net.named_modules()
                    if isinstance(module,torch.nn.LayerNorm)}
    changed, rounding = [], {}
    group_for = {id(p):group for group in optimizer.param_groups for p in group['params']}
    for name,p in new.items():
        if not torch.equal(before[name],p.detach()):
            changed.append(name)
            continue
        state = optimizer.state.get(p,{})
        if (name not in norm_weights or p.grad is None or not torch.isfinite(p.grad).all()
                or not bool(p.grad.abs().sum()>0) or 'exp_avg' not in state
                or not bool(state['exp_avg'].abs().sum()>0)):
            raise RuntimeError('New parameter inactive without rounding explanation: '+name)
        group = group_for[id(p)]
        b1,b2 = group['betas']
        step = float(state['step'])
        expected = (group['lr']*state['exp_avg']/(1-b1**step)
                    /(state['exp_avg_sq'].sqrt()/(1-b2**step)**.5+group['eps']))
        previous_float = torch.nextafter(p.detach(),torch.full_like(p,float('-inf')))
        min_spacing = float((p.detach()-previous_float).abs().min())
        expected_max = float(expected.abs().max())
        if not torch.isfinite(expected).all() or expected_max >= min_spacing/2:
            raise RuntimeError('Unchanged parameter exceeds sub-ULP explanation: '+name)
        rounding[name] = dict(gradient_max=float(p.grad.abs().max()),adam_step=step,
                             expected_last_update_max=expected_max,
                             minimum_fp32_spacing=min_spacing,
                             reason='NONZERO_GRADIENT_AND_ADAM_MOMENT_BUT_SUB_HALF_ULP')
    if not changed:
        raise RuntimeError('No actual new parameter updates')
    return changed,rounding
