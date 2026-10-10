"""P2 observational IASA probe. Hooks never replace outputs or update model state."""
import math
import torch
import torch.nn.functional as F

VARIANTS = ['p0']
STAGES = (0, 3, 7)
QUERY_COUNT = 24
COHERENCE_MIN = 0.6
GRADIENT_MIN = 0.01  # normalized LR Y, fixed before observing errors
ANGLE_COS_MAX = math.cos(math.pi / 6)  # axial difference greater than 30 degrees


def candidate_indices(order, group_size):
    """Exactly reproduce baseline mirror padding and 2gs unfold on index slots."""
    order = order.reshape(-1).cpu().long()
    n = order.numel()
    gs = min(n, group_size)
    ng = (n + gs - 1) // gs
    pad = ng * gs - n
    extended = torch.cat((order, order[n - pad - gs:n].flip(0)))
    windows = extended.unfold(0, 2 * gs, gs)
    if windows.shape != (ng, 2 * gs):
        raise ValueError('Baseline mirror padding cannot form all windows for this N/gs')
    return windows, gs


def _geometry(lr):
    rgb = lr.detach().float().cpu() / 255.0
    weights = rgb.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    y = (rgb * weights).sum(1, keepdim=True)
    p = F.pad(y, (1, 1, 1, 1), mode='replicate')
    gx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2]) / 2
    gy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1]) / 2
    smooth = lambda v: F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode='replicate'), 3, 1)
    a, b, c = smooth(gx * gx), smooth(gx * gy), smooth(gy * gy)
    trace = a + c
    coherence = ((a - c).square() + 4 * b.square()).sqrt() / trace.clamp_min(1e-12)
    theta = 0.5 * torch.atan2(2 * b, a - c)
    strength = trace.sqrt()
    valid = (strength >= GRADIENT_MIN) & (coherence >= COHERENCE_MIN)
    # Boundary geometry is padding-dependent, so exclude outer LR row/column.
    valid[:, :, 0, :] = False
    valid[:, :, -1, :] = False
    valid[:, :, :, 0] = False
    valid[:, :, :, -1] = False
    return theta.flatten(), valid.flatten(), coherence.flatten(), strength.flatten()


def _number(value):
    return float(value) if torch.isfinite(torch.as_tensor(value)).all() else None


def _pearson(x, y):
    if len(x) < 3:
        return None
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    return _number((x * y).sum() / denominator) if denominator > 0 else None


class Session:
    def __init__(self, net, lr, masks, variant):
        if variant not in VARIANTS:
            raise ValueError('P2 supports observation only: p0')
        if net.training or lr.shape[0] != 1:
            raise ValueError('P2 requires eval mode and batch size one')
        self.h, self.w = lr.shape[-2:]
        self.theta, self.valid, self.coherence, self.strength = _geometry(lr)
        if any(mask.dtype != torch.bool or tuple(mask.shape) != (self.h, self.w)
               for mask in masks.values()):
            raise ValueError('Masks must be boolean H,W with exact spatial shape')
        self.masks = {name: mask.detach().bool().cpu().reshape(-1) for name, mask in masks.items()}
        if any(mask.numel() != self.h * self.w for mask in self.masks.values()):
            raise ValueError('LR masks have wrong spatial dimensions')
        self.handles, self.rows, self.means = [], {}, {}
        try:
            for stage in STAGES:
                tab = net.blocks[stage][0]
                if tab.eval_refine_iters != 0:
                    raise ValueError('P2 requires original stored-prototype eval behavior')
                self.handles.append(tab.irca_attn.register_forward_hook(self._mean_hook(stage)))
                self.handles.append(tab.iasa_attn.register_forward_pre_hook(self._attention_hook(stage)))
        except BaseException:
            self.close()
            raise

    def _mean_hook(self, stage):
        def hook(module, inputs, output):
            self.means[stage] = output[2].detach()
        return hook

    def _attention_hook(self, stage):
        def hook(module, inputs):
            with torch.no_grad():
                x, idx, _, _ = inputs
                if x.shape[0] != 1 or x.shape[1] != self.h * self.w:
                    raise ValueError('Unexpected IASA spatial shape')
                # Same GPU expression as TAB to preserve actual label boundary decisions.
                scores = torch.einsum('b i c,j c->b i j', F.normalize(x, dim=-1),
                                      F.normalize(self.means[stage], dim=-1))
                labels = scores.argmax(-1)[0].cpu()
                order = idx[0, :, 0].detach().cpu()
                windows, gs = candidate_indices(order, module.group_size)
                inverse = torch.empty_like(order)
                inverse[order] = torch.arange(order.numel())
                queries = torch.linspace(0, order.numel() - 1, min(QUERY_COUNT, order.numel())).round().long().unique()
                cpu_x = x[0].detach().float().cpu()
                q_weight = module.to_q.weight.detach().float().cpu()
                k_weight = module.to_k.weight.detach().float().cpu()
                records = []
                for query in queries.tolist():
                    keys = windows[int(inverse[query]) // gs]
                    q = F.linear(cpu_x[query], q_weight).reshape(module.heads, -1)
                    k = F.linear(cpu_x[keys], k_weight).reshape(keys.numel(), module.heads, -1).permute(1, 0, 2)
                    logits = (q[:, None, :] * k).sum(-1) / math.sqrt(q.shape[-1])
                    attention = logits.softmax(-1)
                    eligible = self.valid[keys] & self.valid[query]
                    incompatible = eligible & ((self.theta[keys] - self.theta[query]).cos().abs() < ANGLE_COS_MAX)
                    cross = labels[keys] != labels[query]
                    mismatch_mass = attention[:, incompatible].sum(-1)
                    eligible_mass = attention[:, eligible].sum(-1)
                    conditional = [float(m / e) if e > 1e-12 else None for m, e in zip(mismatch_mass, eligible_mass)]
                    records.append({
                        'query': query, 'lr_y': query // self.w, 'lr_x': query % self.w,
                        'region': [name for name, mask in self.masks.items() if bool(mask[query])],
                        'query_geometry_valid': bool(self.valid[query]),
                        'query_coherence': float(self.coherence[query]),
                        'query_gradient_rms': float(self.strength[query]),
                        'candidate_slots': keys.numel(), 'unique_candidate_pixels': keys.unique().numel(),
                        'duplicate_slot_fraction': 1 - keys.unique().numel() / keys.numel(),
                        'cross_label_equal_fraction': float(cross.float().mean()),
                        'cross_label_attention_mass_per_head': attention[:, cross].sum(-1).tolist(),
                        'geometry_eligible_equal_fraction': float(eligible.float().mean()),
                        'geometry_eligible_attention_mass_per_head': eligible_mass.tolist(),
                        'incompatible_equal_mass': float(incompatible.float().mean()),
                        'incompatible_attention_mass_per_head': mismatch_mass.tolist(),
                        'incompatible_equal_conditional': float(incompatible.sum() / eligible.sum()) if eligible.any() else None,
                        'incompatible_attention_conditional_per_head': conditional,
                        'self_attention_mass_per_head': attention[:, keys == query].sum(-1).tolist(),
                        'attention_entropy_per_head': (-(attention * attention.clamp_min(1e-30).log()).sum(-1)).tolist(),
                    })
                self.rows[stage] = records
                del self.means[stage]
        return hook

    def finish(self, sr, hr, metrics):
        if set(self.rows) != set(STAGES):
            raise RuntimeError('Not all registered stages executed')
        if sr.shape != hr.shape or sr.shape[0] != 1:
            raise ValueError('SR/HR must match and use batch size one')
        sh, sw = sr.shape[-2:]
        if sh % self.h or sw % self.w or sh // self.h != sw // self.w:
            raise ValueError('HR does not align with LR integer scale footprints')
        scale = sh // self.h
        delta = (sr.detach().float().cpu() - hr.detach().float().cpu())
        weights = delta.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
        error = F.avg_pool2d((delta * weights).sum(1, keepdim=True).square(), scale, scale).flatten()
        stages = {}
        for stage, records in self.rows.items():
            masses, errors = [], []
            for row in records:
                row['final_unquantized_y_footprint_mse'] = float(error[row['query']])
                if row['geometry_eligible_equal_fraction'] > 0:
                    masses.append(sum(row['incompatible_attention_mass_per_head']) / len(row['incompatible_attention_mass_per_head']))
                    errors.append(row['final_unquantized_y_footprint_mse'])
            stages[str(stage)] = {
                'queries': records, 'eligible_query_count': len(masses),
                'pearson_incompatible_mass_vs_final_error': _pearson(masses, errors),
            }
        return {
            'variant': 'p0', 'observation_only': True, 'metrics': metrics, 'stages': stages,
            'protocol': {
                'query_selection': '24 evenly spaced original row-major LR indices, fixed independently of error',
                'attention_reconstruction': 'detached FP32 CPU, actual mirrored candidate slots, per-head softmax',
                'geometry': 'LR normalized limited-range Y; 3x3 structure tensor; axial angle difference >30 degrees',
                'gradient_rms_min': GRADIENT_MIN, 'coherence_min': COHERENCE_MIN,
                'error': 'unclamped unquantized Y squared error, full HR footprint including image boundary, RGB255 units',
                'scope': 'within-image sample correlation only; queries and heads are not independent statistical replicates',
            },
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.means.clear()


def install(net, lr, masks, variant='p0'):
    return Session(net, lr, masks, variant)


def engineering_checks():
    # Compare index windows against scalar feature padding/unfold from the baseline.
    for n, gs in ((64, 16), (67, 16), (96, 32), (257, 64), (10, 32)):
        order = torch.arange(n).flip(0)
        windows, actual_gs = candidate_indices(order, gs)
        ng = (n + actual_gs - 1) // actual_gs
        pad = ng * actual_gs - n
        feature = order.reshape(1, n, 1).float()
        reference = torch.cat((feature, feature[:, n - pad - actual_gs:n, :].flip(-2)), -2)
        reference = reference.unfold(-2, 2 * actual_gs, actual_gs)[0, :, 0, :].long()
        assert torch.equal(windows, reference)
        assert all(int(order[pos]) in windows[pos // actual_gs] for pos in range(n))
    generator = torch.Generator().manual_seed(2048)
    q = torch.randn(2, 3, 9, generator=generator)
    k = torch.randn(2, 7, 9, generator=generator)
    v = torch.randn(2, 7, 12, generator=generator)
    manual = ((q @ k.transpose(-2, -1)) / math.sqrt(9)).softmax(-1) @ v
    reference = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(manual, reference, rtol=1e-5, atol=1e-6)
    return {'candidate_indices': 'pass', 'softmax_vs_sdpa': 'pass'}


self_check = engineering_checks


if __name__ == '__main__':
    print(engineering_checks())
