"""Frozen stage6 pixel/head gradient predictability; diagnostic regression only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from diagnose_n17_fusion import Net, sha


class Probe:
    def __init__(self, net, stage=6):
        self.module = net.blocks[stage][0].iasa_attn
        self.original = F.scaled_dot_product_attention
        self.active = False
        self.mode = 'gradient'
        self.predictor = None
        self.handles = [self.module.register_forward_pre_hook(self.pre),
                        self.module.register_forward_hook(self.post)]

    def pre(self, module, args):
        self.active = True
        self.x, self.index = args[:2]
        self.responses = []

    def sdpa(self, *args, **kwargs):
        out = self.original(*args, **kwargs)
        if self.active:
            self.responses.append(out)
        return out

    def post(self, module, args, output):
        assert len(self.responses) == 2
        b, n, c = self.x.shape
        heads = module.heads
        responses = []
        for out in self.responses:
            # Match IASA's subgroup/head reshape, trim reflected padding,
            # then restore original pixel order before applying head gates.
            out = out.permute(0, 1, 3, 2, 4).reshape(b, -1, c)[:, :n]
            out = torch.zeros_like(out).scatter(1, self.index.expand_as(out), out)
            responses.append(out.reshape(b, n, heads, c // heads))
        local, global_ = responses
        x = self.x.reshape(b, n, heads, c // heads)
        def norm(z):
            return z.norm(dim=-1, keepdim=True)
        def cosine(a, z):
            return F.cosine_similarity(a, z, dim=-1, eps=1e-8)[..., None]
        features = torch.cat([x, norm(local), norm(global_), cosine(local, global_),
                              norm(local-global_), cosine(x, local), cosine(x, global_)], -1)
        self.features = features.detach()
        if self.mode == 'gradient':
            gate = torch.zeros((b, n, heads, 1), device=x.device, requires_grad=True)
        else:
            gate = self.predictor(self.features).detach()
        self.gate = gate
        correction = module.proj((gate * (local-global_)).reshape(b, n, c))
        self.active = False
        return output + correction

    def __enter__(self):
        F.scaled_dot_product_attention = self.sdpa
        return self

    def __exit__(self, *args):
        F.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()


def tensors(root, index, device):
    lp = root / 'DIV2K_valid_LR_bicubic' / 'X4' / f'{index:04d}x4.png'
    hp = root / 'DIV2K_valid_HR' / f'{index:04d}.png'
    with Image.open(lp) as im:
        lr = np.array(im.convert('RGB'))
    with Image.open(hp) as im:
        hr = np.array(im.convert('RGB'))
    y, x = (lr.shape[0]-64)//2, (lr.shape[1]-64)//2
    def tensor(a):
        return torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().to(device)
    return tensor(lr[y:y+64, x:x+64]), tensor(hr[y*4:(y+64)*4, x*4:(x+64)*4]), {str(lp): sha(lp), str(hp): sha(hp)}


def loss(prediction, hr):
    return ((prediction[..., 4:-4, 4:-4]-hr[..., 4:-4, 4:-4])/255.).square().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original_hash = sha(args.checkpoint)
    net = Net(scale=4).to(device).eval().requires_grad_(False)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    collected, hashes = {}, {}
    route_state = {'capture': True, 'fixed': False, 'index': None, 'changed': []}
    def routing_hook(module, arguments):
        if route_state['capture']:
            route_state['index'] = arguments[1].detach().clone()
        elif route_state['fixed']:
            route_state['changed'].append(int((arguments[1] != route_state['index']).sum().item()))
            return (arguments[0], route_state['index'], *arguments[2:])
    routing_handle = net.blocks[7][0].iasa_attn.register_forward_pre_hook(routing_hook)
    # Baseline equality independently checks the probe's neutral path.
    first_lr, first_hr, _ = tensors(args.data_root, 801, device)
    with torch.no_grad():
        original_output = net(first_lr)
    with Probe(net) as probe:
        for index in range(801, 813):
            lr, hr, h = tensors(args.data_root, index, device)
            hashes.update(h)
            probe.mode = 'gradient'
            prediction = net(lr)
            if index == 801:
                assert torch.equal(prediction.detach(), original_output), 'Neutral path differs'
            mse = loss(prediction, hr)
            gradient, = torch.autograd.grad(mse, probe.gate)
            assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
            features = probe.features.cpu().numpy()[0]
            direction = -gradient.detach().cpu().numpy()[0, ..., 0]
            collected[index] = (features, direction, mse.item())
            if index == 801:
                # Directional finite difference independent of regression.
                direction_tensor = torch.from_numpy(np.sign(direction)).to(device)[None, ..., None]
                analytical = (gradient * direction_tensor).sum().item()
                probe.mode = 'predict'
                route_state['capture'], route_state['fixed'] = False, True
                values = []
                with torch.no_grad():
                    for eps in (-.0001, .0001):
                        probe.predictor = lambda features, eps=eps: eps * direction_tensor
                        values.append(loss(net(lr), hr).item())
                numerical = (values[1]-values[0])/.0002
                relative_error = abs(numerical-analytical)/max(abs(analytical), 1e-12)
                assert relative_error < .1, (analytical, numerical, relative_error)
                finite_difference = {'analytical': analytical, 'numerical': numerical, 'relative_error': relative_error,
                                     'conditional_on_fixed_stage7_routing': True,
                                     'changed_sorted_index_entries_before_freezing': route_state['changed']}
                route_state['capture'], route_state['fixed'] = True, False
            print(f'Gradient labels collected: {index}', flush=True)

        train_x = np.concatenate([collected[i][0] for i in range(801, 807)], 0).astype(np.float64)
        train_y = np.concatenate([collected[i][1] / max(np.sqrt(np.mean(collected[i][1]**2)), 1e-20)
                                  for i in range(801, 807)], 0).astype(np.float64)
        mean, std = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
        coefficients = []
        for head in range(4):
            x = (train_x[:, head]-mean[head])/std[head]
            x = np.column_stack([x, np.ones(len(x))])
            penalty = np.eye(x.shape[1]) * .01
            penalty[-1, -1] = 0
            coefficients.append(np.linalg.solve(x.T@x/len(x)+penalty, x.T@train_y[:, head]/len(x)))
        weights = torch.tensor(np.stack(coefficients), device=device, dtype=torch.float32)
        mu = torch.tensor(mean, device=device, dtype=torch.float32)
        sigma = torch.tensor(std, device=device, dtype=torch.float32)
        constant = torch.tensor(np.sign(train_y.mean(0)), device=device, dtype=torch.float32)[None, None, :, None]
        def linear(features):
            z = (features-mu)/sigma
            return ((z * weights[:, :-1]).sum(-1)+weights[:, -1])[..., None]
        rows = []
        for index in range(807, 813):
            lr, hr, _ = tensors(args.data_root, index, device)
            features, target, base_mse = collected[index]
            probe.mode = 'predict'
            row = {'image': index, 'baseline_mse': base_mse}
            with torch.no_grad():
                for name, predictor in [('linear', lambda f: .05*torch.tanh(linear(f))),
                                        ('constant', lambda f: .05*constant),
                                        ('hr_direction_diagnostic', lambda f: .01*torch.from_numpy(np.sign(target)).to(device)[None, ..., None])]:
                    probe.predictor = predictor
                    value = loss(net(lr), hr).item()
                    assert np.isfinite(value)
                    row[name+'_delta_psnr'] = 10*np.log10(base_mse/value)
                predicted = linear(torch.tensor(features, device=device)[None]).cpu().numpy()[0, ..., 0]
            weight = np.abs(target)
            row['gradient_weighted_sign_accuracy'] = float((weight * (np.sign(predicted)==np.sign(target))).sum()/weight.sum())
            row['constant_weighted_sign_accuracy'] = float((weight * (constant.cpu().numpy().reshape(1, 4)==np.sign(target))).sum()/weight.sum())
            rows.append(row)
            print(json.dumps(row), flush=True)
    assert F.scaled_dot_product_attention is probe.original
    routing_handle.remove()
    assert sha(args.checkpoint) == original_hash and all(p.grad is None for p in net.parameters())
    result = {'checkpoint_sha256': original_hash, 'torch': torch.__version__, 'device': device,
              'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
              'neutral_exact': True, 'checkpoint_unchanged': True, 'model_parameters_unmodified': True,
              'finite_difference': finite_difference, 'train_images': list(range(801, 807)),
              'test_images': list(range(807, 813)), 'stage_zero_based': 6, 'data_sha256': hashes,
              'metric': 'RGB float center LR64/HR256 crop; HR border4; no quantization',
              'rows': rows, 'seconds': time.time()-started,
              'mean_linear_delta': float(np.mean([r['linear_delta_psnr'] for r in rows])),
              'mean_constant_delta': float(np.mean([r['constant_delta_psnr'] for r in rows])),
              'mean_hr_direction_delta': float(np.mean([r['hr_direction_diagnostic_delta_psnr'] for r in rows])),
              'limitations': 'Exploratory chosen stage; six held-out crops; regression fit is a diagnostic, not trained M1; HR direction is not deployable or a strict bound.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('mean_linear_delta', 'mean_constant_delta', 'mean_hr_direction_delta', 'finite_difference', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
