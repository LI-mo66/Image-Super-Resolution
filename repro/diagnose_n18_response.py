"""Compressed retrieval and frozen IASA response intervention, not trained M1."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from diagnose_n18_retrieval import Net, read_pair, sha


def retrieval(x, projection):
    n, c = x.shape
    point = F.normalize(x, dim=-1)
    patch = F.unfold(x.T.reshape(1, c, 64, 64), 3, padding=1)[0].T
    descriptor = F.normalize(patch @ projection, dim=-1)
    yy, xx = torch.meshgrid(torch.arange(64, device=x.device), torch.arange(64, device=x.device), indexing='ij')
    coordinates = torch.stack([yy.flatten(), xx.flatten()], -1)
    interior = ((coordinates >= 4)&(coordinates < 60)).all(-1)
    chosen = []
    # Block queries bound temporary similarity memory, not total complexity.
    for start in range(0, n, 256):
        qi = torch.arange(start, min(start+256, n), device=x.device)
        valid = ((coordinates[qi, None]-coordinates[None]).abs().amax(-1)>=8)&interior[None]
        scores = (point[qi] @ point.T).masked_fill(~valid, -torch.inf)
        shortlist = scores.topk(64, dim=-1).indices
        similarity = (descriptor[qi, None]*descriptor[shortlist]).sum(-1)
        chosen.append(shortlist.gather(1, similarity.topk(8, dim=-1).indices))
    selected = torch.cat(chosen)
    assert interior[selected].all()
    assert ((coordinates[:, None]-coordinates[selected]).abs().amax(-1)>=8).all()
    return selected


class ResponseProbe:
    def __init__(self, net):
        self.module = net.blocks[6][0].iasa_attn
        generator = torch.Generator(device='cpu').manual_seed(1818)
        self.projection = (torch.randint(0, 2, (48*9, 32), generator=generator).float()*2-1).to(next(net.parameters()).device)/np.sqrt(32)
        self.original = F.scaled_dot_product_attention
        self.epsilon = 0.
        self.active = False
        self.handles = [self.module.register_forward_pre_hook(self.pre), self.module.register_forward_hook(self.post)]

    def pre(self, module, arguments):
        self.x, self.order = arguments[:2]
        self.responses = []
        self.active = True

    def sdpa(self, *args, **kwargs):
        value = self.original(*args, **kwargs)
        if self.active:
            self.responses.append(value)
        return value

    def post(self, module, arguments, output):
        assert len(self.responses) == 2
        self.active = False
        if self.epsilon == 0:
            return output
        x = self.x
        b, n, c = x.shape
        assert b == 1 and n == 4096
        selected = retrieval(x[0], self.projection)
        self.last_selected = selected
        q = module.to_q(x).reshape(b, n, module.heads, -1).permute(0, 1, 2, 3)[:, :, :, None]
        k = module.to_k(x)[0][selected].reshape(n, 8, module.heads, -1).permute(0, 2, 1, 3)[None]
        v = module.to_v(x)[0][selected].reshape(n, 8, module.heads, -1).permute(0, 2, 1, 3)[None]
        new_local = self.original(q, k, v).reshape(b, n, c)
        old_sorted = self.responses[0].permute(0, 1, 3, 2, 4).reshape(b, -1, c)[:, :n]
        old_local = torch.zeros_like(old_sorted).scatter(1, self.order.expand_as(old_sorted), old_sorted)
        # Independent gather checks inversion of the existing sorted layout.
        assert torch.equal(old_local.gather(1, self.order.expand_as(old_local)), old_sorted)
        return output+self.epsilon*module.proj(new_local-old_local)

    def __enter__(self):
        F.scaled_dot_product_attention = self.sdpa
        return self

    def __exit__(self, *args):
        F.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()


def mse(prediction, hr):
    return ((prediction[..., 4:-4, 4:-4]-hr[..., 4:-4, 4:-4])/255.).square().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    net = Net(scale=4).to(device).eval().requires_grad_(False)
    original_hash = sha(args.checkpoint)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    rows, quality, hashes = [], [], {}
    started = time.time()
    with torch.inference_mode(), ResponseProbe(net) as probe:
        for index in range(813, 827):
            lr, hr, h = read_pair(args.data_root, index, device)
            hashes.update(h)
            probe.epsilon = 0.
            baseline = net(lr)
            base_mse = mse(baseline, hr)
            if index <= 818:
                x = probe.x[0]
                selected = retrieval(x, probe.projection)
                residual = (hr-F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False))/255.
                patches = F.unfold(residual, 12, padding=4, stride=4)[0].T
                yy, xx = torch.meshgrid(torch.arange(64, device=device), torch.arange(64, device=device), indexing='ij')
                interior = ((yy>=4)&(yy<60)&(xx>=4)&(xx<60)).flatten().nonzero()[:, 0]
                generator = torch.Generator(device='cpu').manual_seed(1813)
                qi = interior[torch.randperm(len(interior), generator=generator)[:128].to(device)]
                ids = selected[qi]
                error = (patches[ids]-patches[qi, None]).square().mean(-1).mean().item()
                cosine = F.cosine_similarity(patches[ids], patches[qi, None], dim=-1, eps=1e-8).mean().item()
                quality.append({'image':index, 'residual_patch_mse':error, 'residual_cosine':cosine})
                print(f'Compressed candidate proxy: {index}', flush=True)
                continue
            for epsilon in (0., .02, .05, .1):
                probe.epsilon = epsilon
                prediction = net(lr)
                assert torch.isfinite(prediction).all() and prediction.shape == hr.shape
                if epsilon == 0:
                    assert torch.equal(baseline, prediction)
                error = mse(prediction, hr)
                row = {'image':index, 'epsilon':epsilon, 'mse':error, 'delta_psnr':10*np.log10(base_mse/error)}
                rows.append(row)
            print(json.dumps(rows[-4:]), flush=True)
    assert F.scaled_dot_product_attention is probe.original
    assert sha(args.checkpoint) == original_hash
    selected_epsilon = max((0., .02, .05, .1), key=lambda e:np.mean([r['delta_psnr'] for r in rows if r['image']<=822 and r['epsilon']==e]))
    heldout = [r for r in rows if r['image']>=823 and r['epsilon']==selected_epsilon]
    # epsilon=0 cannot be called a positive result.
    mean_delta = float(np.mean([r['delta_psnr'] for r in heldout]))
    wins = int(sum(r['delta_psnr']>0 for r in heldout))
    result = {'checkpoint_sha256':original_hash, 'checkpoint_unchanged':True, 'neutral_exact':True,
              'projection_seed':1818, 'dimension':32, 'shortlist':64, 'keys':8,
              'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
              'data_sha256':hashes, 'quality':quality, 'rows':rows, 'selected_on_819_822':selected_epsilon,
              'heldout_823_826':heldout, 'heldout_mean_delta_psnr':mean_delta, 'heldout_wins':wins,
              'response_gate_pass':mean_delta>0 and wins>=3, 'seconds':time.time()-started,
              'limitations':'Frozen stage6 on crops; original QKV not trained for new keys; failed response does not rule out retraining; coarse global retrieval remains quadratic; no final candidate performance claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('selected_on_819_822','heldout_823_826','heldout_mean_delta_psnr','heldout_wins','response_gate_pass','seconds')}, indent=2))


if __name__ == '__main__':
    main()
