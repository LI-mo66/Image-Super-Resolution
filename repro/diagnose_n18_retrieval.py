"""Frozen-feature retrieval proxies; no training or reconstruction claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from model.lfmn import Net


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_layout(index, gs):
    n = len(index)
    pad = ((n+gs-1)//gs)*gs-n
    extended = torch.cat([index, index[n-pad-gs:n].flip(0)])
    windows = extended.unfold(0, 2*gs, gs)
    assert windows.shape == ((n+gs-1)//gs, 2*gs)
    # Independent list construction checks reflection and subgroup offsets.
    explicit = index.tolist()+list(reversed(index[n-pad-gs:n].tolist()))
    expected = torch.tensor([explicit[j*gs:(j+2)*gs] for j in range(len(windows))], device=index.device)
    assert torch.equal(windows, expected)
    return windows


def read_pair(root, index, device):
    lp = root/'DIV2K_valid_LR_bicubic'/'X4'/f'{index:04d}x4.png'
    hp = root/'DIV2K_valid_HR'/f'{index:04d}.png'
    with Image.open(lp) as im:
        lr = np.array(im.convert('RGB'))
    with Image.open(hp) as im:
        hr = np.array(im.convert('RGB'))
    y, x = (lr.shape[0]-64)//2, (lr.shape[1]-64)//2
    def tensor(z):
        return torch.from_numpy(z.copy()).permute(2, 0, 1)[None].float().to(device)
    return tensor(lr[y:y+64, x:x+64]), tensor(hr[y*4:(y+64)*4, x*4:(x+64)*4]), {str(lp):sha(lp), str(hp):sha(hp)}


def analyze(net, stage, saved, hr_patch, residual_patch):
    features, index = saved
    features, index = features[0], index[0, :, 0]
    n, c = features.shape
    module = net.blocks[stage][0].iasa_attn
    gs = min(n, module.group_size)
    windows = key_layout(index, gs)
    inverse = torch.empty_like(index)
    inverse[index] = torch.arange(n, device=index.device)
    yy, xx = torch.meshgrid(torch.arange(64, device=index.device), torch.arange(64, device=index.device), indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], -1)
    interior = ((coords >= 4)&(coords < 60)).all(-1)
    generator = torch.Generator(device='cpu').manual_seed(1813)
    ids = interior.nonzero()[:, 0]
    query = ids[torch.randperm(len(ids), generator=generator)[:128].to(ids.device)]
    far = (coords[query, None]-coords[None]).abs().amax(-1) >= 8
    valid = far & interior[None]
    allowed = torch.zeros_like(valid)
    allowed.scatter_(1, windows[inverse[query]//gs], True)
    point = F.normalize(features, dim=-1)
    fmap = features.T.reshape(1, c, 64, 64)
    structure = F.normalize(F.unfold(fmap, 3, padding=1)[0].T, dim=-1)
    point_score = point[query]@point.T
    structure_score = structure[query]@structure.T
    q = module.to_q(features).reshape(n, module.heads, -1)
    k = module.to_k(features).reshape(n, module.heads, -1)
    qk_score = torch.einsum('qhd,nhd->qn', q[query], k)/np.sqrt(q.shape[-1])/module.heads
    choices = [('iasa_qk', qk_score, valid&allowed),
               ('iasa_point', point_score, valid&allowed), ('global_point', point_score, valid),
               ('iasa_structure', structure_score, valid&allowed), ('global_structure', structure_score, valid)]
    # Some original groups may not offer 8 valid distant candidates.
    # Compare all methods on exactly the common supported queries.
    supported = (valid&allowed).sum(-1) >= 8
    assert supported.any()
    results = {}
    selected_ids = {}
    per_query = {}
    for name, score, mask in choices:
        candidate = score.masked_fill(~mask, -torch.inf).topk(8, dim=-1).indices[supported]
        qi = query[supported]
        assert ((coords[qi, None]-coords[candidate]).abs().amax(-1) >= 8).all()
        assert interior[candidate].all()
        error = (residual_patch[candidate]-residual_patch[qi, None]).square().mean(-1).mean(-1)
        raw_error = (hr_patch[candidate]-hr_patch[qi, None]).square().mean(-1).mean(-1)
        cosine = F.cosine_similarity(residual_patch[candidate], residual_patch[qi, None], dim=-1, eps=1e-8)
        energy = residual_patch[candidate].square().mean(-1).mean()
        target_energy = residual_patch[qi].square().mean()
        results[name] = {'residual_patch_mse': error.mean().item(), 'hr_patch_mse':raw_error.mean().item(),
                         'residual_cosine':cosine.mean().item(), 'candidate_residual_energy':energy.item(),
                         'query_residual_energy':target_energy.item(),
                         'outside_iasa_fraction': (~allowed[supported].gather(1, candidate)).float().mean().item()}
        selected_ids[name] = candidate
        per_query[name] = error.cpu().tolist()
    pairs = {}
    for reference in ('iasa_qk', 'global_point', 'iasa_structure'):
        a = np.array(per_query['global_structure'])
        b = np.array(per_query[reference])
        pairs[reference] = {'mean_mse_difference':float((a-b).mean()),
                            'relative_mean_mse_reduction':float(1-a.mean()/b.mean()),
                            'query_win_rate':float((a<b).mean())}
    return {'stage':stage, 'sampled_queries':128, 'supported_queries':int(supported.sum().item()),
            'methods':results, 'structure_vs':pairs, 'per_query_residual_mse':per_query}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(scale=4).to(device).eval().requires_grad_(False)
    original_hash = sha(args.checkpoint)
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    saved, handles = {}, []
    for stage in (0, 6):
        def capture(module, arguments, stage=stage):
            saved[stage] = (arguments[0].detach().clone(), arguments[1].detach().clone())
        handles.append(net.blocks[stage][0].iasa_attn.register_forward_pre_hook(capture))
    rows, hashes, started = [], {}, time.time()
    try:
        with torch.inference_mode():
            # Verify layout on both no-padding and reflected-padding examples.
            for n, gs in ((4096, 256), (4096, 64), (101, 32)):
                key_layout(torch.arange(n, device=device), gs)
            for index in range(813, 819):
                lr, hr, h = read_pair(args.data_root, index, device)
                hashes.update(h)
                prediction = net(lr)
                assert prediction.shape == hr.shape and torch.isfinite(prediction).all()
                residual = (hr-F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False))/255.
                residual_patch = F.unfold(residual, 12, padding=4, stride=4)[0].T
                hr_patch = F.unfold(hr/255., 12, padding=4, stride=4)[0].T
                assert len(residual_patch) == 4096
                for stage in (0, 6):
                    row = analyze(net, stage, saved[stage], hr_patch, residual_patch)
                    row['image'] = index
                    rows.append(row)
                    print(json.dumps({'image': index, 'stage':stage, 'structure_vs':row['structure_vs'],
                                      'outside_iasa':row['methods']['global_structure']['outside_iasa_fraction']}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        assert torch.equal(prediction, net(lr)), 'Capture hooks changed baseline output'
    assert sha(args.checkpoint) == original_hash
    summary = {}
    for stage in (0, 6):
        subset = [r for r in rows if r['stage']==stage]
        summary[str(stage)] = {ref: {'mean_image_relative_mse_reduction':float(np.mean([r['structure_vs'][ref]['relative_mean_mse_reduction'] for r in subset])),
                                    'images_improved':sum(r['structure_vs'][ref]['mean_mse_difference']<0 for r in subset),
                                    'mean_residual_cosine_gain':float(np.mean([r['methods']['global_structure']['residual_cosine']-r['methods'][ref]['residual_cosine'] for r in subset]))}
                               for ref in ('iasa_qk', 'global_point', 'iasa_structure')}
    result = {'checkpoint_sha256': original_hash, 'source_sha256':sha(ROOT/'LFMN/model/lfmn.py'),
              'torch':torch.__version__, 'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
              'data_sha256':hashes, 'images':list(range(813,819)), 'rows':rows, 'summary':summary,
              'seconds':time.time()-started, 'checkpoint_unchanged':True, 'capture_output_exact':True,
              'limitations':'Crop retrieval proxy; HR only scores LR-feature selected candidates; no model PSNR, no trained candidate, no deployment efficiency claim; averaged QK ranking is not full multihead attention output.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
