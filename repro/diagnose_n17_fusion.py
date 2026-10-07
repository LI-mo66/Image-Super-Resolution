"""Frozen B0 response perturbations; exploratory crop metrics, not M1 results."""
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


class Intervention:
    def __init__(self, model):
        self.original = F.scaled_dot_product_attention
        self.active = None
        self.count = 0
        self.calls = [0] * 8
        self.setting = (-1, 0., 0.)
        self.handles = []
        for stage, pair in enumerate(model.blocks):
            self.handles.append(pair[0].iasa_attn.register_forward_pre_hook(self.pre(stage)))
            self.handles.append(pair[0].iasa_attn.register_forward_hook(self.post))

    def pre(self, stage):
        def hook(module, args):
            assert self.active is None
            self.active, self.count = stage, 0
        return hook

    def post(self, module, args, output):
        assert self.count == 2, self.count
        self.calls[self.active] += 1
        self.active = None

    def sdpa(self, *args, **kwargs):
        result = self.original(*args, **kwargs)
        if self.active is not None:
            stage, dl, dg = self.setting
            delta = (dl, dg)[self.count] if stage in (-1, self.active) else 0.
            self.count += 1
            if delta:
                result = result * (1. + delta)
        return result

    def __enter__(self):
        F.scaled_dot_product_attention = self.sdpa
        return self

    def __exit__(self, *args):
        F.scaled_dot_product_attention = self.original
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(scale=4).to(device).eval()
    net.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True), strict=True)
    net.requires_grad_(False)
    settings = [(-1, 0., 0.), (-1, .1, -.1), (-1, -.1, .1),
                (-1, -.1, -.1), (-1, .1, .1)]
    settings += [(s, dl, -dl) for s in range(8) for dl in (.1, -.1)]
    rows, started = [], time.time()
    hashes = {}
    with torch.inference_mode(), Intervention(net) as intervention:
        for index in range(801, 805):
            lp = args.data_root / 'DIV2K_valid_LR_bicubic' / 'X4' / f'{index:04d}x4.png'
            hp = args.data_root / 'DIV2K_valid_HR' / f'{index:04d}.png'
            hashes[str(lp)] = sha(lp)
            hashes[str(hp)] = sha(hp)
            with Image.open(lp) as im:
                lr = np.array(im.convert('RGB'))
            with Image.open(hp) as im:
                hr = np.array(im.convert('RGB'))
            y, x = (lr.shape[0] - 64) // 2, (lr.shape[1] - 64) // 2
            assert y >= 0 and x >= 0
            a = torch.from_numpy(lr[y:y+64, x:x+64].copy()).permute(2, 0, 1)[None].float().to(device)
            b = torch.from_numpy(hr[y*4:(y+64)*4, x*4:(x+64)*4].copy()).permute(2, 0, 1)[None].float().to(device)
            reference = None
            for setting in settings + [settings[0]]:
                intervention.setting = setting
                prediction = net(a)
                assert prediction.shape == b.shape and torch.isfinite(prediction).all()
                if setting == settings[0]:
                    if reference is None:
                        reference = prediction.clone()
                    else:
                        assert torch.equal(reference, prediction), 'Neutral replay differs'
                        continue
                mse = ((prediction[..., 4:-4, 4:-4] - b[..., 4:-4, 4:-4]) / 255.).square().mean().item()
                rows.append({'image': index, 'stage': setting[0], 'dl': setting[1], 'dg': setting[2],
                             'mse': mse, 'psnr': -10 * np.log10(mse)})
            print(f'Completed {index}: {len(settings)} fixed interventions', flush=True)
        assert intervention.calls == [88] * 8, intervention.calls
    assert F.scaled_dot_product_attention is intervention.original
    baseline = {r['image']: r['psnr'] for r in rows if r['dl'] == r['dg'] == 0}
    for row in rows:
        row['delta'] = row['psnr'] - baseline[row['image']]
    means = [(sum(r['delta'] for r in rows if r['image'] <= 802 and
                   (r['stage'], r['dl'], r['dg']) == s) / 2, s) for s in settings]
    _, selected = max(means, key=lambda pair: pair[0])
    selected_rows = [r for r in rows if (r['stage'], r['dl'], r['dg']) == selected]
    result = {'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
              'source_sha256': sha(ROOT / 'LFMN/model/lfmn.py'), 'data_sha256': hashes,
              'torch': torch.__version__, 'device': device, 'parameters': sum(p.numel() for p in net.parameters()),
              'seconds': time.time()-started, 'neutral_replay_exact': True,
              'metric': 'RGB float PSNR; LR64 center crop; HR border4; no quantization',
              'selected_on_801_802': selected, 'selected_rows': selected_rows, 'rows': rows,
              'limitations': 'Four crops; exploratory fixed perturbation; no training; no significance or M1 performance claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'selected': selected, 'selected_rows': selected_rows, 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()
