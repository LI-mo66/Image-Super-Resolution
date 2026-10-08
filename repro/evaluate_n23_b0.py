"""Re-evaluate all audited B0 checkpoints with the shared exact profile.

No optimizer, no old metrics imported. Controller supplies a frozen manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from data.div2k import DIV2K
from model.lfmn_exact_overlap import Net
import utility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    audit = manifest['baseline_audit']
    if audit['status'] != 'PASS':
        raise RuntimeError('Baseline not audited')
    output = Path(manifest['output']) / 'b0'
    output.mkdir(exist_ok=False)
    (output / 'per_image_metrics').mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    settings = SimpleNamespace(dir_data=manifest['data_root'], data_range='1-800/801-900',
                               ext='img', scale=[4], test_only=True, n_colors=3, rgb_range=255)
    dataset = DIV2K(settings, name='DIV2K', train=False)
    assert [Path(p).stem for p in dataset.images_hr] == [f'{i:04d}' for i in range(801, 901)]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=manifest['protocol']['threads'])
    # utility PSNR queries loader.dataset.benchmark, identical to Trainer.test.
    net = Net(scale=4).cuda().eval()
    curves = {'psnr': [], 'ssim': []}
    for epoch in range(1, 41):
        checkpoint = Path(audit['directory']) / 'model' / f'model_{epoch}.pt'
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert actual == audit['checkpoint_sha256'][str(epoch)], 'B0 checkpoint changed'
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        assert all(bool(v.item()) for k, v in net.state_dict().items() if k.endswith('.initted'))
        rows, psnr, ssim = [], torch.zeros(()), torch.zeros(())
        with torch.no_grad():
            for lr, hr, filename in loader:
                lr, hr = lr.cuda(), hr.cuda()
                raw = net(lr)
                assert raw.shape == hr.shape and bool(torch.isfinite(raw).all())
                sr = utility.quantize(raw, 255)
                p = utility.calc_psnr(sr, hr, 4, 255, dataset=loader)
                s = float(utility.calc_ssim(sr, hr, 4, 255, dataset=loader))
                rows.append(dict(dataset='DIV2K', scale=4, filename=filename[0], psnr=p, ssim=s))
                psnr += p
                ssim += s
        assert len(rows) == 100
        torch.save(rows, output / 'per_image_metrics' / f'epoch_{epoch:04d}.pt')
        curves['psnr'].append(float(psnr / 100))
        curves['ssim'].append(float(ssim / 100))
        for key, values in curves.items():
            torch.save(torch.tensor(values).reshape(-1, 1, 1), output / f'{key}_log.pt')
        print(f'B0 exact epoch {epoch}/40 PSNR={curves["psnr"][-1]:.8f} SSIM={curves["ssim"][-1]:.8f}', flush=True)
    (output / 'evaluation_protocol.json').write_text(json.dumps(
        {'protocol': manifest['protocol'], 'baseline_audit': audit,
         'note': 'New exact metrics; old legacy curves not reused'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
