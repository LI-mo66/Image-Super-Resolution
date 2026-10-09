"""Optimizer-free layout/RNG/zero-map audit, including odd full-layout sizes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'LFMN'))


def main():
    import torch
    from model.lfmnsrprv2 import Net as V1
    from model.lfmn_v1n23c import Net
    from model.lfmn import patch_divide, patch_reverse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT/'experiment/all_runs').resolve())
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(1)
    ref = V1(scale=4)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(1)
    net = Net(scale=4)
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(p.numel() for p in net.parameters()) == 854891
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in ref.state_dict().items())
    for stage in (3,7):
        m = net.blocks[stage][1]
        assert m.context_norm_in is not m.layer[0].norm
        assert m.context_norm_out.normalized_shape == (48,)
    ref, net = ref.to(args.device).eval(), net.to(args.device).eval()
    shapes = ((1,3,32,32),(2,3,33,47),(1,3,64,64),(1,3,65,97),(1,3,67,99))
    with torch.no_grad():
        for shape in shapes:
            x = torch.rand(*shape,device=args.device)*255
            y = net(x)
            assert y.shape == (shape[0],3,shape[2]*4,shape[3]*4)
            assert torch.isfinite(y).all() and torch.equal(y,ref(x)), shape
            net.set_enabled(False)
            assert torch.equal(y,net(x)), shape
            net.set_enabled(True)
        # Pixel-coordinate test independent of learned features and attention.
        coords = torch.arange(2*3*67*99,device=args.device,dtype=torch.float32).reshape(2,3,67,99)
        for window in (32,64):
            patches, _, _ = patch_divide(coords,window-2,window)
            restored = patch_reverse(patches,coords,window-2,window,normalize_overlap=True)
            assert torch.equal(coords,restored)
    report = dict(purpose='LAYOUT_ONLY_NO_OPTIMIZER',params=854891,original_keys=549,
                  shapes=shapes,zero_map=True,disabled_map=True,coordinate_roundtrip=True,
                  new_normalization_axis='48 channels per pixel',torch=str(torch.__version__))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
