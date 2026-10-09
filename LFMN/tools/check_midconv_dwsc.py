"""Sanity checks for Experiment A. Run from LFMN/: python tools/check_midconv_dwsc.py"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.lfmn import Net

def check():
    for scale in (2, 3, 4):
        model = Net(scale=scale).eval()
        layers = list(model.mid_convs)
        assert len(layers) == 8
        assert all(isinstance(m, torch.nn.Sequential) and len(m) == 2 for m in layers)
        assert all(m[0].groups == 48 and m[0].padding == (1, 1)
                   and m[1].kernel_size == (1, 1) for m in layers)
        old = sum(48 * 48 * 3 * 3 + 48 for _ in layers)
        new = sum(sum(p.numel() for p in m.parameters()) for m in layers)
        assert old - new == 143616, (old, new)
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 32 * scale, 32 * scale), (scale, y.shape)
        assert torch.isfinite(y).all(), scale
        print(f"PASS scale={scale}: output={tuple(y.shape)}, total_params={sum(p.numel() for p in model.parameters()):,}, midconv_saved={old-new:,}")

if __name__ == "__main__":
    check()
