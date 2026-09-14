import os, sys, types, math
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F

import model.lfmn as L

net = L.make_model(types.SimpleNamespace(scale=[2]))
sd = torch.load(r"model\scale2_model_996.pt", map_location="cpu", weights_only=True)
net.load_state_dict(sd, strict=True)
net.eval()


def psnr(a, b, peak):
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(peak ** 2 / mse)


# ---- make a structured "real-ish" HR (edges + smooth shading) in [0,1] ----
torch.manual_seed(0)
H, W = 160, 160
smooth = F.interpolate(torch.rand(1, 1, H // 8, W // 8), size=(H, W), mode="bicubic")
blocks = (smooth > 0.5).float()
hr = (0.6 * smooth + 0.4 * blocks)
hr = (hr - hr.min()) / (hr.max() - hr.min())
hr = hr.expand(1, 3, H, W).contiguous()

lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic", align_corners=False, antialias=True)
naive = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)

with torch.no_grad():
    for peak, tag in [(1.0, "输入0~1"), (255.0, "输入0~255")]:
        try:
            x = lr * peak
            sr = net(x).clamp(0, peak)
            n_psnr = psnr(naive * peak, hr * peak, peak)
            s_psnr = psnr(sr, hr * peak, peak)
            print(f"[{tag:8s}] SR PSNR={s_psnr:6.2f}  插值PSNR={n_psnr:6.2f}   "
                  f"SR范围=[{sr.min().item():.3f},{sr.max().item():.3f}]  SR均值={sr.mean().item():.3f}")
        except Exception as e:
            print(f"[{tag:8s}] 出错: {type(e).__name__}: {e}")
