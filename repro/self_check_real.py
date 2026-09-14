import os, sys, types, math
sys.path.insert(0, r"E:\fuxian_LFMN_jianghe\LFMN")
import torch
import torch.nn.functional as F
import numpy as np
from skimage import data, color
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

import model.lfmn as L

net = L.make_model(types.SimpleNamespace(scale=[2]))
sd = torch.load(r"E:\fuxian_LFMN_jianghe\LFMN\model\scale2_model_996.pt",
                map_location="cpu", weights_only=True)
net.load_state_dict(sd, strict=True)
net.eval()


def bicubic_down(img):  # img: float tensor (1,3,H,W) in [0,255]
    return F.interpolate(img, scale_factor=0.5, mode="bicubic",
                         align_corners=False, antialias=True)


# ---- load a REAL photo ----
astro = data.astronaut()                     # (512,512,3) uint8
img = torch.from_numpy(astro).float().permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

# center-crop to a multiple-of-4 size so x2 / x2 is exact
H, W = 400, 400
img = img[:, :, :H, :W]

hr = img                                   # ground truth HR
lr = bicubic_down(hr)                      # degrade to LR
bic = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)

with torch.no_grad():
    sr = net(lr).clamp(0, 255)

def psnr(a, b):
    a = a.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    b = b.squeeze(0).permute(1, 2, 0).numpy().astype(np.float64)
    return ski_psnr(a, b, data_range=255)

print("astronaut 真实照片 (400x400 RGB)")
print(f"  SR(我们的复现)  PSNR = {psnr(sr, hr):6.2f} dB")
print(f"  双三次插值       PSNR = {psnr(bic, hr):6.2f} dB")
print(f"  SR输出范围=[{sr.min().item():.1f}, {sr.max().item():.1f}]  均值={sr.mean().item():.1f}")
