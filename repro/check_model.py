import os, sys, types
sys.path.insert(0, os.getcwd())
import torch

import model.lfmn as L

args = types.SimpleNamespace(scale=[2])
net = L.make_model(args)

sd = torch.load(r"model\scale2_model_996.pt", map_location="cpu", weights_only=True)
r = net.load_state_dict(sd, strict=True)
print("=== strict 考试 ===")
print("missing     :", len(r.missing_keys), r.missing_keys[:12])
print("unexpected  :", len(r.unexpected_keys), r.unexpected_keys[:12])

net.eval()
with torch.no_grad():
    x = torch.rand(1, 3, 48, 48)
    y = net(x)
print("=== 前向冒烟 ===")
print("in ", tuple(x.shape), "-> out", tuple(y.shape))
