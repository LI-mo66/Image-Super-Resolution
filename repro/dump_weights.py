import torch

path = r"..\LFMN\model\scale2_model_996.pt"
sd = torch.load(path, map_location="cpu", weights_only=True)

print("=== 门牌号总数:", len(sd), "===")
total = 0
for k, v in sd.items():
    n = v.numel()
    total += n
    print(f"{k:55s} shape={tuple(v.shape)!s:26s} numel={n}")
print("=== 全部数字相加(含buffer):", total, "===")
