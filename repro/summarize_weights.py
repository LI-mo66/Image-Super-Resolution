import torch, re, collections

path = r"..\LFMN\model\scale2_model_996.pt"
sd = torch.load(path, map_location="cpu", weights_only=True)

with open("scale2_inventory.txt", "w", encoding="utf-8") as f:
    for k, v in sd.items():
        f.write(f"{k:55s} shape={tuple(v.shape)!s:28s} numel={v.numel()}\n")

def mask(k):
    return re.sub(r"\d+", "N", k)

tpl = collections.OrderedDict()
for k, v in sd.items():
    t = mask(k)
    if t not in tpl:
        tpl[t] = {"count": 0, "real": k, "shape": tuple(v.shape)}
    tpl[t]["count"] += 1

print(f"key_total={len(sd)}  template_total={len(tpl)}")
print("-" * 100)
for t, info in tpl.items():
    print(f"{info['count']:3d}x {t:58s} shape={str(info['shape']):26s} e.g.{info['real']}")
print("full list saved to repro/scale2_inventory.txt")
