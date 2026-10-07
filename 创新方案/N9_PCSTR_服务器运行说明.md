# N9 PCSTR：完整服务器运行流程

分支 `codex/n9-pcstr-core-m1`，共同基线 `b198de7`。这是已有 N9 算子的独立 M1 实验入口，未重新命名为 N17。
新服务器使用独立目录。现有 SRPR 服务器继续运行原分支。

## 1. 拉取完整代码

```bash
cd /root/autodl-tmp
git clone --branch codex/n9-pcstr-core-m1 https://github.com/LI-mo66/Image-Super-Resolution.git Image-Super-Resolution-PCSTR
cd Image-Super-Resolution-PCSTR
git log -1 --oneline
git status --short
```

不要使用 `--depth 1`：检查脚本需要读取历史 `bf68bfe` 源码做基线等价验证。

建议 Python3.10、CUDA兼容的 PyTorch2.9.1（本地验证2.9.1+cu128）。服务器已有匹配环境时直接使用；若需安装：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.2.6 einops==0.8.2 opencv-python==5.0.0.93 imageio==2.37.4 matplotlib==3.10.9 tqdm==4.70.0
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

显式使用 CUDA GPU0；保留8个DataLoader worker，至少16个CPU核心。低CPU规格会改变原训练入口实际线程数，入口将拒绝运行。

## 2. 准备数据和已有 B0

数据必须是官方 DIV2K bicubic ×4，完整900对，目录如下（可将已有数据目录作为 `--data-root` 指定，无需复制）：

```text
datasets/DIV2K/DIV2K_train_HR/0001.png ... 0800.png
datasets/DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png ... 0800x4.png
datasets/DIV2K/DIV2K_valid_HR/0801.png ... 0900.png
datasets/DIV2K/DIV2K_valid_LR_bicubic/X4/0801x4.png ... 0900x4.png
```

将交付的 `n9_b0_reuse.tar.gz` 上传到 `/root/autodl-tmp/`，再解压到独立目录：

SHA256：`45cc61193e2e0f0aca2d10b8ff1f4ed5e1a10b687219c2a3dd581fd0e0197924`；上传后用 `sha256sum /root/autodl-tmp/n9_b0_reuse.tar.gz` 核对。

```bash
mkdir -p /root/autodl-tmp/n9_b0_reuse
tar -xzf /root/autodl-tmp/n9_b0_reuse.tar.gz -C /root/autodl-tmp/n9_b0_reuse
ls /root/autodl-tmp/n9_b0_reuse/model/model_20.pt
```

包包含 B0 原始配置、scheduler、150轮完整精度曲线、日志，以及16–20轮权重；不含数据。旧的 `prior_proxy_scratch_seed1_results.tar.gz` **只有曲线和日志，没有checkpoint**，不能代替该包。若已有完整B0目录，可直接指定，无需上传。

## 3. 只检查和复评（无20轮训练）

```bash
cd /root/autodl-tmp/Image-Super-Resolution-PCSTR
bash repro/run_n9_m1_server.sh \
  --baseline /root/autodl-tmp/n9_b0_reuse \
  --data-root /root/autodl-tmp/Image-Super-Resolution-PCSTR/datasets \
  --output /root/autodl-tmp/Image-Super-Resolution-PCSTR/experiment/all_runs/n9/pcstr_20e_seed1_r1 \
  --prepare-only
```

依次：配置/调度器/权重检查 → 所有HR/LR数据hash → 合成结构/梯度/AMP → 旧基线CPU前向/梯度等价 → 真实batch4训练一步及0801整图验证 → 保存严格重载 → B0第20轮全100图复评。单步烟雾输出不作性能证据。

B0回放均值与历史第20轮差值必须 PSNR≤0.0002dB、SSIM≤0.00001。否则停止，保留证据，不允许绕过后继续。旧源码提交没有记录，bf68bfe是时间/脚本推断；实现默认路径等价和复评支持复用，但不补造来源元数据。完整环境、新数据hash和旧权重hash均写manifest。

## 4. 后台只训练 M1

上一步成功后，以同一输出目录启动：

```bash
nohup bash repro/run_n9_m1_server.sh \
  --baseline /root/autodl-tmp/n9_b0_reuse \
  --data-root /root/autodl-tmp/Image-Super-Resolution-PCSTR/datasets \
  --output /root/autodl-tmp/Image-Super-Resolution-PCSTR/experiment/all_runs/n9/pcstr_20e_seed1_r1 \
  --prepared \
  > n9_pcstr_20e_seed1_r1.log 2>&1 < /dev/null &
echo $!
tail -F n9_pcstr_20e_seed1_r1.log
```

Ctrl+C只结束tail查看。也可首次直接去掉 `--prepare-only/--prepared`，一次完成检查、复评和短筛。
已有输出拒绝覆盖；失败后使用新run名，不能通过反复重跑选择最好seed1结果。

固定从零20轮，每轮1000batch、DIV2K完整100图验证、HR256、batch4、seed1、8workers、Adam2e-4、L1、FP32；Cosine **T_max150**、eta_min1e-6。不能把短筛 horizon 改成20，也不能用MultiStep对比该旧B0。

## 5. 自动交付和判定

```text
manifest.json                    commit、协议、环境、数据与B0哈希、状态
smoke/verification.json           结构/真实数据验证
b0_eval/per_image_metrics/epoch_0020.pt
m1/config.txt console.log log.txt
m1/psnr_log.pt ssim_log.pt
m1/per_image_metrics/epoch_*.pt
m1/model/model_*.pt
diagnosis.json                    已训练权重路由、交替计时、显存
summary.json                      同epoch增量、末5、斜率、逐图、CI、纹理组、闸门
```

训练后自动检查路由有效Token>8、质量比<100及LR64延迟/显存≤B0的110%；不合格则STOP。这里只测一个尺寸，不作为全面部署效率结论；总FLOPs、整图峰值仍需后续审计。
最终汇总使用RGB裁10 PSNR、量化Y裁4及有效窗SSIM，与原utility一致。旧N9评审文档“DIV2K Y裁4 PSNR”文字在本流程中纠正。

可重新汇总（只读模型，写summary）：

```bash
python repro/run_n9_m1_server.py \
  --baseline /root/autodl-tmp/n9_b0_reuse \
  --data-root /root/autodl-tmp/Image-Super-Resolution-PCSTR/datasets \
  --output /root/autodl-tmp/Image-Super-Resolution-PCSTR/experiment/all_runs/n9/pcstr_20e_seed1_r1 \
  --summarize-only
```

20轮满足实验卡通过/灰区规则才讨论40轮，不自动启动长训或消融，不自动关机。没有任何已确认PSNR提升。
