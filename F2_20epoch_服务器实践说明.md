# F2 ×4：20 epoch 服务器实践说明

## 任务与模型

只运行 F2，不重训已有原始 LFMN（B0）。目标是原始 LFMN ×4、Self-Ensemble OFF 下涨 PSNR。本次只批准 20 epoch 暂停筛选，不能自行继续 150/1000 epoch。

分支：`codex/f2-attention-budget-reallocation`；共同基线：`88bdc6a`。模型继承原始 `lfmn.py`，不继承历史候选。准确运行包提交以克隆后的 `git rev-parse HEAD` 和实际 `config.json` 为准。

唯一结构变量为 LRSA 内 ConvFFN 隐藏宽度：

| 位置 | 原 LFMN | F2 |
| --- | --- | --- |
| TAB 内 8 个 ConvFFN | 96×8 | 96×8 |
| LRSA 内 8 个 ConvFFN | 96×8 | 64、80、112、128、64、80、112、128 |
| 总隐藏宽度 | 1536 | 1536 |

主干输入输出仍为 48 通道，其他结构保留。这是窗口周期容量分配假设，不代表已经证实窄层冗余、宽层重要。预期标准 ×4 参数量 759,627；主要 FFN MAC 匹配由隐藏宽度总和支撑，实际参数与成本以结构检查为准。MAC 匹配不证明 GPU 延迟相同，更不保证 PSNR 上涨。

从零初始化，不加载 B0/F1/历史候选权重。关闭权重 EMA，保留原始 TAB 自身原型 EMA 更新，二者不可混淆。

## 固定协议

| 字段 | 固定值 |
| --- | --- |
| 训练/独立验证 | DIV2K 0001–0800/0801–0900 |
| 倍率/退化 | ×4，提供的 bicubic X4 LR/HR |
| 初始化 | scratch，pretrained_checkpoint=null |
| HR/LR patch | 256/64 |
| Batch/seed | 4/1 |
| 每 epoch batch | 1000 |
| 优化器 | Adam，lr=0.0002，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 损失/精度 | L1/FP32 |
| Cosine | T_max=150，eta_min=0.000001 |
| 计划/暂停 | planned_epochs=150，stop_epoch=20 |
| 增强 | 原始数据管线增强 |
| Self-Ensemble/chop | OFF/OFF |

20 epoch 是完整 Cosine150 曲线的暂停点，绝不是 T_max=20 独立训练。用户已确认 B0/F1 上述训练协议；B0 具体 config、权重位置及数据身份尚待核对，不能提前报告公平 ΔPSNR。

沿用原评测代码：quantize255；DIV2K 验证 RGB PSNR、crop10；benchmark PSNR 用原 BT601-256 Y 转换、shave4，SSIM 用原 MATLAB 风格 Y/crop4。不得修改指标实现提高显示成绩。实际 config 记录评测口径和源码 SHA256。

每个 epoch 权重均记录 Set5 五张图的 PSNR/SSIM 与均值；Set5 只作观察，不选 checkpoint。最佳 checkpoint 按 DIV2K 独立验证确定。epoch20 五个 benchmark 必须使用同一个固定 `model_20.pt`，不能拼接多个 checkpoint 的最高成绩。

## 服务器准备

在全新目录克隆，不覆盖同学当前工作目录，不复制旧权重：

```bash
git clone --single-branch --branch codex/f2-attention-budget-reallocation \
  https://github.com/LI-mo66/Image-Super-Resolution.git Image-Super-Resolution-F2
cd Image-Super-Resolution-F2
git rev-parse HEAD
git status --short
```

使用服务器已有、与 B0 兼容的环境，先从 B0 config 核实 GPU/PyTorch/CUDA/worker；不要盲目升级 PyTorch。示例 workers=4，实际 B0 不同时首次运行前调整并记录。依赖包括 PyTorch CUDA、Pillow、NumPy、scipy、scikit-image、matplotlib、tqdm，缺失时先检查仓库环境配置。还需要 einops、OpenCV（cv2）、imageio。F2 强制使用登记的可读 `lfmn.py`；如果 CPython 3.9/Linux 优先选中仓库 `.so`，入口会拒绝运行，应核对 B0 的实际导入路径和环境，不能悄悄把编译路径当成同一对照。

数据布局，严格匹配文件名与大小写：

```text
/path/datasets/
├── DIV2K/
│   ├── DIV2K_train_HR/0001.png ... 0800.png
│   ├── DIV2K_train_LR_bicubic/X4/0001x4.png ... 0800x4.png
│   ├── DIV2K_valid_HR/0801.png ... 0900.png
│   └── DIV2K_valid_LR_bicubic/X4/0801x4.png ... 0900x4.png
└── benchmark/
    ├── Set5/HR/*.png
    ├── Set5/LR_bicubic/X4/*x4.png
    ├── Set14/HR/*.png
    ├── Set14/LR_bicubic/X4/*x4.png
    ├── B100/HR/*.png
    ├── B100/LR_bicubic/X4/*x4.png
    ├── Urban100/HR/*.png
    ├── Urban100/LR_bicubic/X4/*x4.png
    ├── Manga109/HR/*.png
    └── Manga109/LR_bicubic/X4/*x4.png
```

正式运行检查全部配对与尺寸，五个 benchmark 图片数应为 5/14/100/100/109。路径和尺寸检查不能代替与 B0 数据内容一致性的审核。

## 一条命令运行

```bash
CUDA_VISIBLE_DEVICES=0 python -u repro/run_f2_screen_server.py \
  --data-root /path/datasets --workers 4 \
  --output-root /path/experiment/all_runs
```

只支持单进程单 GPU，不使用 DDP。入口依次执行结构/日志检查、真实单 batch 及恢复烟雾、F2 20 epoch、epoch20 五 benchmark、汇总。检查不通过就停止，不跳过门槛；不会自动继续150 epoch或关机。

终端输出 `RUN GROUP: ...` 后保留完整路径。只做检查或烟雾可分别运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u repro/run_f2_screen_server.py \
  --data-root /path/datasets --workers 4 --check-only
CUDA_VISIBLE_DEVICES=0 python -u repro/run_f2_screen_server.py \
  --data-root /path/datasets --workers 4 --smoke-only
```

正式入口仍会自动执行前置检查。工程目录不能恢复为正式20e实验；烟雾分数不作为 PSNR 研究证据。

## 每个权重的 Set5 分数

新运行组为 `F2_x4_seed1_时间戳/`，正式训练子目录为 `F2_train_x4_seed1/`，至少保留：

- `train_log.txt`：stdout/stderr 实时镜像，含异常 traceback 与恢复 session。
- `config.json`：配置、源码提交/hash、环境、状态及最终/最佳指标。
- `metrics.csv`：每 epoch 训练、学习率和验证指标。
- `model/model_1.pt` 至 `model/model_20.pt`：全部每轮权重。
- `set5_per_checkpoint.csv`：每个权重的 Set5 平均 PSNR/SSIM，绑定权重路径与 SHA256。
- `set5_per_image.csv`：每个 epoch 的五张图 PSNR/SSIM 与权重身份。
- 原始逐图指标、optimizer/scheduler/RNG 状态：用于追溯与恢复。

运行组的 `benchmark_epoch20.csv` 汇总固定 epoch20 五 benchmark。对应 `epoch20_OFF_时间戳/` 有逐图 CSV、评测日志和清单；`summary.md` 汇总结果，最佳权重只按 DIV2K 判定。结构/日志报告及 `gate2.json` 保存工程门槛证据。

完成后核对 checkpoint 表覆盖 epoch1–20，每轮逐图表恰好五张不同图片、SHA256 匹配实际权重。CSV 保留浮点精度，不能由舍入后的终端值反推。漏测或缺权重时不得当作完整结果。

## 恢复与补评测

从最近完整 epoch 恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python -u repro/run_f2_screen_server.py \
  --data-root /path/datasets --workers 4 \
  --resume /path/experiment/all_runs/F2_x4_seed1_实际时间戳
```

恢复追加原日志，恢复模型、optimizer、scheduler、RNG及数据生成器状态。协议、源码、worker、数据路径或环境不一致时拒绝；不要删指标行、换权重或改清单绕过检查。不完整 epoch 的现场保留，由入口检查能否从上一完整 epoch 恢复。如果已有下一轮权重但指标边界尚未完整，入口会拒绝覆盖该权重，需要保留现场后人工核查。

已有完整 epoch20，仅补评测：

```bash
CUDA_VISIBLE_DEVICES=0 python -u repro/run_f2_screen_server.py \
  --data-root /path/datasets --workers 4 \
  --eval-only /path/experiment/all_runs/F2_x4_seed1_实际时间戳
```

该模式不训练；没有 `model_20.pt` 或清单不一致时失败。

## 结果交回与淘汰标准

交回完整运行组路径及：

1. `protocol.json`、训练 `config.json`、完整 `train_log.txt`、`metrics.csv`。
2. epoch1–20 权重的保留路径及 SHA256；权重较大可先给服务器位置，不能丢弃。
3. 两个 Set5 CSV。
4. `benchmark_epoch20.csv`、epoch20逐图 CSV 与评测 config。
5. `summary.md`、工程检查报告及 `gate2.json`。
6. 已有 B0 实际 config、对应权重与逐图结果位置；不要求重新训练 B0。

权重、数据、日志和实验输出不提交 Git。比较前审核 B0 协议、源码、数据、实际评测分支与权重规则；不能不加说明地用 B0 最佳 Set5 权重比较 F2 固定 epoch20。

20e只决定后续验证价值，不按150e门槛提前宣称成功。150e须另获批准；同协议 ΔPSNR≤0 不通过涨分筛选，0～0.03 dB需重复验证，0.03～0.05 dB保留，≥0.05 dB优先多seed与独立验证。单seed/单Set5/参数下降不足以证明创新或泛化收益。

## 本地交付验证（2026-10-09）

RTX 4060 Laptop、PyTorch2.9.1+cu128/CUDA12.8/Python3.10.20：结构/预算、公共初始化与RNG、非方形尺寸、全部FFN梯度及更新、严格保存重载、自集成OFF均通过。真实DIV2K单batch的epoch1→恢复epoch2、验证和Set5逐图记录通过；重载Set5数值一致。日志实时stdout/stderr、非覆盖、受控异常traceback及恢复追加通过。汇总以合成fixture验证，不能当作真实训练结果。未启动20e、未获得F2涨分结论；服务器仍自动重复必要检查。
