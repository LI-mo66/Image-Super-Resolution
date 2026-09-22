# N12 / SRPRv2 实验卡

## 身份

```text
候选编号：N12
候选名称：SRPRv2 (State-Routed Proximal Reconstruction v2)
分支：codex/n12-srprv2
共同基线：feature/prior-update-screen / b198de7
状态：SCREENING；20 epoch 质量指标显示正向信号，待服务器机制与效率审计
```

## 核心假设

现有轻量 SISR 阶段主要串联特征变换，没有显式维护由当前观测残差驱动并回写主干的跨阶段重建状态；N12 通过 LR persistent state、固定观测一致性、学习型 proximal 更新和 stage feature writeback 改变阶段组织。如果去掉 residual、state 或 writeback 后性能不降，则假设失败。

## 与 N11 的实质区别

N11 采用 HR 48-channel state 和 average-pool surrogate，且 proximal feature 不回写原始八阶段主干。N12 将 state/feature/writeback 放在 LR 48-channel 分辨率，HR 只保留 3-channel estimate 和 3-channel backprojection；每个 proximal correction 明确写回下一 stage 的 `feat`，使用显式 bicubic analysis 与精确矩阵转置。N12 不继承 N11 代码，不修改 `lfmn.py`，不更新 `fs`/`beta/gamma`，不使用 Token、蒸馏、FFT 或额外损失。

## 公式与形状

```text
xhat_i = bilinear(y + q_i) + PixelShuffle4(stage_probe(feat_i))
r_i = y - D(xhat_i)
b_i = D^T r_i
a_i = A(b_i)
c_i = Phi(feat_i, h_i, a_i)
g_i = sigmoid(G(feat_i, h_i, a_i))
h_(i+1) = h_i + g_i ⊙ (c_i - h_i)
feat_(i+1) = backbone_i(feat_i) + P_i(feat_i,h_(i+1),a_i)
q_(i+1) = q_i + S_i(h_(i+1),a_i)
```

`y/r/a`: `B×3×h×w`; `feat/h`: `B×48×h×w`; `b`: `B×3×4h×4w`; `xhat`: `B×3×4h×4w`; `q`: `B×3×h×w`。
`stage_probe` 是跨阶段共享的轻量 1×1 投影，使每阶段观测残差依赖当前主干特征；最终输出仍由原 LFMN decoder 读取回写后的第8阶段特征并叠加 `bilinear(q)`。观测估计是低成本代理，不等同于最终 decoder 输出。固定 bicubic analysis 与 DIV2K 实际 LR 图像在本地 0001 的 64×64 patch 上平均绝对差为 0.001113（输入归一化到0–1）；伴随精确性只相对于所实现的 analysis 算子成立，不表示与数据生成算子逐像素完全一致。

## 首轮协议

N12 从零训练 20 epoch；B0 复用 N9 已完成从零 B0：`/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/scratch/n9_pcstr_20e_seed1/baseline`。DIV2K `1-800/801-900`、x4、HR patch 256、batch 4、seed 1、L1、Adam、lr 2e-4、Cosine T_max 150、eta_min 1e-6。20 epoch 不通过不得进入 40 或 1000 epoch。

## 验证

本地已通过 bicubic D/D^T float64 内积（测试误差最大 6.66e-15）、尺寸、有限性、proximal/writeback梯度、阶段特征到观测残差梯度、基线权重迁移后输出逐元素一致、真实 DIV2K 单样本前后向、内存保存重载及公共 Trainer 单图训练/评测/机制 JSONL 保存。单图评测 26.186 dB / 0.7147 仅为链路烟雾测试，不能作为性能结论。本地 RTX 4060 Laptop、PyTorch 2.9.1+cu128、FP32、LR 64×64、warmup 10/repeats 50 的同一 profiler：B0/N12 参数 759627/841563，Conv2d MAC 2.833G/3.303G（仅卷积，不含 bicubic、注意力与插值），median 38.99/49.70 ms，peak allocated 214.74/221.49 MiB。正式效率须在服务器复测。

## 服务器入口

```bash
B0_REFERENCE=/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/scratch/n9_pcstr_20e_seed1/baseline \
DATA_ROOT=/root/autodl-tmp/Image-Super-Resolution/datasets \
PYTHON_BIN=python GPU=0 \
bash repro/run_n12_srprv2_screen_server.sh n12/srprv2_20e_seed1
```

## 闸门

20 epoch 末 5 轮相对 B0 稳定为正、逐图 CI 下界为正、至少 60% 图像正收益、SSIM 不退化、机制诊断非零且效率预算合格，才允许 40 epoch；40 epoch通过后才设计 T_max=1000 的正式训练。否则 NO-GO，不用长训赌反转。

## 20→40 epoch 续训入口

`repro/run_n12_srprv2_40e_server.py` 自动寻找 N12 的 `srprv2/model/model_20.pt` 和已有 B0 的 `model/model_40.pt`。B0 40 必须与登记的 N9 B0 在前20轮 PSNR/SSIM 逐点一致（绝对误差≤1e−6），且训练配置与逐图 epoch40 文件齐全；不匹配即停止，不重训 B0。N12 原目录必须有第20轮模型、优化器、调度器、曲线、逐图、机制诊断和 profile。续训固定原 Cosine `T_max=150`，载入第20轮模型/优化器/调度器，并用轻量索引 DataLoader 推进训练随机生成器20轮，使第21轮数据顺序承接已完成的20轮；公共默认训练路径不受影响。本地已在两个 worker 与零 worker 的等长 DataLoader 上核对随机状态，并用真实 DIV2K 单图完成1→2轮续训烟雾测试。若已有 B0 40 不在默认搜索根目录，可显式传 `--b0-40`；先用 `--check-only` 核验，不开始训练。正式 B0 40 的路径、权重哈希及结果仍待服务器发现与记录。
