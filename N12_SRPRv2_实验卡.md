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

首次服务器 `--check-only` 在默认两个实验目录下未发现匹配的 B0 40，安全停止且没有启动训练。新增 `--inspect-b0` 输出搜索到的每个 `model_40.pt` 目录及配置、逐图文件或前20轮曲线不匹配原因；可配合 `--search-root /root/autodl-tmp` 扩大只读搜索。当前不把历史 PriorProxy 的不同 B0 轨迹误用为 N12 对照。

全盘服务器检查找到的 PriorProxy 40 轮目录缺少本次逐图指标配置，RGCRD 40 轮 B0 使用 MultiStep 而本次使用 Cosine，因此没有可复用的同协议 B0 40。下一步仅从上述 N9 B0 第20轮 checkpoint 续训剩余20轮。`repro/run_n12_b0_40e_server.py --check-only` 先核验来源、完整曲线、逐图文件及 optimizer/scheduler/loss；正式运行复制原目录到独立 `experiment/all_runs/n12/b0_40e_seed1_from_n9`，核对 model_20 哈希，按原 Cosine T_max=150 继续到40轮。新增默认关闭、只允许 LFMN 的 `--resume_data_epochs 20` 推进采样器与 worker 种子流；不修改 N9 原始20轮目录。完成后原 N12 续训入口才能复用这个同轨迹 B0 40。服务器40轮结果仍待验证。

本地验证：使用真实 DIV2K 单图、x4、LFMN、一个训练 batch 完成第1轮保存和第1→2轮续训，日志明确从第2轮开始，`model_2.pt`、逐图 epoch2、两轮 PSNR 曲线和 `scheduler.last_epoch=2` 均存在；这个缩小数据协议仅为功能烟雾测试，不是研究性能结果。错误协议的烟雾目录被 B0 正式入口拒绝（`data_range` 不符）。

## 40 epoch 决策（2026-09-23）

服务器配对续训成功完成，源码分支 `codex/n12-srprv2`，续训入口提交 `3f42769`。N12 目录为 `/root/autodl-tmp/Image-Super-Resolution/experiment/all_runs/n12/srprv2_20e_seed1_20260922_121528/srprv2`，同轨迹 B0 目录为 `/root/autodl-tmp/Image-Super-Resolution/experiment/all_runs/n12/b0_40e_seed1_from_n9`。两组均存在 `model_40.pt`，自动汇总文件为 N12 目录下的 `n12_40_vs_b0.json`。

| 指标 | 结果 |
| --- | ---: |
| epoch 40 PSNR：N12 / B0 | 28.625822 / 28.630186 dB |
| epoch 40 PSNR 差值 | −0.004364 dB |
| epoch 36–40 PSNR 差值均值 | +0.007066 dB |
| 1–40 正 PSNR 差值轮次 | 36/40（90%） |
| epoch 40 SSIM：N12 / B0 | 0.830547 / 0.830504 |
| epoch 40 SSIM 差值 | +0.000043 |
| epoch 40 逐图 PSNR 均值差 | −0.004357 dB |
| epoch 40 逐图胜率 | 53% |
| epoch 40 paired bootstrap 95% CI | [−0.012260,+0.002455] dB |

结果呈现弱而不稳定的正向训练信号：末5轮均值为正，且第36–39轮均领先，但第40轮落后；终点逐图均值为负、胜率低于预注册的60%，置信区间下界未大于0。再结合服务器相同输入下参数约增加10.8%、Conv2d MAC约增加16.6%、中位延迟约增加45.8%，N12 当前实现未通过40轮晋级闸门，状态记为 **NO-GO**。不启动1000 epoch、五 benchmark 或完整消融。该结论针对当前 SRPRv2 实现，不等同于否定“跨阶段重建状态”这一更广泛研究方向。
