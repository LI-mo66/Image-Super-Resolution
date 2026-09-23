# N15 / SRPR-SS 实验卡

## 1. 身份信息

```text
候选编号：N15
候选名称：SRPR-SS (Semi-Shared State-Routed Proximal Reconstruction)
候选类型：跨阶段重建框架 / 参数共享
当前状态：VERIFIED（待服务器 20 epoch 筛选）
分支：codex/n15-srprss
共同基线提交：b198de7
日期：2026-09-23
```

## 2. 一句话核心假设

由于 SRPRv2 在八个阶段中分别学习结构完全相同的 proximal 更新器，它在小数据轻量 SR 中引入了阶段参数冗余和后期漂移；N15 使用一个跨阶段共享的重建动力学核，并仅保留阶段特异的通道调制，预期在保留已验证状态—观测—写回因果链的同时，减少额外参数并改善 20→40 epoch 稳定性。

## 3. 证据与反证

- N12/SRPRv2 在 epoch 20 约为 `+0.0128 dB`，但 epoch 40 为 `-0.0044 dB`，未通过稳定性闸门。
- N12 相对 B0 参数 `+10.8%`、Conv2d MAC `+16.6%`、服务器中位延迟约 `+45.8%`，不符合最终轻量化目标。
- N12 因果审计显示：置零 observation、重置 persistent state、禁用 feature writeback 或 q 路径都会明显降低已训练模型性能；因此 N15 不删除这些已验证环节。
- 未知：40 epoch 失稳是否主要来自八套更新器的过参数化；共享是否会反而造成阶段表达不足。

## 4. 机制与公式

对阶段 `i=1..8`，保留 N12 的固定 bicubic analysis `D`、精确矩阵转置 `D^T`、LR persistent state `h_i`、LR RGB state `q_i`和主干写回。八套 `Phi_i/G_i/P_i/S_i` 改为一套共享核 `Phi/G/P/S`，再以零初始化的通道调制表达阶段差异：

```text
xhat_i = bilinear(y + q_i) + PixelShuffle4(stage_probe(feat_i))
r_i = y - D(xhat_i)
a_i = resize(D^T r_i)
j_i = concat(feat_i, h_i, a_i)
c_i = Phi(j_i) * (1 + tanh(alpha_i)) + beta_i
g_i = sigmoid(G(j_i) + zeta_i)
h_(i+1) = h_i + g_i * (c_i - h_i)
p_i = P(concat(feat_i, h_(i+1), a_i)) * (1 + tanh(pi_i))
s_i = S(concat(h_(i+1), a_i)) * (1 + tanh(sigma_i))
feat_(i+1) = backbone_i(feat_i) + p_i
q_(i+1) = q_i + s_i
```

`alpha_i/beta_i/zeta_i/pi_i` 的形状为 `1×48×1×1`，`sigma_i` 为 `1×3×1×1`。所有调制参数为零时，八阶段执行同一共享动力学。共享 `P/S` 的末层仍零初始化，迁移 B0 权重后初始输出与 B0 一致。

共享 gate 的权重和偏置同样零初始化，使初始 `g_i=0.5`。这是对 SRPRv2 在 0–255 输入标度下随机 gate 大量饱和的直接稳定化；不锁定门控，其权重与阶段偏置均可学习。

## 5. 功能边界

- 持续状态 `h`：跨八阶段累积的隐式重建证据。
- 观测 `a`：由当前 LR 输入与阶段 HR proxy 的数据残差导出，不是可学习状态。
- RGB state `q`：对低频/颜色残差的紧凑显式累积。
- 共享核：表达八阶段共通的重建更新规则。
- 阶段调制：只表达更新强度、偏置和通道选择，不复制完整核。

## 6. 与近邻方法的界线

- 与 LFMN：不改八个 TAB/LRSA/SFML 主干、PixelShuffle decoder 和 bilinear residual；新增的是贯穿主干的显式重建状态闭环。
- 与 N12/SRPRv2：不删减因果审计必要路径，只将八套同构 proximal core 重参数化为“共享核 + 阶段调制”。
- 与普通 recurrent weight sharing：共享的对象是由显式数据观测驱动、同时更新 feature/state/q 的耦合 proximal dynamics；但“共享参数”本身不构成足够新颖性，必须靠稳定增益、效率与消融建立贡献。
- 本轮不加 C1 普通蒸馏，避免把优化收益误认为结构收益。

## 7. 复杂度预算

- 额外参数：相对 B0 `<=2.5%`（目标约 `1.8%`）。
- Conv2d MAC：共享参数不自动减少八次计算，预期仍接近 N12；必须实测。
- 延迟：N15 若精度通过但延迟仍 `>B0 15%`，只允许进入等价算子折叠/加速阶段，不能直接宣称轻量优势。
- 峰值显存：目标 `<=B0 1.10x`。

## 8. 预注册反证

- 共享核或任一阶段调制无梯度：实现失败。
- 置零/重置状态、观测或写回对训练后输出无实质影响：机制假设失败。
- 阶段调制长期均接近零且去除后性能不变：“semi-shared”解释失败，方法退化为普通权重共享。
- 20 epoch 不达到下述闸门：不进入 40 epoch，不加 C1。

## 9. 固定训练协议

```text
data: DIV2K train 1-800 / validation 801-900
scale: x4
HR patch: 256
batch: 4
seed: 1
augmentation: LFMN 默认
optimizer: Adam
lr: 2e-4
scheduler: CosineAnnealingLR, T_max=150, eta_min=1e-6
loss: 1*L1
epochs: 20 first; 40 only after promotion
metric: 与 B0 同一 PSNR/SSIM 实现、逐图指标和边界裁剪
```

## 10. B0 复用决策

Baseline reuse: `YES` for 20 epoch，仅当服务器脚本逐字段核对 N9 B0 `config.txt`、曲线、epoch20 权重和逐图指标全部通过时。候选从共同基线提交构建，本轮不修改影响 B0 的公共前向、训练器或评测代码。任一协议字段不匹配则拒绝运行，不跨协议比较。

## 11. 成功、灰区与停止条件

### 20 epoch 晋级到 40 epoch

必须同时满足：

1. epoch20 PSNR delta `>0` 且末 5 轮平均 delta `>= +0.010 dB`；
2. epoch20 逐图 PSNR bootstrap 95% CI 下界 `>0`，逐图胜率 `>=60%`；
3. epoch20 SSIM 不低于 B0；
4. 无 NaN/梯度断路，状态、observation、writeback 机制诊断非零；
5. 额外参数 `<=2.5%`。

若最终 delta 与末5轮均值为正，但 CI 跨零或胜率为 55–60%，记为灰区；只允许补一个相同协议 seed，不允许改结构或加 C1。其他情况 `NO-GO`。

### 40 epoch 晋级到 150 epoch

必须同时满足：epoch40 delta `>0`、末5轮平均 `>=+0.010 dB`、正向 epoch 占比 `>=80%`、逐图 CI 下界 `>0`、胜率 `>=60%`、SSIM 不退化。未满足则停止，不用 150/1500 epoch 赌反转。

## 12. 后续消融（仅 M1 通过后）

- B0：LFMN + L1。
- M1：SRPR-SS + L1。
- M0：完全共享，去掉阶段调制。
- C0：N12 八套独立 proximal core，区分收益是否来自共享正则。
- C1：B0 + 普通输出蒸馏。
- M1+C1：SRPR-SS + 相同蒸馏，只在 M1 本身通过后运行。

## 13. 当前未验证主张

N15 尚无任何训练结果。“更稳定”、“更高 PSNR”、“更低延迟”和“具有足够论文新颖性”都只是待检验假设，不得写成已证实结论。

## 14. 本地实现验证

PyTorch 2.9.1+cu128 / RTX 4060 Laptop 下已通过：B0 权重迁移后输出逐元素一致；候选模型构造前后的全局 CPU RNG 流与 B0 一致；全网仅有一个 `SharedProximalCore`；首步 writeback/q 及第二步 shared candidate/gate/全部阶段调制梯度均有限非零；显式 `D/D^T` 的 float64 内积最大误差 `8.89e-15`；保存与严格重载输出一致。真实 DIV2K 0001 的 32×32 LR crop 前向、L1 反向和一次 Adam 更新有限。训练器每 epoch 仅采样首个训练 batch 和每个验证集首张图的八阶段诊断，避免诊断 I/O 成为训练干扰。

端到端烟雾使用 DIV2K 1 张训练图和 0801 验证图、x4、HR patch64、batch1，完成一个真实 batch 训练、评测、checkpoint、逐图指标和机制 JSONL/PT 保存。烟雾评测数值只用于链路验证，不是性能结论。

实测 B0/N15 参数为 759,627/773,487，新增 13,860（`+1.8246%`）。推理时机制诊断默认关闭；LR 64×64、FP32、warmup10/repeats50 的同一 profiler 中，Conv2d MAC 为 2.833G/3.228G（`+13.95%`），中位延迟 40.52/44.53 ms（约 `+9.90%`），peak allocated 214.74/218.86 MiB（约 `+1.92%`）。这证明参数预算通过，但尚未证明同等或更低计算/延迟；服务器必须复测。
