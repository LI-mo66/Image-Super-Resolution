# N16：SRPRv2 × 普通输出蒸馏交互实验卡

## 1. 身份信息

```text
候选编号：N16
候选名称：SRPRv2 × C1 interaction
候选类型：训练交互诊断（结构候选 + 已验证训练策略）
当前状态：APPROVED
分支：codex/n16-srpr-c1-interaction
共同基线提交：b198de7cd2b3a8ddf463c573278b1385a223adf7
负责人/AI：Codex
日期：2026-09-27
```

本轮不是把 C1 包装成新的结构创新，而是对已经出现“20 epoch 正、40 epoch 反转”的
SRPRv2 做最后一次、可证伪的优化交互测试。

## 2. 一句话核心假设

由于 SRPRv2 的持久状态路径在早期产生正收益、但后期可能受到输出目标漂移影响，已验证有效的
普通输出蒸馏 C1 可能稳定其结构增量；若该解释成立，`SRPRv2+C1` 应在相同 Cosine 协议下稳定
超过 `B0+C1`，而不只是超过无蒸馏 B0。

## 3. 已有证据与反证

- N12/SRPRv2：20 epoch 相对 B0 约 `+0.0128 dB`，40 epoch 约 `-0.0044 dB`。
- N12 因果审计：state、observation、writeback、q 路径都被已训练模型使用。
- N12 代价：参数约 `+10.8%`，服务器延迟约 `+45.8%`。
- C1 在 MultiStep 40-epoch RGCRD 对照中相对 B0 最终约 `+0.0317 dB`，但该结果不能跨协议
  直接复用到本轮 Cosine 实验。
- 跨阶段共享与 gate 低秩审计均未产生可实施方案，因此本轮不再修改 SRPRv2 结构。

## 4. 机制和比较对象

两组都从零训练，并使用同一个冻结 SwinIR 教师：

```text
C1：       LFMN + L1(student, HR) + 0.1 L1(student, teacher)
SRPR+C1：  SRPRv2 + L1(student, HR) + 0.1 L1(student, teacher)
```

主效应定义为：

```text
delta_SRPR_given_C1(e) = PSNR(SRPRv2+C1, e) - PSNR(B0+C1, e)
```

输出蒸馏只在训练期存在。C1 不改变推理参数和计算；SRPRv2 的推理代价保持不变。

## 5. 保持、替换和新增

| 项目 | 处理 |
|---|---|
| LFMN 主干、PixelShuffle 尾部、全局残差 | 保持 |
| SRPRv2 八阶段 state/observation/writeback/q | 原样复用已验证 N12 实现 |
| C1 输出蒸馏 | 两组完全相同 |
| RGCRD 关系/演化损失 | 不使用 |
| 公共 LFMN 默认行为 | 不修改 |

## 6. 协议指纹与基线复用

```text
data_train: DIV2K 1-800
validation: DIV2K 801-900
scale: 4
HR patch: 256
batch: 4
seed: 1
optimizer: Adam
learning_rate: 2e-4
scheduler: cosine
T_max: 150
eta_min: 1e-6
epochs_screen: 20
loss: 1*L1 + 0.1*output KD
metric: repository PSNR/SSIM, full 100-image paired metrics
```

Baseline reuse：**NO**。既有 C1 使用 MultiStep，调度器与本轮不同；本轮必须重新训练 `B0+C1`。
无蒸馏 B0/N12 曲线只作为历史上下文，不进入本轮主效应统计。

## 7. 复杂度预算

- C1 推理：与 B0 相同，约 759,627 参数。
- SRPRv2+C1 推理：约 841,563 参数；与原 N12 相同。
- 教师只在训练期使用，显存通过 microbatch 控制。
- 本轮不宣称轻量性改善；只有性能稳定性通过后才重新评估是否值得保留 SRPR 推理开销。

## 8. 可证伪预测

若假设成立：`SRPR+C1 - C1` 在后期仍为正，末 5 轮不回落，逐图优势覆盖多数验证图，且
SSIM 不退化。

若只看到 `C1 > 无蒸馏 B0`，但 `SRPR+C1 <= C1`，则收益来自普通输出蒸馏，SRPR 关闭。

## 9. 20-epoch 预注册闸门

进入 40 epoch 必须同时满足：

1. epoch 20 PSNR 增量 `>= +0.010 dB`；
2. 末 5 轮平均增量 `>= +0.010 dB`；
3. 20 轮至少 15 轮为正；
4. 100 图 paired bootstrap 95% CI 下界 `> 0`；
5. 逐图胜率 `>= 60%`；
6. epoch 20 SSIM 不低于 C1。

灰区：终点和末 5 轮均为正，但未达到上述幅度或 CI/胜率门槛。灰区只允许补一个同协议 seed，
不允许改结构、调权重或挑最佳 epoch。

停止：终点或末 5 轮非正、逐图 CI 上界不大于 0、或 SSIM 明显退化。停止后不进入 40/150/1500
epoch，不把 `SRPR+C1 > 无蒸馏 B0` 解释为 SRPR 成功。

## 10. 40-epoch 与后续纪律

只有通过 20 epoch 才编写/运行续训。40 epoch 仍需相对 `B0+C1` 满足最终与末 5 轮均
`>= +0.010 dB`、CI 下界大于 0、SSIM 不退化，才允许进入 150 epoch。由于 SRPR 推理开销较高，
最终若没有约 `+0.02 dB` 以上的稳定增量或后续可验证的效率改进，不作为轻量 SR 主方案。

## 11. 输出资产

```text
model:     LFMN/model/lfmnsrprv2.py
check:     repro/check_n16_srpr_c1.py
run:       repro/run_n16_srpr_c1_screen_server.sh
summary:   repro/summarize_n16_srpr_c1.py
output:    experiment/all_runs/n16_srpr_c1_*/{c1,srpr_c1}
```
