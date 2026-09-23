# SRPR 共享边界零训练审计卡

## 目的

N12/SRPRv2 在 epoch 20 有弱正信号，但参数和延迟过高；N15/SRPR-SS 将 candidate、gate、writeback 和 q-update 全部共享后，epoch 20 相对 B0 为 `-0.007008 dB`，逐图 95% CI 为 `[-0.012066,-0.002061] dB`。本审计不训练新模型，直接使用 N12 epoch-20 checkpoint，判断四类更新中哪些具有跨阶段可交换性，避免继续凭直觉设计 N16。

## 审计对象

1. `candidate`：状态候选更新 `Phi_i`。
2. `gate`：状态路由 `G_i`。
3. `writeback`：对 LFMN 主干特征的直接修正 `P_i`。
4. `q_update`：LR RGB 残差状态更新 `S_i`。

## 干预方法

- `stage ablation`：只在指定阶段将 candidate 设为当前 state、gate 设为 0.5、writeback 设为0或 q-update 设为0，测量阶段边际敏感性。
- `single donor sharing`：其余路径不变，强制八阶段均调用某一已训练阶段的同类组件。对八个 donor 全部测试，避免挑选单一阶段。
- `cyclic/repeated-spec swap`：使用相邻循环替换和 `(1,5),(2,6),(3,7),(4,8)` 同 LFMN 规格替换，作为可交换性辅助证据，不直接当作参数共享方案。
- `partition confirmation`：在屏幕子集选出每类组件损失最小的 donor，再在 DIV2K 801–900 确认单组件共享、`candidate+gate`、`writeback+q_update` 和全共享。

## 数据与口径

- checkpoint：N12/SRPRv2 epoch 20，不更改权重。
- 屏幕集：DIV2K validation 801–820，固定前20张，仅用于排序 donor 和阶段敏感性。
- 确认集：DIV2K validation 801–900，报告逐图 PSNR/SSIM 差和 paired bootstrap 95% CI。
- 评测：复用仓库 `quantize/calc_psnr/calc_ssim`，与 N12 已保存 epoch20 逐图指标核对；不匹配即停止。

## 解释边界

- donor 强制共享是对“已训练阶段功能可交换性”的严格压力测试，不等价于从零共享训练的最优性能。
- 明显负值可以证明该组件不能被直接无条件共享；接近0只能说明“值得实现后验证”，不能证明新候选一定有效。
- 预定建议线：确认集最佳 donor 相对 Full N12 的平均差 `>=-0.003 dB` 为“可考虑共享”；`[-0.010,-0.003)` 为不确定；`<-0.010 dB` 为保留阶段特异性。

## 输出

`srpr_sharing_audit.json`保存全部屏幕干预、确认结果、逐图差、CI、阶段敏感性和自动建议；`srpr_sharing_audit.txt`提供可直接粘贴的摘要。本审计不生成新 checkpoint，不修改 N12/N15 结果目录。
