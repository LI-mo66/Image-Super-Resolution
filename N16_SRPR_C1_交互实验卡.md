# N16：SRPRv2 × 普通输出蒸馏交互实验卡

## 1. 身份信息

```text
候选编号：N16
候选名称：SRPRv2 × C1 interaction
候选类型：训练交互诊断（结构候选 + 已验证训练策略）
当前状态：PROMOTED（20/40/150 epoch均通过；固定epoch150五benchmark待执行）
分支：codex/n16-srpr-c1-interaction
共同基线提交：b198de7cd2b3a8ddf463c573278b1385a223adf7
负责人/AI：Codex
日期：2026-09-29
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

## 12. 20-epoch服务器结果

服务器提交为 `6e2442619c72d33dfc195fecf7d6e01a197de08d`，结果目录为
`experiment/all_runs/n16/srpr_c1_20e_seed1_6e24426_r3`，包装退出状态为0。主比较
`SRPRv2+C1 - B0+C1` 如下：

```text
epoch20 PSNR delta:       +0.038300 dB
last-5 mean delta:        +0.015380 dB
positive epochs:          19/20
epoch20 SSIM delta:       +0.0008213
paired-image mean delta:  +0.038288 dB
paired-image win rate:    92.0%
bootstrap 95% CI:         [+0.029417,+0.048775] dB
paired-image SSIM delta:  +0.0008209
```

六项20轮预注册闸门全部通过，自动决定为
`PROMOTE_TO_MATCHED_40E_CONTINUATION`。该结论只批准严格配对40轮续训，不等于长期性能或
论文创新已经成立。

## 13. 40-epoch续训完整性要求

两组分别从自己的epoch20目录复制到独立40轮目录，并恢复模型、Adam状态、Cosine scheduler
状态、损失日志、PSNR/SSIM历史及RGCRD日志。训练DataLoader使用seed1重新构造后，先以空数据
迭代精确推进20个epoch的generator消耗，再开始epoch21，避免续训随机裁剪流从epoch1重启。

续训前必须验证：源目录状态为0、20轮自动决定已通过、两组曲线恰好20点、scheduler
`last_epoch=20/T_max=150`、optimizer state非空、蒸馏日志恰好20行、epoch20逐图指标恰好100张，
以及模型/优化器/scheduler复制前后SHA256一致。40轮汇总还必须证明复制后的前20轮曲线逐元素
不变，并检查最终scheduler为`last_epoch=40`。

## 14. 40-epoch结果与一次性外部基准闸门

严格续训退出状态为0，且`first20_history_exact=True`。相对C1，SRPR+C1在epoch40最终/末5轮
分别为`+0.031443/+0.026326 dB`，续训20轮中19轮为正；100图均值`+0.031436 dB`、胜率
90%、95% CI `[+0.021654,+0.044618] dB`，SSIM `+0.0004949`。四项40轮预注册闸门全部
通过，自动决定为`PROMOTE_TO_150E_CONFIRMATION`。

在150轮前允许一次固定epoch40外部评测，数据集锁定为Set5、Set14、B100、Urban100和Manga109，
两组分别加载自己的`model_40.pt`，×4、无self-ensemble、不调参、不选择benchmark最佳checkpoint。
主比较仍为`SRPR+C1-C1`。该评测检验跨数据集泛化，不能替代150轮对训练后期稳定性的检验。

预注册解释：Urban100与Manga109都为正、至少4/5数据集平均PSNR为正且没有数据集低于
`-0.01 dB`，记为`BROAD_EXTERNAL_SUPPORT`；至少3/5为正且无大幅退化记为混合支持；否则
记为弱支持。benchmark结果只决定是否值得立即投入150轮，不允许用于调结构或选择checkpoint。

固定epoch40端点的实际结果为：Set5、Set14、B100、Urban100、Manga109分别提升
`+0.056861`、`+0.047473`、`+0.026658`、`+0.067673`、`+0.157692 dB`，五组逐图
bootstrap 95% CI下界均大于0，合并328图均值为`+0.084056 dB`、胜率83.2%。自动判定为
`BROAD_EXTERNAL_SUPPORT`，因此批准严格40→150续训；不据此修改结构、KD权重或checkpoint。

## 15. 150-epoch严格续训与预注册闸门

两组从各自epoch40状态继续，必须恢复模型、Adam和Cosine scheduler，保持`T_max=150`，并重放
40轮DataLoader generator消耗。输出目录与40轮源目录隔离，复制前后校验模型、优化器和调度器
SHA256；汇总必须证明前40轮PSNR/SSIM历史逐元素不变，最终scheduler为`last_epoch=150`。

150轮通过必须同时满足：

1. epoch150 PSNR增量`>= +0.020 dB`；
2. 最后10轮平均增量`>= +0.015 dB`；
3. epoch41–150至少88/110轮为正；
4. DIV2K 100图paired bootstrap 95% CI下界`> 0`；
5. epoch150 SSIM不低于C1。

全部通过时自动记为`PROMOTE_TO_LONG_RUN_VALIDATION`；否则记为`STOP_OR_REVIEW_N16`，不得用最佳
epoch替代终点，也不得未经审计直接投入1000/1500轮。150轮通过后，才使用固定epoch150端点
复测五benchmark，并进入多seed、复杂度和四组交互归因。

新增资产：

```text
resume check: repro/check_n16_150e_resume.py
run:          repro/run_n16_srpr_c1_150e_server.{py,sh}
summary:      repro/summarize_n16_srpr_c1_150e.py
summary test: repro/check_n16_150e_summary.py
```

## 16. 150-epoch服务器结果

服务器提交为`6a683288f7e8905a7be5e5112fa7817a54736b4f`，结果目录为
`experiment/all_runs/n16/srpr_c1_150e_seed1_6a68328_r2`，包装退出状态为0，全部关键模型、优化器、
调度器和汇总文件存在，且`first40_history_exact=True`。相对C1，SRPR+C1结果为：

```text
epoch40 delta:              +0.031443 dB
epoch50 delta:              +0.034922 dB
epoch75 delta:              +0.044744 dB
epoch100 delta:             +0.040394 dB
epoch125 delta:             +0.045511 dB
epoch150 delta:             +0.046227 dB
last-10 mean delta:         +0.046059 dB
last-20 mean delta:         +0.046088 dB
epoch41-150 positive:       110/110
epoch150 SSIM delta:        +0.0011098
paired-image mean delta:    +0.046229 dB
paired-image median delta:  +0.035883 dB
paired-image win rate:      97.0%
bootstrap 95% CI:           [+0.038581,+0.055531] dB
paired-image SSIM delta:    +0.0011098
```

五项预注册闸门全部通过，自动决定为`PROMOTE_TO_LONG_RUN_VALIDATION`。该结果证明在当前单seed、
Cosine `T_max=150`、普通输出蒸馏协议内，SRPRv2相对C1的结构增量不仅未在后期消失，而且在最后
20轮稳定保持约`+0.046 dB`。下一步只允许使用固定epoch150端点复测五benchmark，并开展推理
复杂度和多seed确认；不得根据外测结果回调结构、KD权重或checkpoint。

固定epoch150五benchmark评测沿用epoch40外测口径：Set5、Set14、B100、Urban100、Manga109
×4、无self-ensemble、两组相同评测代码和逐图配对统计。只允许加载各自`model_150.pt`，入口在
创建输出目录前严格检查150轮汇总决定、checkpoint结构加载、SHA256和数据集数量。外测仍使用
既定`BROAD_EXTERNAL_SUPPORT`判据，仅用于确认长训后的跨数据集泛化，不用于选择epoch或回调
结构。入口为`repro/run_n16_benchmarks_150e_server.py`，合成验证为
`repro/check_n16_benchmark_150e.py`。

## 17. 固定epoch150五benchmark结果

评测提交为`c703d36856a262a70046edc9b7fe833ddd89fc4f`，输出目录为
`experiment/all_runs/n16/benchmark_epoch150_c703d36_r1`，包装退出状态为0。固定两组
`model_150.pt`、×4且无self-ensemble，结果如下：

| 数据集 | C1 PSNR | SRPR+C1 PSNR | PSNR增量 | SSIM增量 | 胜率 | 95% CI |
|---|---:|---:|---:|---:|---:|---|
| Set5 | 32.102416 | 32.098894 | -0.003522 | +0.0003293 | 40.0% | [-0.102767,+0.093223] |
| Set14 | 28.548450 | 28.590853 | +0.042403 | +0.0009214 | 57.1% | [+0.005173,+0.086618] |
| B100 | 27.539085 | 27.563725 | +0.024640 | +0.0010816 | 78.0% | [+0.016833,+0.033328] |
| Urban100 | 25.995297 | 26.079735 | +0.084438 | +0.0023650 | 83.0% | [+0.052297,+0.116805] |
| Manga109 | 30.087600 | 30.279747 | +0.192146 | +0.0019643 | 97.2% | [+0.168358,+0.216918] |

4/5数据集PSNR为正，Urban100与Manga109均显著为正，没有数据集低于`-0.01 dB`；合并328图
均值`+0.098865 dB`、胜率84.5%、bootstrap 95% CI `[+0.083459,+0.114254] dB`，自动判定
为`BROAD_EXTERNAL_SUPPORT`。Set5中心值为`-0.003522 dB`，其5图CI大幅跨0且SSIM为正，应记为
不确定/近零而非确定退化，也不得把本轮表述成5/5提升。相较epoch40，Urban100和Manga109增量
扩大，而Set14/B100大致保持；这只形成“收益可能偏向复杂纹理”的待验证假设，不能仅凭数据集
均值直接作机制结论。

## 18. 整体创新定位、近邻边界与效率审计

N16后续暂以“蒸馏稳定的持久重建过程”作为**待证伪的整体框架定位**，不把普通输出KD本身
声称为新损失，也不声称现有阶段“独一无二”或“无人相近”。截至2026-10-04检索到的主要近邻
包括：USRNet的展开式数据/先验迭代、PropMambaSR的跨层隐藏状态传播、DVMSR的轻量状态空间
超分与输出蒸馏、FAKD等轻量SR知识蒸馏，以及UCAN/ASID的跨层共享或信息蒸馏。它们分别覆盖
本方案的状态传播、数据一致性、教师监督或跨层组织思想，因此禁止作无边界的首创声明。

当前可检验的窄差异是：SRPR在LFMN八阶段中显式维护由LR观测残差、bicubic分析算子及其转置
回投影驱动的LR持久重建状态，并把状态写回主干特征和LR重建变量；C1仅在训练期从教师最终输出
提供约束。现有150轮结果证明二者组合相对C1有稳定结构增量，但尚未证明KD专门稳定了状态轨迹，
也未证明收益不是两个独立主效应的相加。因而论文整体命名只有在四组交互归因和状态诊断支持后
才能升级为统一机制；否则应诚实表述为“SRPR结构在KD训练下的最佳配置”。

预注册四组归因量为：

```text
interaction = (SRPR+C1 - C1) - (SRPR - B0)
```

需要在相同150轮协议、至少补充seed下估计该量，并比较SRPR与SRPR+C1的逐阶段state norm、
state update、feature writeback、q update与data-consistency ratio。只有交互项稳定为正，且KD使
后期状态轨迹更稳定或更接近有效重建方向，才支持“蒸馏稳定持久重建”这一因果叙事。当前不修改
已成功模型，也不立即增加中间蒸馏损失，以免把归因实验和新候选混在一起。

在多seed和四组长训前，先执行固定epoch150 checkpoint的同硬件推理审计。入口
`repro/profile_n16_efficiency.py`严格加载C1和SRPR+C1的各自权重，固定batch=1、FP32、×4，
默认测LR 64×64与128×128，采用轮换顺序的10次预热和50次计时，报告首次冷启动、稳态median/
P90、CUDA峰值/增量显存及参数量。由于自定义einsum和显式bicubic算子缺少依赖安全且完整的
精确计数器，本轮不伪报MAC/FLOPs；后续如加入经校验的算子级计数，再单独补报。参数量已由
结构检查确定为C1 `759627`、SRPR+C1 `841563`，即`+81936/+10.79%`；教师不进入推理。
