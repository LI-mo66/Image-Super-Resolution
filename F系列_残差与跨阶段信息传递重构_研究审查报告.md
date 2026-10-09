# F 系列：LFMN 残差与跨阶段信息传递重构研究审查报告

日期：2026-10-09

状态：`AUDITED / WAITING_FOR_USER_APPROVAL`

本轮权限：只审查和写文档；未修改模型，未运行训练、推理或性能评测。

## 0. 研究重置与结论摘要

本项目从本报告起重置为全新的 `F` 系列。以前的候选、命名、结构、训练结果和失败判断均不作为本轮经验、起点或正负证据；后续候选依次命名为 `F1`、`F2`、`F3`。获批实施时，每个候选仍按项目 SOP 从共同原始 LFMN 基线建立独立分支，例如 `codex/f1-identity-delta-esa`。

当前可以成立的结论只有三条：

1. 原始 LFMN 的每阶段写回确实为 `ESA(F_previous + Delta_F)`；ESA 的 sigmoid 掩码作用于完整的“旧特征+更新量”，代码中没有显式恒等旁路。
2. 这构成可检验风险，但不是已证实缺陷。LFMN 论文的组件消融反而表明，在其整体训练条件下加入 ESA 提高了 Urban100 和 Manga109 的结果。因此必须先测掩码、更新量和梯度，不能先宣布 ESA 过度抑制。
3. 六个候选中只条件性推荐 **F1：恒等携带的增量 ESA 更新**。它直接回答核心问题，新增参数只有 8 个阶段标量，理论主算子数量不增加；但只有诊断门通过且同协议实验为正时，才可称为改进。

## 1. 可复核的基线身份

### 1.1 原作者仓库与 checkpoint

| 项目 | 已核实值 | 证据等级 |
|---|---|---|
| 原作者仓库 | `https://github.com/hehesjtu/LFMN` | 官方仓库 |
| 2026-10-09 核查的远端 HEAD | `4fe1614ef24f03bee81d53f5d7ec7df91606d93b` | `git ls-remote` 与临时浅克隆一致 |
| 官方 x4 checkpoint | `model/scale4_model_939.pt` | 官方仓库文件 |
| 官方/本地 checkpoint SHA256 | `44999471d8cc2d5f7dbf10d354766e08a5d23a84200a1060dfa9a9ed7a7711dd` | 两份文件逐哈希一致 |
| 官方模型源码形态 | `lfmn.cpython-39-x86_64-linux-gnu.so` | 官方仓库；不是可读 Python 源码 |
| 本仓库 Python 重构 | `LFMN/model/lfmn.py` | 本地可读实现；当前 blob `41fc0c6d37c0a99fd194d4cab675115c53c066e4` |
| Python 重构最近修改提交 | `1e51b2068b62a793860e7b7192a43877d9ccc2f1` | 本地 Git |
| x4 Python 模型参数量 | `759,627` | 本机 PyTorch 2.9.1+cu128 构造实测 |

重要边界：官方发布的是编译扩展，当前 Python 文件是可严格加载官方权重的项目重构，不应写成“原作者公开 Python 源码”。当前默认 `eval_refine_iters=0`、`normalize_overlap=False`；新增的可选评测分支默认关闭，不改变下述原始阶段公式。

### 1.2 原始阶段前向

令 `P_i=F_{i-1}`，浅层先验为 `F_S`，则当前 Python 重构的第 `i` 阶段是：

```text
M_i = beta_i(F_S) * P_i + gamma_i(F_S)
T_i = TAB_i(M_i)
S_i = LRSA_i(T_i)
Delta_i = Conv3x3_i(S_i)
Z_i = P_i + Delta_i
m_i = sigmoid(ESA_body_i(Z_i))
F_i = Z_i * m_i
```

因此，`P_i` 虽在 ESA 前以残差形式加入，但最终仍整体乘以 `m_i in (0,1)`。对恒等传播而言，局部 Jacobian 不是显式的 `I + J_update`，而是包含 `diag(m_i)` 与掩码分支导数的组合。它可能抑制、筛选或重新加权旧特征，也可能正是论文消融中有效的空间增强来源。

八个 ESA 共 `31,872` 参数，八个 `mid_conv` 共 `166,272` 参数。ESA 不是参数主项，但其逐阶段乘法位于状态写回点，功能影响可能大于参数占比。

### 1.3 LFMN 论文内证据

本地论文文本给出的 x4 组件消融为：

| 结构 | Params | FLOPs | Urban100 | Manga109 |
|---|---:|---:|---:|---:|
| TAB+LRSA+SFML，无 ESA | 552K | 41.95G | 26.47 / 0.7969 | 30.81 / 0.9130 |
| TAB+LRSA+SFML+ESA | 577K | 42.48G | 26.54 / 0.7987 | 30.86 / 0.9129 |

该消融表中的完整配置为 577K，与当前发布 x4 权重对应的 759,627 参数模型不一致，因此这里只采用同表受控比较的方向，不用表中 Params/FLOPs 推算当前发布模型成本。它支持“ESA 在论文消融配置中有用”，但没有回答以下问题：

- 掩码应作用于 `P_i+Delta_i`，还是只作用于 `Delta_i`；
- 八阶段中是否存在过强抑制或阶段不均衡；
- ESA 的收益来自空间选择，还是来自额外非线性与参数；
- 相似阶段特征是否具有不同功能。

所以本任务是重新审查写回位置，而不是预设删除 ESA。

## 2. 指定参考资料核验

### 2.1 EchoSR

- 论文：*EchoSR: Efficient Context Harnessing for Lightweight Image Super-Resolution*，Information Fusion 2026，DOI `10.1016/j.inffus.2026.104471`，另有 [arXiv:2605.17470](https://arxiv.org/abs/2605.17470)。
- 官方代码：[funnyWang-Echoes/EchoSR](https://github.com/funnyWang-Echoes/EchoSR)，本次核查 HEAD `09999643d2088ddbf39539244b046b9e4a951c3a`（2026-05-19）。
- 代码事实：CHB 将局部聚合、不同核尺度的分支和下采样全局上下文分开；全局支路用通道尺度参数，初始化为 `0.1`；ResidualGroup 最终为 `x + COFB(group(x))`，保留显式组级恒等路径。COFB 使用连续 `7x7`、`15x15` 深度卷积后再写回。
- 消融事实：论文 x2 消融报告移除 GCP 时 Urban100/Manga109 分别下降约 `0.09/0.07 dB`；移除 MRFE 的 identity branch 也退化；COFB 的核顺序和连续覆盖会影响结果。论文的整网收益不能外推为 LFMN 中任意历史融合必然有效。
- 可借鉴内容：分层职责、显式 identity、受控缩放、跨尺度融合顺序。
- 不直接移植：完整 CHB/COFB、多大核堆叠和 FFT loss；它们会改变本任务的变量与延迟预算。

### 2.2 DFFN

- 论文：*Disentangled feature fusion network for lightweight image super-resolution*，Digital Signal Processing 154 (2024), 104697，DOI `10.1016/j.dsp.2024.104697`；出版社页面可核实 DFES、FIM、FFM 与论文元数据。
- 公开机制：固定/学习式特征解耦形成高低频双流；对称层参数共享；FIM 让高低频交互；FFM 用通道加权与分层并行融合减少层级信息损失。
- 代码审计缺口：论文标注的 `https://github.com/zhoujy-aust/DFFN` 在 2026-10-09 实际克隆返回 `Repository not found`。因此本轮**没有**核实代码实现、仓库 commit 或精确消融表，不能把网络描述扩写成代码事实。
- 证据使用规则：只把“高低频职责分开”和“层级信息并行融合”作为候选启发；F1 的优先级不依赖 DFFN。取得可访问的作者代码或全文消融表之前，不声称已复现或已验证其部件收益。

### 2.3 SPAN

- 论文：*Swift Parameter-free Attention Network for Efficient Super-Resolution*，[CVPR Workshops 2024](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Wan_Swift_Parameter-free_Attention_Network_for_Efficient_Super-Resolution_CVPRW_2024_paper.html)，arXiv `2311.12770`。
- 官方代码：[hongyuanyu/SPAN](https://github.com/hongyuanyu/SPAN)，本次核查 HEAD `c77a5917759f09e66fbc7124220c5afc5ee221e5`（2026-02-28）。
- 代码事实：SPAB 先计算卷积输出 `H`，再用 `sigmoid(H)-0.5` 形成无参数注意力，并计算 `(H+x)*attention`；整网只稀疏拼接浅层输出、末级输出、早期块和倒数第二块的中间特征，而非密集拼接所有历史。
- 消融事实：论文在相同 x4、48 通道设置下分别移除块内 residual、参数自由 attention 及两者，结果均支持二者共同作用；重参数化提高部署效率。该证据说明 attention 中的残差补偿值得单独检查，不证明 SPAB 公式适合直接替换 ESA。
- 可借鉴内容：低成本注意力、注意力与 residual 的职责拆分、稀疏选取历史特征、部署时实测速度。

## 3. 待验证问题与证据等级

| 问题 | 当前事实 | 当前证据等级 | 能否下结论 |
|---|---|---|---|
| ESA 是否过度抑制完整残差 | 掩码确实乘在 `P_i+Delta_i` 上 | B：代码事实 | 不能；需 mask/范数/旁路诊断 |
| 是否缺少恒等直通 | 阶段末没有显式绕过 ESA 的 identity | B：代码事实 | 只能称结构风险 |
| 阶段更新量是否异常 | 未对官方 x4 权重测量 | D：未知 | 不能 |
| 历史阶段是否冗余 | 未测相关性与功能删除 | D：未知 | 不能；相似不等于冗余 |
| 是否丢失后重复提取 | 尚无因果干预或频谱轨迹 | D：未知 | 不能 |
| 训练稳定性来源 | 尚无原始 LFMN 梯度/数值轨迹 | D：未知 | 不能 |

证据等级定义：A=同协议重复实验；B=源码/权重可复核事实；C=论文在其它网络上的受控证据；D=待验证假设。

## 4. 诊断阶段 D0（获批后才实施）

### 4.1 固定对象与协议

诊断只使用官方 x4 checkpoint `scale4_model_939.pt`，SHA256 如第 1 节。第一项评测必须是同一 checkpoint、同一 Set5 数据、同一进程配置的 Self-Ensemble OFF/ON 配对；先输出逐图和平均 PSNR/SSIM，再进入任何候选比较。

官方评测代码的关键口径为：

```yaml
scale: 4
rgb_range: 255
quantize: clamp(0,255) -> round
benchmark_psnr: Y-like weighted RGB difference
psnr_coefficients: [65.738, 129.057, 25.064] / 256
psnr_shave: 4
ssim_input: rounded RGB -> MATLAB-style Y + 16
ssim_coefficients: [65.481, 128.553, 24.966]
ssim_shave: 4
chop: fixed and identical between groups
self_ensemble: explicitly recorded, development comparisons OFF
```

OFF/ON 只测自集成本身的增量和实际延迟/显存；以后结构筛选一律 OFF 对 OFF。ON 通常包含 8 个几何变换，但必须实测端到端开销，不能直接报告“8 倍”。

### 4.2 阶段张量与统计

对每个阶段记录 `P_i, Delta_i, Z_i, m_i, F_i`，按图、阶段、通道同时保存汇总，不保存大规模完整特征：

```text
update_ratio_i   = RMS(Delta_i) / (RMS(P_i) + eps)
pre_esa_ratio_i  = RMS(Z_i) / (RMS(P_i) + eps)
mask_mean/std_i  = mean/std(m_i)
mask_sat_i       = mean(m_i<0.05 or m_i>0.95)
esa_retention_i  = RMS(F_i) / (RMS(Z_i) + eps)
identity_drift_i = RMS(F_i-P_i) / (RMS(P_i) + eps)
```

额外记录 p01/p05/p50/p95/p99、最大绝对值、非有限值、通道间离散度和空间掩码熵。只用均值会隐藏少数通道或少数纹理区的饱和。

### 4.3 历史相关性与功能性

- 表示相似：相邻和跨两/四阶段的 centered cosine、linear CKA、每图频带能量相关性。
- 更新相似：`Delta_i` 之间的方向、幅值和频谱重叠。
- 功能干预：同权重分别令单阶段 ESA 为 identity、令 `Delta_i=0`、交换相邻阶段更新，记录最终输出与 PSNR 的变化。
- 解释边界：高相关只说明表示相似，不能直接称功能冗余；旁路后变好也只是该 checkpoint 的依赖诊断，不是独立训练收益。

### 4.4 梯度与数值稳定性

在固定 LR/HR 对上只做前后向诊断，不执行 optimizer step：

- `dL/dP_i`、`dL/dDelta_i`、`dL/dZ_i`、`dL/dm_i` 的 RMS、最大值、有限性和跨阶段比值；
- ESA 与主干参数梯度范数、梯度夹角及零梯度比例；
- FP32 为主，AMP 仅在后续训练协议批准后检查 scaler overflow/跳步；
- 同输入重复前向、保存重载、奇数尺寸和非方形尺寸；
- TAB 离散排序变化与连续路径梯度分开报告。

### 4.5 D0 推进门

只有同时出现以下至少两类信号，才优先实现 F1：

1. 多数图和多个阶段存在持续偏低的 `esa_retention` 或明显 mask 饱和；
2. `Delta_i` 幅值合理，但 `F_i-P_i` 被 ESA 系统性压小或方向改变；
3. 深阶段 `dL/dP_i` 相对 ESA 前显著衰减；
4. 单阶段 identity 旁路在锁定诊断集上呈一致正向，且不是单图驱动。

若这些现象不存在，则不因公式直觉强行推进 F1，转而选择由诊断直接支持的 F2–F6，或停止结构改动。

## 5. F 系列候选机制

以下成本均为解析预估，实施后必须用实际模型、MAC 覆盖率和同硬件延迟复核。所有候选默认保留原始 `SFML -> TAB -> LRSA -> mid_conv` 主体、最终解码和全局图像残差；不叠加完整参考网络。

### F1：恒等携带的增量 ESA 更新（唯一首推）

**瓶颈与证据等级**：原写回让 ESA 同时门控旧特征和新更新，缺少显式 identity；B 级结构事实，性能影响未知。

```text
Delta_i = Conv3x3_i(LRSA_i(TAB_i(SFML_i(F_{i-1}, F_S))))
F_i = F_{i-1} + alpha_i * ESA_i(Delta_i)
```

- 参数/计算/延迟：每阶段一个标量，合计 `+8` 参数；ESA 和主干算子数基本不变，只增加标量乘加。实际 latency/FLOPs/显存待测。
- 权重继承：原参数可全部按键继承；新增 `alpha_i` 初始化为 `1.0`。但函数不等价于原 LFMN，官方 checkpoint 只能作为诊断/热启动来源，不能把零训练输出当基线。
- 初始化：若从零训练，`alpha_i=1`；若热启动，必须配相同官方 checkpoint、相同微调预算的未改 LFMN 对照。
- 风险：ESA 对纯增量的幅值分布可能与原输入不同；恒等累积可能导致特征范数漂移；标量可能不足以适配通道差异。
- 最小实验：D0 通过后，`B0原LFMN` 对 `F1` 的 20e 配对筛选，均用 Cosine `T_max=150` 的前 20e并可无歧义续训。
- 否决条件：D0 无抑制/梯度证据；20e 终点或末 5 轮不正；非有限值、范数单调爆炸；实测延迟显著增加。
- 新颖性：单独的残差缩放与 attention-on-update 都有广泛先例，当前新颖性中低。若阶段动力学、稳定性和跨数据集收益形成统一证据，才可能构成 LFMN 特定的可发表设计。

### F2：残差化 ESA 校正

**瓶颈**：需要在保留原写回函数的同时，直接测“ESA 应完全接管还是只作校正”。

```text
Z_i = F_{i-1} + Delta_i
F_i = Z_i + lambda_i * (ESA_i(Z_i) - Z_i)
```

- `lambda_i=1` 时严格恢复原阶段映射，便于完整继承官方权重；`lambda_i=0` 为 ESA identity 旁路。
- 额外 `8` 参数和少量逐元素运算，主算子不变。
- 最小实验：先在官方 checkpoint 上只训练 `lambda` 的有界诊断，不作为性能结论；若方向一致，再独立配对训练。
- 否决：`lambda` 长期停在 1 附近且 F1 所需诊断不成立；或偏离 1 后验证集下降。
- 新颖性：低，更适合作为机制桥梁和强消融，不优先作为论文主候选。

### F3：稀疏 dyadic 历史锚点

**瓶颈**：若 D0 显示非相邻阶段包含互补方向，而相邻阶段高度相似，可避免密集 concat，只选择稀疏历史。

```text
A_i = softmax(w_i0,w_i2,w_i4) weighted_sum(F_0, F_{i-2}, F_{i-4})
U_i = Delta_i + beta_i * (A_i - F_{i-1})
F_i = F_{i-1} + alpha_i * ESA_i(U_i)
```

- 仅使用存在的锚点；早期阶段自动缩短集合。第一版不加 1x1 投影。
- 参数约每阶段 2–5 个标量；计算为少量逐元素加权，但训练需保存稀疏历史激活，峰值显存风险高于 F1。
- 权重继承：原参数可继承；历史权重初始化为只选最近合法锚点，`beta=0` 使初始不注入历史。
- 最小实验：先以冻结特征验证被选锚点是否提供与当前更新不同且与重建误差相关的方向，再决定是否训练。
- 否决：只有表示低相关而无功能增益；历史权重塌缩；峰值显存或延迟不合格。
- 新颖性：中等风险。稀疏历史融合并不新，价值取决于与 LFMN 阶段更新的可解释选择规则。

### F4：增量状态的轻量指数记忆

**瓶颈**：若价值信息存在于过去更新而不是过去完整特征，存 `Delta` 的摘要比反复读取整阶段特征更直接。

```text
H_i = rho_i * H_{i-1} + (1-rho_i) * Delta_i
F_i = F_{i-1} + alpha_i * ESA_i(Delta_i + beta_i * H_i)
```

- 新增每阶段 `rho/alpha/beta` 标量和一个 `Bx48xHxW` 状态；参数极少，但逐阶段读写会增加带宽和激活内存。
- `beta=0` 初始不使用历史；`rho=sigmoid(r_i)`，初始化对应 `0.5`，避免直接饱和。
- 权重可继承；候选函数初始等价于 F1 而非原 LFMN。
- 最小实验：比较 `H_i` 与 `Delta_i` 的方向、频带和最终误差相关性；仅在记忆带来新增信息时训练。
- 否决：`H_i` 与当前更新近乎等价、状态范数漂移、带宽导致延迟超预算。
- 新颖性：中低；EMA 状态常见，必须依靠明确的增量语义和实证区别于普通 recurrence。

### F5：高低频职责分离的单 ESA 更新

**瓶颈**：若 D0 显示低频结构更新被空间注意力不必要地抑制，而高频更新需要选择，可让 ESA 只处理高频增量。

```text
L_i = AvgPool3x3(Delta_i)
H_i = Delta_i - L_i
F_i = F_{i-1} + alphaL_i * L_i + alphaH_i * ESA_i(H_i)
```

- 每阶段两个标量，`+16` 参数；保留一次 ESA，新增固定 3x3 平均滤波和逐元素运算。
- 原参数可继承；`alphaL=alphaH=1`，但不恢复原函数。
- 与 DFFN 的关系仅限“高低频职责分开”的论文启发；由于 DFFN 代码未取到，不声称复制其 FDM/FIM/FFM。
- 最小实验：先验证 `L_i/H_i` 的能量、与误差方向及 ESA mask 关系；固定滤波边界必须检查。
- 否决：高低频分量不能形成稳定互补、边界伪影、平均池带来可测延迟但无精度收益。
- 新颖性：中低；固定频率分解已有大量先例，需要跨阶段作用机制才可能提升贡献度。

### F6：近无参数幅值注意力替代 ESA

**瓶颈**：若 ESA 的参数化掩码冗余且延迟占比明显，可使用只作用于增量的无参数幅值调制，同时保留 raw update。

```text
C_i = mean_hw(abs(Delta_i))
A_i = 2 * sigmoid(abs(Delta_i) - C_i)
F_i = F_{i-1} + alpha_i * Delta_i * A_i
```

- `A_i` 位于 `(0,2)`，可相对抑制或增强更新，同时不改变 `Delta_i` 的符号。删除八个 ESA 时理论减少 `31,872` 参数；新增 `alpha` 共 `8` 个标量，净减少 `31,864` 参数。是否减少 FLOPs 和延迟必须实测，`abs/mean/sigmoid` 可能受内存带宽和 kernel launch 限制。
- 可继承除 ESA 外的权重；ESA 权重明确列为丢弃，不使用 `strict=False` 隐藏。
- 初始化 `alpha=1`；动态中心 `C_i` 无持久状态，从零训练优先。
- SPAN 只提供“无参数注意力+残差信息补偿”受控证据；这里不是 SPAB 复制，也不能引用其整网收益作为预期涨点。
- 最小实验：先 profile ESA 实际耗时；若不是瓶颈，不以轻量化名义推进。
- 否决：主要尺寸不加速；更新只增强不抑制导致伪纹理；20e PSNR/SSIM 下降。
- 新颖性：低到中；更可能是效率消融，除非能证明新的稳定性/效率边界。

## 6. 为什么只首推 F1

F1 用最少变量直接检验本任务最核心的因果问题：**空间注意力应选择新增修正，而不应反复改写已经存在的阶段状态**。它不需要历史缓存、频率分支或完整参考模块，参数和理论算子预算最小，也与用户指定公式一致。

审稿人视角下，F1 的最大弱点是概念本身不够新。论文价值不能来自改一行残差公式，而必须来自完整证据链：原结构的阶段抑制现象、F1 对该现象的定向修复、跨 seed 和五 benchmark 的稳定收益、以及几乎不变的真实成本。如果诊断不支持上述链条，F1 应停止，而不是再叠 F3/F5 来凑创新点。

## 7. 获批后的实验顺序

### 7.1 阶段 A：只读评测与 D0

1. 官方 checkpoint Set5 x4：OFF/ON 各一次，逐图配对，记录 latency 与 peak memory。
2. OFF 状态下完成阶段范数、mask、相关性、梯度和冻结干预。
3. 输出 D0 报告；不修改模型，不训练。

### 7.2 阶段 B：F1 工程实现

只有用户批准后：

```text
branch:  codex/f1-identity-delta-esa
model:   LFMN/model/lfmnf1.py
check:   repro/check_f1_identity_delta_esa.py
run:     repro/run_f1_screen_server.py/.sh
card:    创新方案/F1_恒等携带增量ESA_实验卡.md
```

实现检查覆盖 shape、奇数/非方尺寸、有限输出/梯度、optimizer 覆盖、真实更新、严格保存重载、参数量、FLOPs 覆盖率、延迟和峰值显存。共同 `lfmn.py` 不覆盖。

### 7.3 阶段 C：20e 配对筛选

`B0` 与 `F1` 从零、同 seed、同 batch 流、同初始化规则、同优化器和同 `Cosine T_max=150` 启动计划150e训练；在epoch 20保存模型、优化器、scheduler和数据随机流状态后强制暂停。开发比较全部Self-Ensemble OFF。checkpoint选择只使用预先锁定的独立验证集规则，不使用任何benchmark挑权重。

epoch20使用固定端点checkpoint分别评测Set5、Set14、B100、Urban100和Manga109，逐数据集列出结果，不计算跨数据集宏平均；每个数据集仍保留标准逐图均值和逐图明细，Set5五张图全部列出。报告终点、末5轮验证均值、逐图中位数/胜率/bootstrap、SSIM、训练非有限事件、参数、MAC/FLOPs、延迟和峰值显存。20e只用于筛选，不作为论文收益；未经用户审阅和明确批准不得恢复到epoch150。

### 7.4 阶段 D：150e 与最终评测

- `Delta PSNR <= 0`：淘汰或只排查工程错误。
- `0–0.03 dB`：重复验证，不直接长训。
- `0.03–0.05 dB`：保留并做多 seed。
- `>=0.05 dB`：优先独立复验，再申请更长预算。

150e 必须是B0/F1同一150e Cosine协议，不与其它时间轴的前150e比较。依据独立验证集的预注册规则分别确定B0/F1 checkpoint后，依次报告Set5、Set14、B100、Urban100、Manga109的OFF，不计算跨数据集平均。本轮不运行Self-Ensemble ON；1000e仍需用户单独批准。

## 8. 接口与状态表

| 对象 | 形状/layout | 范围/归一化 | 生命周期 | 梯度 |
|---|---|---|---|---|
| LR 输入 | `B x 3 x H x W` NCHW | `rgb_range=255` | 每图 | 有 |
| `F_S` | `B x 32 x H x W` | Conv+ReLU，无显式归一化 | 每图共享八阶段 | 有 |
| `F_i/P_i` | `B x 48 x H x W` | 无全局归一化 | 阶段状态 | 有 |
| `Delta_i` | `B x 48 x H x W` | 3x3 conv 输出 | 当前阶段 | 有 |
| ESA mask | `B x 48 x H x W` | sigmoid `(0,1)` | 当前阶段 | 有 |
| TAB means | `num_tokens x 48` | 参与 cosine 前归一化 | 训练 EMA buffer；eval 读取 | 更新不走梯度 |
| LRSA patch | token 顺序来自 NCHW 展平/还原 | 当前默认 legacy overlap | 当前阶段 | 有；排序另审计 |
| 最终 SR | `B x 3 x 4H x 4W` | 最后 conv + bilinear 全局残差 | 每图 | 有 |

任何 F 候选都不得靠隐式广播、任意 reshape 或 `strict=False` 掩盖接口差异。mask、历史状态、detach 点、padding 和 overlap 口径必须写入候选卡。

## 9. 需要用户批准的唯一下一步

F1已经完成隔离实现与Gate 0/1。下一步先补齐公共训练日志机制、D0诊断和Gate 2真实数据烟雾测试；随后按本报告锁定的Cosine 150时间轴启动B0/F1，在epoch20强制暂停。seed、patch、batch、优化器、初始学习率和eta_min必须在启动前写入实验卡，不能临时推断。

## 10. 来源

- [LFMN 官方仓库](https://github.com/hehesjtu/LFMN)
- [EchoSR 官方仓库](https://github.com/funnyWang-Echoes/EchoSR)
- [EchoSR arXiv](https://arxiv.org/abs/2605.17470)
- [EchoSR DOI](https://doi.org/10.1016/j.inffus.2026.104471)
- [DFFN DOI / 出版社页面](https://doi.org/10.1016/j.dsp.2024.104697)
- [DFFN 论文标注但当前不可访问的代码地址](https://github.com/zhoujy-aust/DFFN)
- [SPAN 官方仓库](https://github.com/hongyuanyu/SPAN)
- [SPAN CVPRW 2024 论文页](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Wan_Swift_Parameter-free_Attention_Network_for_Efficient_Super-Resolution_CVPRW_2024_paper.html)
