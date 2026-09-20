# N9：TAB 替换结构设计与论文级预评审

日期：2026-09-20
状态：结构已冻结并完成 A0/A1/C0 实现与本地链路验证；未进行正式训练，无性能结论。

## 1. 结论先行

推荐把首个待验证结构收缩为 **Prior-Conditioned Soft Token Router（PCSTR，先验条件软 Token 路由器）**：保留原 TAB 的外层残差、LayerNorm 和原 `ConvFFN`，只替换固定 EMA centers、硬 `argmax/sort`、IASA 和 IRCA。首版使用每图像动态 Token、`Fs+Fm` 联合生成分配、固定 `K=32`、4 头、Q/K 总维度 24、V 总维度 48；训练和推理完全同构。

这不是“已定位到 LFMN 的真实瓶颈”，也不是可直接承诺提升的方案。现有证据只支持以下较弱判断：

- **文献事实**：LFMN Table III 中，加入 TAB 使参数从 404K 增至 511K、FLOPs 从 34.14G 增至 41.05G，而 Urban100 ×4 / Manga109 ×4 的 PSNR 分别只变化 `+0.01/+0.07 dB`。这是累计式消融，不是等容量、等算力、随机种子统计充分的 TAB 因果试验。
- **源码事实**：当前复现的 8 个 TAB 共 211,584 个可训练参数，每阶段 26,448；其阶段 Token 数为 `[16,32,64,128,16,32,64,128]`，不是固定 8。训练态更新当前 batch 的中心，评测态默认直接使用 EMA 中心，且硬分组路径无梯度。
- **已有实验事实**：N5 在 DIV2K 100 图上将评测中心做一次图像自适应细化，仅得到 PSNR `+0.003380 dB`、SSIM `-0.000092`。因此“固定中心本身就是主要瓶颈”没有得到支持。
- **结构推断**：PCSTR 同时改变分配可微性、Token 生成方式、Token 间交互和训练/推理一致性，检验范围明显大于 N5；但正因改变较大，若没有等参数对照，结果无法归因于“软路由”。
- **待验证假设**：对 Urban100 的重复结构和 Manga109 的非局部线条/网纹，按当前图像、当前阶段形成的软 Token 可能比训练集 EMA 原型和硬排序更适合；也可能因低秩聚合而抹平像素级细节。

在 A/B/C 三个候选中，推荐顺序为 **A（PCSTR）> C（LMLT 式多尺度）> B（PSA 大窗口）**。推荐 A 的原因不是预期增益最大已被证明，而是它最直接检验当前 TAB 特有的硬聚类与训练/推理不一致问题，同时能保持全图内容相似交互；B 与现有 LRSA 的功能重叠最强，C 的效率与稳定性更有吸引力，但作为论文创新与 LMLT/SAFMN 的重合风险更高。

## 2. 文献边界与近邻关系

### 2.1 已有工作已经覆盖什么

- [CATANet（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CATANet_Efficient_Content-Aware_Token_Aggregation_for_Lightweight_Image_Super-Resolution_CVPR_2025_paper.html)：共享训练期更新的 Token centers，用内容分组、IASA 和 IRCA 建立远距离相似区域交互。LFMN 的 TAB 基本沿用这一机制。
- [SRFormer（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_SRFormer_Permuted_Self-Attention_for_Single_Image_Super-Resolution_ICCV_2023_paper.html)：PSA 通过通道/空间置换降低大窗口注意力负担。它解决的是大窗口成本，不是图像自适应聚类。
- [ATD（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Transcending_the_Limit_of_Local_Window_Advanced_Super-Resolution_Transformer_with_CVPR_2024_paper.html)：训练集级 Token dictionary、测试图像自适应细化、dictionary cross-attention 与 category-based attention。2026 年扩展稿还明确包含 ATD-light。因此“字典/Token + 图像适配 + 全局交互”不能作为 N9 的独占创新点。
- [LMLT（ICCVW 2025）](https://openaccess.thecvf.com/content/ICCV2025W/AIM/html/Kim_LMLT__Low-to-high_Multi-Level_Vision_Transformer_for_Lightweight_Image_Super-Resolution_ICCVW_2025_paper.html)：按通道拆头，在不同下采样尺度做注意力，并将低分辨率全局信息向高分辨率局部特征传递。
- [TokenLearner（NeurIPS 2021）](https://proceedings.neurips.cc/paper_files/paper/2021/file/6a30e32e56fce5cf381895dfe6ca7b6f-Paper.pdf)：由输入动态生成少量空间权重图并聚合成 Token。N9 的“软分配聚合”与它高度相邻。
- [Slot Attention（NeurIPS 2020）](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)：以注意力做迭代式软聚类并形成 slots。若 N9 加入多轮 Token 递推，将进一步靠近该路线。
- [SPIN（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Lightweight_Image_Super-Resolution_with_Superpixel_Token_Interaction_ICCV_2023_paper.html)：超像素 Token、区域内注意力和跨超像素交互，说明轻量 SR 中“聚合 Token 后交互”已有直接近邻。
- [CAMixerSR（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CAMixerSR_Only_Details_Need_More_Attention_CVPR_2024_paper.html)：学习内容路由，对复杂窗口使用注意力、简单区域使用动态卷积。它更接近稀疏计算分配，不等同于 N9 的软 Token 压缩，但“内容门控融合”表述需要避免重合。
- [SAFMN（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.html)：多尺度空间自适应调制与局部 channel mixer；它使 LMLT 式候选的多尺度调制新颖性进一步降低。
- [Pure-Pass（2025 arXiv）](https://arxiv.org/abs/2510.01997)：细粒度动态 Token-mixing 路由。当前检索到的是预印本，不能与正式同行评审证据等量表述，但写作时必须列为近邻。

### 2.2 N9 能成立的差异与不能成立的表述

可以检验的差异是：**LFMN 的共享浅层先验 `Fs` 不直接产生输出 Token，而是与当前阶段 `Fm` 联合决定每个像素到每图像 Token 的软归属；相同归属矩阵完成聚合和解聚合；八阶段各自重新路由；无外部字典、无 EMA 原型、无硬排序，训练与推理完全一致。**

不能直接声称：

1. “首次图像自适应 Token 聚合”——TokenLearner、Slot Attention、SPIN、ATD、CATANet 已覆盖相邻思想。
2. “首次可微 Token 路由用于 SR”——需要更完整专利/论文检索，当前证据不足。
3. “解决了 TAB 的真实瓶颈”——必须先有性能、路由诊断和等容量对照。
4. “先验条件带来提升”——需要 `Fm-only` 对照；仅比较原 TAB 与 `Fs+Fm` 无法隔离 `Fs` 的贡献。

若实验成立，较稳妥的论文贡献表述应是：**在层级调制 SR 骨干中，对固定原型硬聚类模块做先验条件、每阶段、端到端软聚合的受控替换，并证明其精度—效率与路由行为。**这仍是增量型架构贡献，强度取决于消融和跨数据集结果。

## 3. 三个大改动候选的决策矩阵

评分为结构先验下的评审判断，不是实测。`高`不表示承诺 PSNR 增益。

| 维度 | A：PCSTR 软 Token 路由 | B：SRFormer 式 PSA 大窗口 | C：LMLT 式低到高多尺度 |
| --- | --- | --- | --- |
| 预期 PSNR 潜力 | 中高；保留全图内容相似交互，但有低秩平滑风险 | 中；大窗口有文献支持，但与 LRSA 重叠 | 中高；全局/局部多尺度兼顾，但 48 通道下每头容量有限 |
| Urban100 适配 | 高（重复结构、远距相似） | 高（大窗口建筑线条） | 高（跨尺度结构） |
| Manga109 适配 | 高，但软聚合可能混淆相似线条 | 中高 | 高 |
| 参数量 | 约 733.3K（首版估算，低于 759.6K） | 约 730–760K，依窗口投影实现 | 约 735–770K，依下采样/融合实现 |
| FLOPs | 预计不高于基线；路由矩阵是 `O(NKC)` | 大窗口为 `O(Nw²C)`，窗口增大后敏感 | 通常较低；低分辨率头节省明显 |
| 训练显存 | 中；需保存 `A∈R^(B×N×K)` | 高；大窗口注意力图占用明显 | 低到中 |
| 推理延迟 | 中；矩阵乘法规则，但小矩阵/重排可能不饱和 | 中；SDPA 友好，窗口重排有成本 | 低到中；池化、逐头串联和融合影响实测 |
| 实现风险 | 中高；Token 坍缩、归一化、软解聚合细节敏感 | 中；已有官方实现范式 | 中高；多尺度尺寸、对齐、串联依赖较多 |
| 训练稳定性 | 中；需监控质量与熵，但不首加辅助损失 | 高 | 中高 |
| 项目内针对性 | 高；直接替换 TAB 的特有机制 | 低；与后续 LRSA 都是窗口注意力 | 中；针对跨窗口，但非 TAB 特有 |
| 创新性 | 中；差异依赖 `Fs+Fm` 条件与阶段设计 | 低 | 低到中 |
| 与论文重合风险 | 中高：TokenLearner/ATD/CATANet/SPIN | 高：SRFormer 直接近邻 | 高：LMLT/SAFMN 直接近邻 |
| 首轮推荐 | **是** | 否，作为强工程基线 | 否，作为第二候选 |

B 的主要否决点是：LFMN 已在 TAB 后使用 `patch_size=[16,20,24,28]×2` 的 LRSA；再用 PSA 替换 TAB，会把“全图内容相似交互 + 局部窗口交互”变为“两次窗口交互”，难以解释为何针对当前结构缺陷。C 比 B 更互补，但若不引入 LFMN 特有条件，论文故事几乎就是把 LMLT 模块移植到 LFMN。

## 4. PCSTR 首版的严格定义

### 4.1 张量与公式

令批量为 `B`，LR 空间为 `H×W`，`N=HW`，主干通道 `C=48`，浅层先验通道 `Cs=32`，路由隐维 `r=16`，Token 数 `K=32`。将 `Fm`、`Fs` 展平为：

```text
F ∈ R^(B×N×48)
S ∈ R^(B×N×32)
```

保留原 TAB 对 `F` 的 LayerNorm，并为 `S` 加独立 32 通道 LayerNorm：

```text
Z = GELU(LN(F) Wx + LN_s(S) Ws)              ∈ R^(B×N×16)
A = Softmax_K(Z Wa / τ)                       ∈ R^(B×N×32)
m = Σ_n A[:, n, :]                            ∈ R^(B×32)
Vpix = LN(F) Wp                               ∈ R^(B×N×48)
T0 = (A^T Vpix) / (m[..., None] + ε)          ∈ R^(B×32×48)
```

其中 `Softmax_K` 沿 Token 维归一化，因此每个像素对 32 个 Token 的权重和为 1；`m` 再修正不同 Token 接收的总质量。原初稿中的 `T = A V(F)` 在当前形状下不成立，必须是 `A^T Vpix`。

Token 内使用 4 头注意力：

```text
Q = T0 Wq ∈ R^(B×32×24) → R^(B×4×32×6)
Kt = T0 Wk ∈ R^(B×32×24) → R^(B×4×32×6)
Vt = T0 Wv ∈ R^(B×32×48) → R^(B×4×32×12)
P = Softmax_32(Q Kt^T / sqrt(6))              ∈ R^(B×4×32×32)
T1 = Concat_heads(P Vt) Wo                    ∈ R^(B×32×48)
Y = A T1                                      ∈ R^(B×N×48)
```

广播恢复必须是 `A T1`，不是原初稿的 `A^T T_hat`。`Wo` 在线性条件下放在 Token 空间再广播，可避免对 `N` 个像素做一次额外 48×48 投影。

完整块保持原 TAB 的两段残差骨架：

```text
F1 = F_residual + Y
Fout = F1 + ConvFFN(LN(F1))
```

`τ` 首版固定为 `1.0`，`ε=1e-6`（混合精度实现时质量累加与除法用 FP32）。不首加可学习温度、Gumbel、top-k、熵损失或负载均衡损失；先记录 Token 质量分布、熵和有效 Token 数，只有实测坍缩才单变量处理。

### 4.2 Q/K/V 的明确选择

- 4 个头。
- Q/K 总维度固定 24，即每头 6。
- V 总维度固定 48，即每头 12。
- 像素聚合投影 `Wp:48→48`，不压缩 Value。

理由是 SR 输出对细节通道敏感，Value 压缩比 Q/K 压缩更可能形成不可逆信息瓶颈；而 K=32 时 Token 注意力本身很小，Q/K 从基线 36 降到 24 主要用于减少参数并形成明确首版。`d_qk=36` 只作为失败诊断对照，不能与首版同时扫参。

### 4.3 Token 数与阶段策略

首版八阶段全部使用 `K=32`，不沿用原 `[16,32,64,128]×2`。原调度与 IASA 的 `group_size=[256,128,64,32]×2` 联动，近似保持 `K×group_size=4096`；PCSTR 已删除组内像素注意力，因此直接继承该调度没有理论依据。固定 K 还能避免把阶段调度收益混入“软路由是否有效”的首个因果问题。

若 K=32 首版通过 Go 门槛，再只做 `K∈{16,32,64}` 三点消融。阶段可变 K 仅在固定 K 已显示清晰容量不足（后期 Token 质量分布饱和且 K=64 显著更好）时考虑。TokenLearner 的 8–16 Token 证据来自识别任务，不能直接作为像素级 SR 的 K 依据。

### 4.4 Token 从哪里生成、是否共享

首版选择：

- 每张图像动态生成；
- 分配由 `Fs+Fm` 联合生成；
- Value 只来自 `Fm`；
- 八阶段不共享 Token、不递推 Token 状态；
- 各阶段路由器参数也不共享。

`Fs` 提供稳定结构锚，`Fm` 提供当前阶段语义；让 Value 只来自 `Fm` 可避免浅层先验直接混入重建内容，使 `Fs` 的角色限制为路由条件。跨阶段共享/递推会同时引入记忆、状态对齐和误差累积，且此前先验演化/阶段记忆路线未获益，因此首版禁止。

首版通过后，关键消融顺序是 `Fm-only → Fs-only → Fs+Fm`。其中 `Fs-only` 预计难以反映阶段变化，但必须由实验而非直觉判定。

### 4.5 是否需要局部 DWC 与内容门控

首版不加入原初稿的 `L=DWC3×3(Fm)` 和 `G=Sigmoid(Wg([Fm,Fs]))`。原因不是局部分支必然无效，而是：

1. 原 TAB 的 `ConvFFN` 已含 96 通道 5×5 DWC，并在 PCSTR 首版保留；
2. PCSTR 后仍有 LRSA、LRSA 内第二个 5×5 DWC、3×3 `mid_conv` 和 ESA；
3. N6 已显示额外方向局部分支真实生效但与现有路径基本冗余；
4. 加入 DWC+门控后，无法区分收益来自软路由还是局部容量/动态门控。

若路由首版通过而高频分层明显退化，再测试 `PCSTR + DWC`；必须同时有等参数普通 mixer 对照。门控 `G` 至少推迟到 DWC 本身有效之后。

## 5. 参数量与计算预算

### 5.1 参数符号推导

忽略偏置时的主项，单个 PCSTR 路由器为：

```text
P_route = C·r + Cs·r + r·K + K
        + C²                         # Wp
        + 2C·d                       # Wq, Wk
        + 2C²                        # Wv, Wo
```

加上 `LN_s` 的 `2Cs`，以及保留的原 TAB 外层 `LN` 和 `ConvFFN`，按当前 PyTorch 层的偏置精确计数：

```text
P_block = 22,608 + 17K
```

当 `K=32`：

```text
P_block = 23,152
8×PCSTR = 185,216
8×原TAB = 211,584
整网估算 = 759,627 - 211,584 + 185,216 = 733,259
```

因此首版预计减少 26,368 参数，约为基线的 `-3.47%`，低于 800K 目标。该数字是依据冻结公式的静态预算，不是代码实测；实现后必须用模型实例化重新核对。

### 5.2 MACs/FLOPs 粗算

单块路由主项（`N=HW`）为：

```text
MAC_route = N[(C+Cs)r + rK + C² + 2KC]
          + K[2Cd + 2C²]
          + K²(d+C)
```

保留 ConvFFN 的主项：

```text
MAC_ffn = N(2CM + 25M),  M=96
```

对训练裁块的 LR `64×64`，`N=4096`、`C=48`、`Cs=32`、`r=16`、`K=32`、`d=24`：

- 路由约 29.66M MAC/阶段；
- 保留 ConvFFN 约 47.58M MAC/阶段；
- 合计约 77.23M MAC/阶段，八阶段约 0.618G MAC（若按乘加记 2 FLOPs，约 1.236G FLOPs）。

这不是整网 FLOPs，也不能直接与论文 57.2G 混用，因为论文计数工具、输入尺寸和 MAC/FLOP 定义需一致。现阶段只能判断该替换本身处于预算内；实现后必须对 baseline/candidate 使用同一 profiler、同一 `1×3×64×64` 输入，并在同一 GPU 测峰值显存和端到端延迟。禁止用 Table III 的 `6.91G` 差额直接相减得到新模型 FLOPs。

## 6. 公平对照与可归因设计

至少需要以下四组，首轮训练设置完全相同：

| 组别 | 目的 | 预计参数 |
| --- | --- | ---: |
| B0 原 LFMN | 真实基线 | 759,627 |
| A0 PCSTR，K=32，M=96 | 检验推荐机制；参数更少 | 约 733,259 |
| A1 PCSTR，K=32，M=123 | 将保留 ConvFFN 加宽到近等参数 | 约 759,827 |
| C0 普通 mixer 替换 | 与 A0 等参数/近 FLOPs，但无 Token 聚合 | 约 733K |

`M=123` 相比 96 每块增加约 `27×123=3,321` 参数，八块增加 26,568，使总量与 baseline 仅差约 200。A0 若已胜出，说明不是靠增加容量；A1 用于判断恢复同等容量是否进一步有效。C0 应采用规则的点卷积—DWC—点卷积 mixer，并通过通道宽度匹配 A0 的参数与 MACs；不能用明显更弱的 identity 充当对照。

若后续加入 `Fs` 条件，必须有 `Fm-only`；若加入 DWC/门控，必须有同参数非门控局部分支。机制、参数容量和训练策略结果分别报告。

## 7. 不误杀慢收敛结构的筛选协议

### 7.1 训练与验证固定项

- DIV2K train 0001–0800，×4 bicubic；验证固定 0801–0900 全部 100 图。
- 从零训练为主，所有组使用相同初始化规则、seed、数据顺序、增强、HR patch 256、batch 4、L1、Adam、初始学习率 `2e-4`、`eta_min=1e-6`、Cosine `T_max=150`。
- 结构性替换不以“加载基线除 TAB 外权重的短微调”作为主结论，因为新模块随机初始化会使比较不对称。可额外做 warm-start 工程筛选，但需单列。
- epoch 5/10/15/20 保存并评测；为计算末 5 轮均值，epoch 16–20 每轮都评测完整 100 图。评测统一为量化后 Y 通道、裁去 4 像素、单次推理。
- 每图保存 baseline/candidate 的配对 PSNR/SSIM；禁止只报告最佳单点。

### 7.2 曲线与分层统计

每个检查点报告：

1. 100 图平均 PSNR/SSIM 与逐图差值；
2. 逐图差值的 median、正收益图像比例、bootstrap 95% CI；
3. epoch 16–20 的均值，以及对这 5 点做线性拟合的斜率；同时报告 candidate-baseline 的斜率差；
4. 以 HR 图像固定规则计算高频能量（例如灰度 Laplacian 绝对响应均值），在训练前一次性按三分位划分低/中/高纹理组；报告每层的配对差值，不根据结果移动阈值；
5. Token 诊断：每阶段 `m/K` 分布、像素分配熵、有效 Token 数、最大/最小质量比、A 在图像增强前后的稳定性；这些是机制诊断，不作为额外挑选指标。

### 7.3 延长到 40 epoch 的客观条件

在 epoch 20 只允许三种决定：

- **直接停止**：末 5 轮 PSNR 差值 `≤0` 且斜率差 `≤0`；或 SSIM 持续下降超过 `0.0001`；或出现 Token 坍缩/数值不稳定；或参数、FLOPs、显存、延迟超预算。
- **直接延长**：末 5 轮差值 `≥+0.01 dB`，逐图中位数为正，且 epoch 16–20 没有明显向下发散。
- **灰区延长一次**：末 5 轮差值位于 `(0,+0.01)`，但差值曲线斜率至少 `+0.001 dB/epoch`，并且 5 个末轮点中至少 4 个比各自 baseline 更好。只延长到 40，不允许反复以“可能慢收敛”为由追加预算。

epoch 40 的 **Go** 条件：末 5 轮平均相对同条件 baseline `≥+0.02 dB`，逐图 bootstrap 95% CI 下界 `>0`，至少 60% 图像为正，SSIM 不低于 baseline 超过 `0.0001`，且高纹理组不是负收益；参数 `<800K`、同口径 FLOPs `<60G`，端到端延迟和峰值显存不恶化超过预先登记的 10%。

epoch 40 任一核心精度条件不满足即 **No-Go**，不进入 benchmark，不用单个最佳 epoch、单个纹理组或仅 Urban100 结果救回路线。通过后再做至少 3 seeds 的完整训练确认；Go 门槛是项目资源决策规则，不等价于统计学或论文显著性保证。

## 8. 一次性 benchmark 计划

只有结构、K、Q/K/V、损失和训练时长在 DIV2K validation 上全部冻结，且多 seed 确认后，才对同一最终 checkpoint 一次性评测：Set5、Set14、B100、Urban100、Manga109。

- 五个集合全部报告 PSNR/SSIM、单次推理；若需要 ×8 self-ensemble，另表报告且所有模型一致。
- 不根据 Urban100/Manga109 结果回改 K、温度、门控或 checkpoint；任何回改都视为新一轮研究，旧 benchmark 不再作为无偏最终测试。
- 预注册主要终点为 Urban100/Manga109 的平均 PSNR，Set5/Set14/B100 用于检查通用性与退化；仍需完整披露全部结果。
- 延迟只在同一硬件、软件、输入尺寸、warm-up 和重复次数下比较，报告 median/P90；FLOPs、参数、峰值显存与延迟同时给出。

## 9. N9 与 N8 的组合逻辑

概念上两者可能互补：N9 作用于 LR 域八阶段主干的非局部结构交互，N8 从浅层高频到 HR 重建提供局部细节短路径。但“一个全局、一个局部”只是结构叙事，不能证明互补。

只有 N8、N9 分别独立通过后，才做固定预算的 2×2 因子实验：

```text
baseline, N8, N9, N8+N9
interaction = (N8+N9 - N9) - (N8 - baseline)
```

- 两者独立为正、组合近似相加：可以称正交贡献，但仍是模块组合，不应宣称协同。
- interaction 稳定为正，且 N9 改善低/中频结构、N8 改善高频纹理的逐图/频段证据一致：才有“互补协同”的依据。
- 单模块无效、组合有效：需要额外因果实验，不能用组合结果回填单模块合理性。
- 两者都增加 `Fs` 依赖并共同偏向高纹理图：审稿人很可能视为 A+B 堆叠；需以频段误差、路由图和 2×2 交互消融回应。

## 10. 最终冻结建议

### 推荐结构

选择 A：PCSTR。保留 TAB 外层 LayerNorm、两段残差和原 5×5-DWC ConvFFN，只替换 CATANet 的 EMA centers、硬分组、IASA、IRCA。

### 首版必须保留

- `Fs+Fm` 联合产生 A，Value 只来自 `Fm`；
- 每图像、每阶段重新生成 Token；
- `K=32` 固定八阶段；
- 4 heads，Q/K=24，V=48；
- 质量归一化 `m` 与 FP32 除法；
- 相同 A 聚合与广播；
- 原 TAB ConvFFN、后续 LRSA/ESA 不改；
- Token 熵/质量/有效数诊断；
- 训练和评测同一路径。

### 首版禁止加入

- 局部 DWC 分支及内容门控 G；
- 跨阶段 Token 共享、递推或记忆；
- 可学习温度、Gumbel、top-k、硬化、EMA dictionary；
- 熵/均衡/边缘/频率/蒸馏辅助损失；
- 动态 K、阶段不同 K；
- 与 N8 同时训练；
- 为候选单独调学习率或选择有利 checkpoint。

### 最小可证伪版本

`B0 原 LFMN` 对 `A0 PCSTR(K32,d24,V48)`，从零训练 20 epoch，同步完整 DIV2K 100 图评测，并保留 `C0` 等参数普通 mixer。若 A0 相对 B0/C0 没有稳定信号，先用已有 checkpoint 检查 Token 坍缩和 `Fs` 真实依赖，不立即增加新机制。

### Go / No-Go

- **Go 到 40 epoch**：满足第 7.3 节的 `+0.01 dB` 或严格灰区斜率条件。
- **Go 到多 seed / benchmark**：40 epoch 末 5 轮 `≥+0.02 dB`、95% CI 下界为正、SSIM与效率合格，并优于等参数普通 mixer。
- **No-Go**：20 epoch 明确非正且无追赶斜率；40 epoch 仍低于门槛或 CI 跨 0；收益仅来自 M=123 容量版而 A0 不优于 C0；高频组系统退化；路由坍缩且一次单变量修复仍无效；或实际延迟/显存破坏轻量目标。

当前结论只批准按上述最小版本进行正式对照训练；本文件本身不批准把 N9 写成有效改进，也不批准在单模块通过前组合 N8。

## 11. 实现与结构验证记录（2026-09-20）

A0、A1、C0 已分别实现为 `LFMNPCSTR`、`LFMNPCSTRWide` 和 `LFMNMixControl`。实现没有修改原 `lfmn.py` 的基线定义。结构检查得到：

- baseline/A0/A1/C0 参数量分别为 759,627 / 733,259 / 759,827 / 733,867；
- A0 与 C0 在 LR 64×64 下的路由/普通 mixer 解析 MAC 分别为 29,655,040 / 29,183,744，差 1.589%；
- A0 的分配张量为 `B×N×32`，沿 K 归一化，且分别扰动 `Fm` 或 `Fs` 都会改变 A；
- 随机初始化时归一化像素分配熵均值约 0.9837，有效 Token 数约 31.37/32，Token 质量最大最小比约 2.30，未见初始坍缩；
- A0/C0 全部参数获得有限非零梯度，A0 的训练/评测数据流一致，CUDA FP16 autocast 前向有限，checkpoint 严格重载输出一致；
- 三个候选都已完成真实 DIV2K 单 batch 训练、0801 单图验证、逐图指标保存和 checkpoint 保存。该运行仅验证端到端链路，不构成精度证据。

本机 RTX 4060 Laptop、PyTorch 2.9.1+cu128、单张 LR 64×64、10 次预热、50 次交替顺序计时中，baseline/A0/A1/C0 的中位延迟为 41.466/31.238/30.387/27.897 ms，P90 为 43.162/32.255/32.115/29.069 ms；峰值 allocated memory 为 214.7/213.8/213.9/213.8 MiB。该结果只说明当前本地实现没有显著效率异常，正式效率仍需在训练服务器按同一脚本复测；不能从随机权重延迟推出最终精度或论文优势。

正式入口为 `repro/run_n9_scratch_server.sh`：依次从零训练 B0、A0、A1、C0，保存每轮100图逐图指标，并用 `repro/summarize_n9_scratch.py` 输出 epoch 5/10/15/20、末5轮、斜率、bootstrap 95% CI、正收益比例与固定高频三分层结果。当前尚未启动该正式训练。
