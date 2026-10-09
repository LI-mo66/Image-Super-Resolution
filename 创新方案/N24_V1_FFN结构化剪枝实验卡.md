# N24：V1 ConvFFN 局部结构化剪枝实验卡

> 本卡在编码前建立。N24 是效率候选，不预设 PSNR 提升；未经匹配恢复训练与固定端点评测，不得写成已证实的精度或速度改进。

## 1. 身份信息

```text
候选编号（唯一）：N24
候选名称：V1 ConvFFN 局部结构化剪枝
候选类型（结构/训练框架/损失/数据/推理/效率/组合）：结构 + 效率
当前状态（PROPOSED/AUDITED/APPROVED/IMPLEMENTED/VERIFIED/SCREENING/DECIDED/PROMOTED/ARCHIVED）：ARCHIVED
分支：codex/n24-v1-ffn-pruning
共同基线分支/提交：codex/n16-srpr-c1-interaction / 982463c8739ae84e10f7b2d41c9a42aa5578fae5
候选实现提交：见本实验卡所在提交
负责人/AI：Codex
日期：2026-10-09
```

## 2. 一句话核心假设

```text
由于 V1 的 16 个 ConvFFN 使用统一 96 维隐藏宽度、合计占 22.54% 参数，而其外部接口始终为 48 维，当前基线可能在局部 FFN 隐藏通道上存在可恢复冗余；本候选通过保持全部 48 维主干、SRPR 状态和解码接口不变，仅真实删除低重要性的 FFN 隐藏通道，因此预期在实测推理成本下降时，经匹配 C1 恢复后保持 V1 的固定端点 PSNR/SSIM。
```

## 3. 源码事实与问题证据

- 相关源码位置：`LFMN/model/lfmn.py` 的 `ConvFFN`、`TAB`、`LRSA`；`LFMN/model/lfmnsrprv2.py` 的 `Net`。
- 原始数据流：每阶段 `SFML -> TAB(含 ConvFFN) -> LRSA(含 ConvFFN) -> mid_conv/ESA -> SRPR observation/state/writeback/q`，最后经两级 PixelShuffle 解码。
- 可测缺陷：V1 固定使用 16 个 `48 -> 96 -> 48` ConvFFN；是否所有阶段都需要完整 96 维尚未验证。V1 快路径在 RTX 4090、FP32 下相对 C1 的稳态延迟仍约为 LR64 `+23.46%`、LR128 `+9.98%`。
- 已有实验支持：N16 在单 seed、Cosine T_max=150 协议下相对 C1 的 epoch150 增量为 `+0.046227 dB`，固定端点五 benchmark 在复杂纹理数据集上有正向证据；这些结果要求剪枝优先保护 Urban100/Manga109 收益。
- 已有实验反证：N17--N20 表明代理评分改善、教师优势、存在梯度或短适配均不自动转为稳定 PSNR；因此 N24 不以重要性分数或本地可学习性代替端到端证据。
- 参数审计：V1 共 841,563 参数；TAB FFN 94,848、LRSA FFN 94,848，合计 189,696（22.54%）。统一 `96 -> 88/80` 理论上分别减少 15,744/31,488 参数，但参数减少不等于延迟下降。
- 哪些判断仍是未知：FFN 的实际 CUDA 时间占比；96->88/80 的真实延迟变化；幅值评分与任务敏感度评分谁更好；匹配恢复后的 PSNR/SSIM；多 seed 稳定性；完整 MAC/FLOPs。

## 4. 机制与公式

- 输入：每个 ConvFFN 接收 `B x N x 48` token，`N=H*W`，dtype/device 与 V1 保持一致。
- 状态/表示：只把隐藏表示宽度从 `h=96` 改为 `h'<96`；不修改持久 `state: Bx48xHxW`、`q: Bx3xHxW`、EMA prototype、排序标签或 attention head。
- 更新规则：`z=GELU(fc1(x))`，`z'=z+DWConv5x5(z)`，`y=fc2(z')`；对选定索引集合 `I` 同步保留 `fc1` 输出行/bias、DWConv 通道/bias与 groups、`fc2` 输入列。
- 初始评分：每个 FFN 内对三类依赖权重分别按通道归一化后求和，形成确定性 magnitude score；任务敏感度评分登记为 M1 通过工程与速度门后才比较的替代选择规则。
- 输出：形状仍为 `B x N x 48`，外部残差、归一化、空间顺序和 V1 forward 不变。
- 损失：服务器恢复阶段保持 C1：`L1(SR,HR)+0.1*L1(SR,teacher.detach())`；本地工程验证只使用有限标量损失检查梯度和更新，不产生 PSNR 结论。
- 训练态与推理态差异：教师仅训练期存在；候选推理为真实窄 Linear/DWConv/Linear，不保留 96 维 mask 算子。
- 为什么这是统一机制而不是模块并列：同一个隐藏通道依赖规则作用于 V1 全部 16 个同构 ConvFFN，不增加旁路、门控、loss 或推理模块。

## 5. 与共同基线的差异

```text
共同基线：LFMNSRPRV2，16 个 ConvFFN 隐藏宽度均为 96。
候选：LFMNN24FFNPRUNE，首轮隐藏宽度 88；80 仅作速度不足时的预注册成本边界。
保持不变的部分：48维主干、SFML、TAB/LRSA attention与路由、ESA、全部SRPR、stage probe、state/q、观测算子、PixelShuffle、C1与评测口径。
被替换的部分：16 个 ConvFFN 的 fc1/DWConv/fc2 由96维真实缩为选定宽度。
训练期新增：无永久模块；仅权重迁移/评分工具。
推理期新增：无；实际减少隐藏通道。
```

## 6. 最近邻与新颖性风险

| 工作 | 相同点 | 实质差异 | 风险 |
|---|---|---|---|
| FMP, ICML 2024 | 面向轻量 SR 的通道/权重剪枝 | N24 不引入 hypernetwork 或联合非结构化稀疏，只缩 V1 局部 FFN | 工程价值可能高于论文新颖性 |
| Isomorphic Pruning, ECCV 2024 | 同构结构内排名，避免异构分数直接混排 | N24 首轮只在每个 ConvFFN 内排名，并保护 V1 持久状态接口 | 评分原则已有明确先例 |
| MDP, CVPR 2025 | 以真实延迟约束结构搜索 | N24 先用小型实测宽度表，不实现 MINLP 或多维联合剪枝 | 仅做统一宽度可能未达到最优延迟 |
| HFCP, TIP 2025 | 低层复原任务不能机械使用分类范数评分 | N24 的任务敏感度将直接使用 SR/C1 损失，不把高频能量直接等同于有用性 | HFCP 是去噪，不能外推其收益 |
| Torch-Pruning/DepGraph | 真实结构依赖删除 | N24 使用显式 V1 局部依赖映射，并以工具作交叉检查而非自动正确性保证 | 动态排序、einsum和状态循环可能不被自动图完整理解 |

拟议贡献必须避免的表述：

```text
不得声称剪枝必然提高PSNR、首次提出SR结构化剪枝、参数减少等于加速、单seed等于稳定，或把其它网络的压缩率外推到V1。
```

## 7. 复杂度预算

```text
基线参数：841,563
候选参数（h=88）：825,819（预计算，待代码实测）
参数变化：-15,744 / -1.87%（预计算）
候选参数（h=80边界）：810,075（预计算，待代码实测）
推理MAC/FLOPs变化：未知；不得在依赖安全计数前报告
训练额外计算：与V1相同教师成本；窄FFN理论略低，待实测
预计显存：未知
预计延迟：未知；目标设备和输入尺寸实测决定是否继续
尚未实测字段：算子占比、LR64/LR128/目标尺寸 median/P90、peak allocated/reserved、AMP、导出后延迟
```

## 8. 可证伪预测

若假设成立：

```text
h=88真实窄模型严格继承选中权重，所有接口/梯度/保存重载通过；目标设备至少一个主要尺寸有可复现延迟下降；匹配恢复后固定端点PSNR/SSIM不低于同预算未剪V1，且Urban100/Manga109优势未明显退化。
```

若假设失败：

```text
只有参数下降而主要尺寸不加速；或同预算恢复后PSNR持续为负；或复杂纹理数据集退化；或宽度依赖导致无法严格加载/保存/恢复，则停止当前版本，不靠追加模块或无限延长恢复掩盖。
```

机制退化信号：

```text
FFN输出/梯度非有限、所选通道未进入最终损失、新参数或迁移参数遗漏、TAB硬路由变化导致任务评分不稳定、最坏纹理图损失扩大、延迟方差/P90恶化。
```

## 9. 基线协议指纹

```yaml
baseline_id: V1_N16_epoch150_Cosine_seed1
source_commit: 982463c8739ae84e10f7b2d41c9a42aa5578fae5
baseline_model: LFMNSRPRV2
pretrained_checkpoint: N16-V1/model/model_150.pt (本地未跟踪副本)
checkpoint_sha256: 0c3594c88c725e4f0e195a2be61c61c15580225f748ffa96b5cb57db5bd5eee0
data_train: DIV2K
data_range_train: 1-800
data_range_validation: 801-900
scale: 4
patch_size_hr: 256
batch_size: 4
seed: 1
augmentation: repository default; exact random-stream replay required before screening
optimizer: Adam(beta=(0.9,0.999), eps=1e-8)
learning_rate: 2e-4 historical start; recovery LR to be preregistered before server run
scheduler: CosineAnnealingLR historical; recovery scheduler unknown and not yet authorized
scheduler_horizon_or_milestones: historical T_max=150
eta_min_or_gamma: historical eta_min=1e-6
epochs: historical 150; recovery budget unknown and not yet authorized
steps_per_epoch: unknown; must recover from source manifest/log before screening
loss: 1*L1 + 0.1*output KD from frozen SwinIR-M x4
rgb_range: 255
evaluation_datasets: DIV2K validation; fixed endpoint Set5/Set14/B100/Urban100/Manga109 only after screen gate
metric_code_commit: 982463c (must freeze exact screening commit later)
metric_protocol: repository PSNR/SSIM; x4; no self-ensemble; exact RGB/Y, quantization and shave fields must be copied from N16 manifest before screening
torch_cuda_environment: local engineering check PyTorch 2.9.1+cu128 / RTX 4060 Laptop; server screening environment unknown
result_directory: not created
```

## 10. 基线复用决定

```text
Baseline reuse: NO（历史曲线）；YES（仅作为两组共同初始化权重）
复用的baseline ID：V1_N16_epoch150_Cosine_seed1 checkpoint
逐字段比较结果：结构、数据、倍率、损失和评测目标计划保持；恢复LR、scheduler、epoch、steps尚未注册。
已知差异：剪枝后需要恢复训练；历史V1没有接受同一额外恢复预算。
为何差异不影响或为何必须重训：同一checkpoint可以作为确定性共同起点，但正式主比较必须给未剪V1相同数据流、更新数、优化器、LR和C1预算，不能直接拿剪枝恢复结果减历史epoch150曲线。
```

## 11. 最小实验矩阵

| 组别 | 模型 | 改动 | 回答的问题 | 何时运行 |
|---|---|---|---|---|
| B0 | 未剪 V1 | 从同一epoch150权重进行匹配恢复 | 额外训练本身带来多少变化 | 获得服务器训练授权后与M1配对运行 |
| M1 | N24-h88-magnitude | 16个FFN真实缩为88并继承每层幅值top-88通道 | 最保守局部剪枝能否降本且恢复到V1 | 第一阶段 |
| C1 | N24-h88-task | 同宽度，改用HR+C1任务敏感度选通道 | 收益是否来自任务评分而非普通幅值 | M1通过结构与速度门后 |
| M0 | N24-h80-magnitude | 更窄成本边界 | h88不够快时，损失/速度边界在哪里 | 仅h88主要尺寸延迟不足时 |

## 12. 预注册闸门

```text
Gate 0结构检查：841563基线参数；h88=825819；16个FFN均为48->88->48且DWConv groups=88；外部state/q/head/PixelShuffle不变；迁移键、索引和元素覆盖率有报告。
Gate 1梯度/数值检查：奇数且非方形输入、至少两种尺寸输出有限；最终标量损失连通全部窄FFN；梯度有限；optimizer无遗漏/重复；一步后参数真实更新。
Gate 2真实单batch：在真实DIV2K batch可得时检查C1前向、AMP反缩放/overflow、保存严格重载；本地无数据时不得以随机tensor冒充此门通过。
Gate 3短筛epoch：尚未授权；训练前补齐恢复LR/scheduler/epoch/steps并冻结命令。
短筛通过线：相对同预算B0，固定端点PSNR与末N轮均不低于预注册容差，SSIM不退化，逐图CI/胜率支持；阈值在读取N16波动和确定恢复预算后锁定。
灰区：精度近零差但CI跨0，或只有次要尺寸加速；最多补一个预注册seed或一个h80成本边界，不改结构叠模块。
立即停止线：主要尺寸无可复现加速；输出/梯度/迁移不正确；匹配恢复后终点与末段均为负；Urban100或Manga109出现超过预注册容差的退化。
允许延长到150/200 epoch的条件：短恢复相对匹配B0非负且真实延迟达到目标，协议完整，用户批准预算。
允许1000 epoch的条件：150/200轮、多seed和五benchmark均支持，且效率收益足以抵偿恢复成本；需另行授权。
多seed要求：声称稳定保持/提高PSNR前至少补多个训练seed；图片bootstrap不能替代训练seed。
五benchmark要求：只使用固定端点；不按benchmark选比例或checkpoint；分别报告每集、逐图和复杂纹理子集，不用混合均值隐藏退化。
```

## 13. 实际服务器协议

```text
未启动服务器训练。候选在本地效率 Gate 即触发停止线。
```

## 14. 结果

本地环境：RTX 4060 Laptop GPU，PyTorch 2.9.1+cu128，CUDA 12.8，batch1，FP32，V1
checkpoint SHA256 `0c3594c88c725e4f0e195a2be61c61c15580225f748ffa96b5cb57db5bd5eee0`。

结构与学习启动检查：

- h96控制模型参数841,563，与原V1同输入最大绝对输出差为0；
- h88参数825,819（-15,744/-1.87%），h80参数810,075（-31,488/-3.74%）；
- 16个FFN都是真实窄Linear/DWConv/Linear，迁移形状不匹配键80个且全部显式处理，目标参数覆盖率100%；
- h88在`1x3x31x37`输入上产生`1x3x124x148`有限输出，96个FFN参数张量均获得有限梯度；optimizer无遗漏/重复，一步更新非零；严格保存重载输出最大绝对差为0；
- h96在奇数非方形输入上与原V1逐元素一致；真实DIV2K batch、C1教师、AMP未检查，不能视为Gate 2通过。

本地轮换延迟使用10次预热、30次计时：

| 宽度/输入 | V1 median/P90 ms | N24 median/P90 ms | median变化 | P90变化 |
|---|---:|---:|---:|---:|
| h88 / LR64 | 216.833 / 234.488 | 222.705 / 233.922 | +2.71% | -0.24% |
| h88 / LR128 | 273.354 / 289.275 | 272.099 / 291.734 | -0.46% | +0.85% |
| h80 / LR64 | 218.769 / 235.332 | 223.143 / 245.921 | +2.00% | +4.50% |
| h80 / LR128 | 277.209 / 302.666 | 272.843 / 292.432 | -1.58% | -3.38% |

不同宽度各自与同轮V1比较。两档在LR64均变慢，LR128的下降不足2%且与尾延迟波动同量级，
没有达到主要尺寸约10%的预注册目标。没有运行恢复训练或PSNR评测，不能判断剪枝后的精度。

## 15. 决策与后续

```text
结论：NO-GO / ARCHIVED
是否超过B0：未知；未训练，禁止作PSNR结论
是否需要C1/M0：不触发；h88未通过速度门，h80成本边界也未通过
是否允许延长：否
失败归因：当前GPU和FP32 eager执行下，仅缩16个ConvFFN隐藏宽度没有改变整网主要延迟瓶颈；参数下降不等于kernel/整网延迟下降。该结果不否定其它硬件、编译后端、非均匀宽度或其它剪枝对象。
不可宣称的内容：不能宣称PSNR保持/提高、部署普遍无效、所有结构化剪枝失败，或h80/h88在其它设备一定不加速。
下一步：停止N24训练投入；若继续降本，应先做逐模块CUDA热点审计，再登记不同处理对象的新候选，不在N24上叠补丁。
结果记录提交：见本实验卡所在提交
```
