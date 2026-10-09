# F1 恒等增量 ESA 实验卡

## 1. 身份信息

```text
候选编号（唯一）：F1
候选名称：Identity-Preserved Delta ESA
候选类型：结构
当前状态：VERIFIED
分支：codex/f1-identity-delta-esa
共同基线分支/提交：本仓库原始LFMN；审查起点88bdc6a
候选实现提交：17461e8
负责人/AI：Codex
日期：2026-10-09
```

## 2. 一句话核心假设

由于原LFMN让ESA对上一阶段状态与当前增量之和整体门控，跨阶段持久状态可能被反复缩放；F1通过恒等保留旧状态、只对当前增量施加ESA并学习阶段残差尺度，检验这种状态更新是否能改善梯度传播、稳定性和x4重建精度。

## 3. 源码事实与问题证据

- 相关源码位置：`LFMN/model/lfmn.py`的`Net.forward()`。
- 原始数据流：`feat = ESA(prev + mid_conv(s))`。
- 可测缺陷：历史状态没有绕过ESA的严格恒等路径；所有阶段的旧状态均受新掩码影响。
- 已有实验支持：尚无F1训练结果；原LFMN论文消融只支持ESA整体有效，不回答门控位置是否最优。
- 已有实验反证：无。
- 未知：官方checkpoint上的掩码饱和、阶段更新比、梯度衰减是否达到有害程度；F1是否提高PSNR；学习尺度是否形成稳定阶段差异。

证据等级：源码事实为高；“全状态门控造成性能限制”为待验证机制假设；性能收益为未知。

## 4. 机制与公式

- 输入：上一阶段特征`F_(i-1)`和当前块生成的增量`Delta_i=mid_conv_i(s_i)`，形状均为`B x 48 x H x W`。
- 状态：每阶段一个可学习标量`alpha_i`，共8个，FP32初始化为1。
- 基线更新：`F_i = ESA_i(F_(i-1) + Delta_i)`。
- F1更新：`F_i = F_(i-1) + alpha_i * ESA_i(Delta_i)`。
- 输出、损失、训练态与推理态：保持原LFMN协议；没有额外损失或仅训练时分支。
- 统一性：同一更新规则作用于全部八阶段，只重新定义持久状态与阶段增量的写回关系，不并列堆叠新特征模块。

接口约束：`F_(i-1)`与`Delta_i`必须shape、dtype和device完全一致；不允许广播空间或通道维。`alpha_i`只按单标量广播，梯度由最终重建损失直接回传。ESA的mask、padding、插值与基线保持不变。

## 5. 与共同基线的差异

```text
共同基线：LFMN/model/lfmn.py::Net
候选：LFMN/model/lfmnf1.py::Net
保持不变：浅层特征、SFML、TAB、LRSA、ESA内部、重建头和全局残差
被替换：八阶段最后一行的状态更新公式
训练期新增：8个可学习标量alpha_i
推理期新增：每阶段一次标量乘法；ESA次数和卷积shape不变
```

## 6. 最近邻与新颖性风险

| 工作 | 相同点 | 实质差异 | 风险 |
|---|---|---|---|
| ResNet/残差缩放 | 恒等路径与可学习/固定缩放 | F1把ESA限定为阶段增量写回算子 | 基础算子本身不新 |
| SPAN | 低成本注意力与残差聚合 | SPAN的注意力表达和历史拼接位置不同 | 可能被评价为已有残差注意力变体 |
| EchoSR | 强调跨层信息保留与残差组 | F1不移植其完整融合块或上下文结构 | 叙事接近但机制较简单 |
| DFFN | 历史与频率信息协同 | F1不显式分频，也不复用其完整网络 | 只能作为问题动机，不能作为代码证据 |

不得表述为“首次提出残差缩放”“ESA必然过度抑制”或“已提高PSNR”。即使性能成立，也必须通过近邻检索与必要对照确认是否具有论文级新颖性。

## 7. 复杂度预算

```text
基线参数：759,627（本地源码实例化值）
候选参数：预计759,635
参数变化：+8（约0.0011%）
推理卷积/MAC变化：主卷积和ESA调用不变
新增逐元素运算：每阶段一次alpha标量乘法，共8*C*H*W
训练额外计算：8个标量的梯度与更新
预计显存：只增加8个参数及其优化器状态；激活数量同阶
预计延迟：接近基线，必须在统一设备实测
尚未实测：FLOPs/Multi-Adds口径、GPU延迟、峰值显存
```

## 8. 可证伪预测

若假设成立：阶段梯度传播更稳定；旧状态不再随ESA mask直接衰减；`alpha_i`形成非平凡阶段差异；同协议训练的x4 PSNR在预注册端点和末轮均值上稳定超过B0。

若假设失败：F1与B0持平或更差；尺度长期停留在初始化附近；ESA增量幅值不足或恒等累积导致特征范数增长、训练振荡或重建退化。

机制退化信号：非有限输出/梯度、阶段范数持续放大、`alpha`无梯度或异常增大、ESA mask极端饱和。特征相关性只作为表示诊断，不直接等同功能冗余。

## 9. 基线协议指纹

2026-10-09用户锁定以下开发协议：

```yaml
baseline_id: F1-B0-COS150-SCRATCH
source_commit: 待训练入口提交后登记
baseline_model: LFMN
candidate_model: LFMNF1
initialization: scratch
pretrained_checkpoint: null
data_train: DIV2K
data_range_train: 1-800（沿项目开发协议；启动前最终核对）
data_range_validation: 801-900（独立验证集；启动前最终核对）
scale: 4
seed: 1
patch_size_hr: 256
batch_size: 4
optimizer: Adam (betas=0.9,0.999; eps=1e-8; weight_decay=0)
learning_rate: 0.0002
scheduler: CosineAnnealingLR
scheduler_horizon: 150
epochs_planned: 150
mandatory_pause_epoch: 20
eta_min: 0.000001
self_ensemble: false
evaluation_datasets: [Set5, Set14, B100, Urban100, Manga109]
```

五个benchmark分别报告，不计算跨数据集宏平均。每个数据集仍按统一SR协议报告自身标准逐图均值，并保存逐图指标；Set5的五张图也全部列出。benchmark只评测固定端点或由独立验证集预先选定的checkpoint，不参与checkpoint选择。

## 10. 基线复用决定

```text
Baseline reuse：NO
复用的baseline ID：无
逐字段比较结果：B0与F1均按本卡协议从零训练
已知差异：只允许模型更新公式与新增8个标量不同
为何必须重训：用户明确要求两组从零训练；历史checkpoint和其它调度时间轴均不作为本轮B0
```

## 11. 最小实验矩阵

| 组别 | 模型 | 改动 | 回答的问题 | 何时运行 |
|---|---|---|---|---|
| B0 | 原LFMN | 无 | 同协议基线 | 获批训练后 |
| M1 | F1 | 恒等状态+缩放ESA增量 | 主假设是否成立 | B0配对运行 |
| C1 | 固定alpha F1 | `alpha=1`且不学习 | 收益是否来自可学习阶段尺度 | M1通过后 |
| M0 | 无ESA的增量残差 | `prev+alpha*Delta` | ESA是否仍有必要 | M1通过后 |

## 12. 预注册闸门

```text
Gate 0：参数、键覆盖、公式恒等旁路、shape、严格保存重载
Gate 1：有限输出/梯度、alpha与ESA梯度非零、优化器覆盖
Gate 2：真实单batch训练、验证、保存重载；同时通过强制日志测试
Gate 3：B0/F1同seed、同batch流、同Cosine T_max=150启动计划150e训练，在epoch20保存完整状态并强制暂停
epoch20报告：固定epoch20 checkpoint；五benchmark分别列出，不做跨数据集平均；Self-Ensemble OFF
继续条件：用户审阅epoch20训练/验证曲线、末5轮、五benchmark逐项结果和资源成本后明确批准
短筛/150e判定：沿用F系列研究审查报告预注册阈值
1000 epoch：必须用户单独批准
```

用户授权交付服务器运行包后，剩余运行参数按同协议开发默认值登记如上。独立入口为`repro/run_f1_screen_server.py`，实际训练入口为`repro/f1_train_entry.py`；启动时自动执行日志测试与Gate 2，不得绕过失败检查直接开始20e。参数和命令详见根目录`F1_服务器运行说明.md`。

## 12.1 实现验证记录

2026-10-09使用`E:\anaconda\envs\dl\python.exe`与RTX 4060 Laptop GPU运行：

```text
python -m py_compile LFMN/model/lfmnf1.py repro/check_f1_identity_delta_esa.py
python repro/check_f1_identity_delta_esa.py --device auto
```

结果：Baseline/F1参数量为759,627/759,635；官方x4 checkpoint SHA256严格匹配登记值，除新增`residual_scales`外无missing/unexpected keys，共享checkpoint元素覆盖率100%。零尺度严格返回旧状态，单位尺度严格符合登记公式；`1x3x31x35`输入得到`1x3x124x140`有限输出；八个尺度均获得有限非零梯度，ESA梯度非零；`--model LFMNF1`动态入口与候选checkpoint严格保存重载通过。

本轮未运行真实数据、训练、PSNR/SSIM、FLOPs、延迟或峰值显存测试，所以状态保持`IMPLEMENTED`，尚不进入`VERIFIED`。原报告建议先D0再实现；用户本轮明确授权先实现并上传，因此只调整工程顺序，不改变D0与Gate 2仍需完成的事实。

## 12.2 服务器包交付验证

用户随后要求交付服务器代码，现新增父进程日志管理器、独立训练入口、B0/F1配对运行器、资源审计和服务器说明。公共`main.py/option.py/trainer.py`正在存在其他未提交日志开发，本包不依赖这些改动；使用共同基线提交的Git archive源码加本次文件进行隔离验证。

Gate 2已通过：真实DIV2K 0001单batch（HR patch256、batch4）及0801验证，两组均成功保存模型/优化器/scheduler/随机流状态，并恢复到下一轮。完整训练/验证轨迹留在`experiment/f1_release_smoke/F1_pair_x4_seed1_20261009_142737_637491/`。两组共享初始state SHA256完全相同，恢复后的Cosine `T_max=150`和epoch连续性通过。实时stdout/stderr、独立目录防覆盖、受控异常Traceback、session追加及状态记录均通过。

另以F1烟雾权重完成Set5测试-only链路，逐图文件正常保存；参数/FLOPs计数范围、LR64/128延迟和显存统计链路在本机通过。以上不是20e性能证据，尚未执行服务器20e、正式五benchmark比较或D0机制诊断。状态升级为`VERIFIED`只表示工程门槛通过，进入`SCREENING`以服务器实际启动为准。
