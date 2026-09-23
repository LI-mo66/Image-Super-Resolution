# N13 / SRPRv3 实验卡

## 1. 身份信息

```text
候选编号（唯一）：N13
候选名称：SRPRv3 / Output-Consistent Progressive Reconstruction
候选类型：结构 / 重建框架 / 效率
当前状态：VERIFIED
分支：codex/n13-srprv3
共同基线提交：b198de7
候选实现提交：待实现
负责人/AI：Codex
日期：2026-09-23
```

## 2. 一句话核心假设

由于 N12 的阶段观测代理、48通道隐状态、3通道旁路与最终 LFMN decoder 并非同一重建对象，数据一致性反馈只能形成弱优化偏置；本候选用单一、可直接输出的渐进图像状态贯穿八阶段，并以共享轻量重建头和稀疏LR残差反馈更新该状态，因此预期以更低复杂度把反馈信号稳定转化为最终 PSNR。

## 3. 源码事实与问题证据

- 相关源码：共同基线 `LFMN/model/lfmn.py`；失败候选 N12 的 `LFMN/model/lfmnsrprv2.py` 只作为负结果证据，不作为实现基础。
- 原 LFMN：八阶段更新48通道特征，最后只解码一次；各阶段没有显式图像状态。
- N12：每阶段用 `bilinear(y+q)+PixelShuffle(stage_probe(feat))` 形成代理图像，但最终由另一套两级 decoder 输出，并同时维护 `feat/state/q`。
- N12 40轮证据：36/40轮领先且末5轮均值 +0.007066 dB，但 epoch40 −0.004364 dB，逐图胜率53%，95% CI [−0.012260,+0.002455]；服务器中位延迟约 +45.8%。
- 可测缺陷：阶段闭环对象与最终输出不一致；三种状态职责重叠；八套 proximal 与显式伴随计算成本高。
- 未知：把唯一输出状态与反馈对象统一后是否能保留N12早期正偏置并提高终点稳定性。

## 4. 机制与公式

保留 LFMN 的 stem、`fea_Net`、八组 SFML/TAB/LRSA/ESA；替换原两级重建 tail。定义四个状态更新阶段 `S={2,4,6,8}`（按1开始计数），HR RGB状态初始化为双线性基底：

```text
x_0 = Bilinear4(y)
f_0 = Stem(y)
```

每个 LFMN 阶段仍执行原特征更新。若 `i∈S`，共享轻量头在LR网格预测PixelShuffle打包增量：

```text
u_i = PixelShuffle4(R(f_i))
x_i = x_(i-1) + a_i * u_i
r_i = y - BicubicDown4(x_i)
```

在 `i=2,4,6`，残差由共享投影写回下一阶段特征：

```text
f_i^+ = f_i + b_i * P(r_i)
```

第8阶段更新后的 `x_8` 直接作为唯一SR输出。`R` 为 `1×1(48→32) + LeakyReLU + DW3×3 + LeakyReLU + 1×1(32→48)`，48个输出通道表示 `3×4²` 打包RGB增量；`P` 为 `3×3(3→12) + LeakyReLU + 1×1(12→48)`。`R/P` 跨更新阶段共享，`a_i/b_i` 为有界可学习标量。损失仍为原始 L1，训练和推理使用同一路径。

统一性：图像状态既接收每阶段重建增量、产生观测残差，又是最终输出；不存在独立代理头、额外隐状态或最终旁路。

## 5. 与共同基线的差异

```text
共同基线：八阶段LFMN特征映射 + 两级PixelShuffle重建tail + bilinear基底
候选：相同八阶段主干 + 四次共享轻量增量更新 + 三次共享LR残差反馈
保持不变：stem、fea_Net、SFML、TAB、LRSA、ESA、阶段数、训练损失和公共返回协议
被替换：原upconv1/upconv2/last_conv重建tail
训练期新增：无辅助损失
推理期新增：四次轻量共享头、四次bicubic down、三次LR反馈投影
推理期删除：原两级全卷积PixelShuffle tail
```

## 6. 最近邻与新颖性风险

| 工作 | 相同点 | 实质差异 | 风险 |
|---|---|---|---|
| DBPN, CVPR 2018 | 上下投影误差反馈 | DBPN使用成对可学习上下采样层与密集HR/LR特征；本候选在LFMN阶段间更新唯一RGB输出状态 | 误差反馈并不新，贡献不能写成首次反投影 |
| SRFBN, CVPR 2019 | 逐步SR与反馈 | SRFBN是展开时间的RNN隐状态；本候选固定八阶段、共享图像增量头、反馈对象是同一最终RGB状态 | “渐进反馈”已有充分先例 |
| USRNet, CVPR 2020 | 数据项与先验交替 | USRNet展开MAP子问题并处理核/噪声；本候选只针对固定bicubic轻量SISR且没有闭式数据子问题 | 不可宣称严格优化展开 |
| LFMN | 复用八阶段全局/局部特征主干 | 替换一次性tail为输出一致的稀疏渐进重建 | 最终新颖性需继续检索2024–2026近邻 |

拟议贡献只能称为“面向轻量LFMN主干的输出一致渐进重建候选”，在完成更全面检索和实验前不宣称首创或SOTA。

## 7. 复杂度预算

```text
基线参数：约759,627
候选目标：不高于基线（删除原tail后预计明显下降）
参数变化上限：+0%，优选负增长
Conv2d MAC目标：不高于基线
服务器median延迟目标：不高于基线+10%
训练额外计算：四次状态更新与观测降采样
显存目标：allocated peak不高于基线+10%
尚未实测：候选精确参数、MAC、显存、FP32/AMP延迟
```

## 8. 可证伪预测

若假设成立：四次状态更新均非零；前三次反馈能改变后续特征和最终输出；更新后的观测残差多数下降；相对B0的末5轮、逐图胜率和CI稳定为正，同时复杂度满足预算。

若失败：状态增量或反馈趋零；观测一致性改善与最终PSNR无关；共享轻量tail损害原LFMN重建能力；或终点/逐图指标仍围绕零波动。

机制退化信号：某一阶段贡献占据全部输出、`a_i/b_i`饱和、残差比持续大于1、关闭反馈不改变输出、共享头梯度为零。

## 9. 基线协议指纹

```yaml
baseline_id: N9 scratch B0 seed1
source_commit: b198de7
baseline_model: LFMN
pretrained_checkpoint: none
data_train: DIV2K
data_range_train: 1-800
data_range_validation: 801-900
scale: 4
patch_size_hr: 256
batch_size: 4
seed: 1
augmentation: random flip/rotation enabled
optimizer: Adam
learning_rate: 2e-4
scheduler: CosineAnnealingLR
scheduler_horizon_or_milestones: T_max=150
eta_min_or_gamma: 1e-6
epochs: 20 screen
steps_per_epoch: 1000 optimizer steps / 4000 augmented samples
loss: 1*L1
rgb_range: 255
evaluation_datasets: DIV2K 801-900
metric_protocol: project full-precision PSNR/SSIM, paired per-image metrics
result_directory: /root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/scratch/n9_pcstr_20e_seed1/baseline
```

## 10. 基线复用决定

```text
Baseline reuse: YES，前提是服务器逐字段配置与完整前20轮曲线核验通过
复用baseline ID：N9 scratch B0 seed1
已知差异：候选模型结构；训练和评测协议不变
决定：只运行M1；任何公共训练/指标代码变化都必须重新审查复用资格
```

## 11. 最小实验矩阵

| 组别 | 模型 | 改动 | 回答的问题 | 何时运行 |
|---|---|---|---|---|
| B0 | LFMN | 无 | 统一基线 | 合法复用 |
| M1 | LFMNSRPRV3 | 完整输出一致渐进重建 | 主假设是否成立 | 第一阶段 |
| C1 | 待定 | 等参数轻量tail、无反馈 | 是否只是替换tail | M1通过后 |
| M0 | 待定 | 状态更新保留、反馈关闭 | 反馈是否必要 | M1通过后 |

## 12. 预注册闸门

```text
Gate 0：独立导入、48通道/8阶段、四次状态更新、单tensor输出、保存重载、基线键名审计
Gate 1：FP32/AMP有限；共享重建头、反馈头、所有阶段标量梯度非零；反馈因果检查
Gate 2：真实DIV2K单batch训练/评测；状态与残差诊断有效；关闭反馈改变后续特征和输出
Gate 3：同协议从零20 epoch，只与登记B0比较
短筛通过线：末5轮PSNR差值>0；epoch20逐图bootstrap CI下界>0；胜率≥60%；SSIM不退化；复杂度全部合格
灰区：末5轮为正但CI跨0，或CI为正但延迟超过10%；只允许机制复盘，不直接延长
立即停止线：末5轮≤0、SSIM明确下降、状态/反馈失活、或延迟/显存明显超预算
允许40 epoch：20轮全部通过后，同协议配对确认
允许1000 epoch：40轮再次满足末5轮、CI、胜率、SSIM与效率闸门，并完成至少一个额外seed确认
多seed：正式长训前至少增加seed 2；论文主张至少3 seeds或说明资源限制
五benchmark：40轮及多seed通过后一次性固定权重评测
```

## 13. 实际服务器协议

待运行。

## 14. 结果

待运行，不预写收益。

## 15. 决策与后续

```text
结论：本地实现与验证通过，待服务器20轮短筛
不可宣称：性能提升、首创、严格优化展开、SOTA
下一步：服务器先运行check-only，再只训练M1 20轮
```

## 16. 本地实现与验证记录（2026-09-23）

独立模型为 `LFMN/model/lfmnsrprv3.py`。结构、基线键名迁移审计、随机输入与矩形输入、反馈ON/OFF因果、所有新增参数梯度、FP32/AMP有限性、内存严格保存重载均通过。真实DIV2K `0001` 的32×32 LR / 128×128 HR裁剪完成一次Adam训练步，L1有限，四次图像更新与前三次反馈均非零；公共Trainer又以真实DIV2K 0001/0801完成单batch训练、评测、`model_1.pt`和逐图指标落盘。烟雾PSNR/SSIM不是性能结果。

本机 RTX 4060 Laptop、PyTorch 2.9.1+cu128、FP32、LR 64×64、warmup10/repeats50、相同 profiler：

| 指标 | LFMN | SRPRv3 | 变化 |
|---|---:|---:|---:|
| 参数 | 759,627 | 596,495 | −163,132（−21.48%） |
| Conv2d MAC | 2.832816G | 1.115298G | −60.63% |
| median latency | 40.622 ms | 41.348 ms | +1.79% |
| P95 latency | 46.816 ms | 48.661 ms | +3.94% |
| allocated peak | 214.738 MiB | 219.505 MiB | +2.22% |

MAC只统计Conv2d，不包含bicubic插值、PixelShuffle、注意力和张量操作；延迟包含完整前向。当前满足预注册本地效率预算，服务器必须用同一脚本复测。
