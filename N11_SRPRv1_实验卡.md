# N11 / SRPRv1 实验卡

## 1. 身份信息

```text
候选编号：N11
候选名称：SRPRv1 (State-Routed Proximal Reconstruction v1)
候选类型：结构/重建框架
当前状态：IMPLEMENTED（本地验证后进入 VERIFIED）
分支：codex/n11-srprv1
共同基线分支/提交：feature/prior-update-screen / b198de7
候选实现提交：待提交
负责人/AI：Codex
日期：2026-09-21
```

## 2. 一句话核心假设

由于原 LFMN 的阶段主要进行串联特征变换而没有显式维护由 LR 观测残差驱动的跨阶段重建状态，阶段可能重复修复或无法识别仍缺失的结构；SRPRv1 将观测残差、成对反投影、持久状态和学习型近端更新统一到每个阶段，预期改善复杂纹理和长边缘重建。若假设失败，状态会接近零/常数、门控饱和、去掉残差或持久状态后性能不降，或观测残差下降而 PSNR/SSIM 不升。

## 3. 独立性边界

SRPRv1 不更新 `fs`，不修改 SFML 的 `beta/gamma`，不使用 P01 反馈、RDSM 需求监督、N3/PriorProxy 先验演化、N4 阶段差分记忆、N8 高频直达头、N9 Token 路由/记忆或 RGCRD 教师蒸馏。唯一新增因果链是当前 HR 估计经过固定 `D` 得到观测残差，再经配对 `D^T` 驱动持久 HR 状态、近端特征更新和 HR 更新。

## 4. 机制与形状

```text
x_hat_0 = original LFMN decoder output x_base
r_i = y - D(x_hat_i)
 b_i = D^T(r_i)
h_tilde_i = Phi(lift(X_i), b_i, h_i)
g_i = sigmoid(G(lift(X_i), b_i, h_i))
h_(i+1) = h_i + g_i * (h_tilde_i - h_i)
z_(i+1) = z_i + P(lift(X_i), h_(i+1), b_i)
x_hat_(i+1) = x_hat_i + R(h_(i+1), b_i)
```

实际第一版在 HR 分辨率维护 `feature_hr` 与 `h`，8 个阶段共享结构但各阶段独立参数：LR feature `B×48×h×w`；HR feature/state `B×48×4h×4w`；HR estimate `B×3×4h×4w`；LR residual `B×3×h×w`；backprojection `B×3×4h×4w`。

`D` 为固定 4×4 average-pool analysis，`D^T` 为其 exact transpose：nearest expansion divided by 16。它是 bicubic 数据协议下的固定观测一致性代理，不能称为严格 bicubic adjoint；伴随误差必须由检查脚本报告。

## 5. 协议

SRPRv1 从零训练 20 epoch。B0 不重新训练，直接复用 N9 已完成的同协议从零训练 LFMN B0；具体路径由服务器运行时通过 `B0_REFERENCE` 指定，并由脚本校验配置。当前本地仓库只确认了 N9 的从零训练脚本和协议，尚未找到正式 20 epoch B0 输出目录；因此服务器运行前必须提供该实际路径。预期协议为 DIV2K `1-800/801-900`、×4、HR patch 256、batch 4、seed 1、增强开启、L1、Adam、lr `2e-4`、Cosine `T_max=150`、`eta_min=1e-6`。若 B0 文件缺失、epoch 不完整或协议字段不一致，脚本停止，不自动重训 B0。

## 6. 指标和闸门

必须记录参数、新增参数、MAC/FLOPs（1 MAC=1 multiply-accumulate，FLOPs=2×MAC）、峰值 allocated/reserved CUDA 显存、warmup+同步后的 median/P95 延迟、每 epoch PSNR/SSIM、逐图指标，以及 residual/backprojection/state/update/gate 诊断。未测字段保持待测。

G0/G1/G2：尺寸、有限性、伴随、非零梯度、真实输入依赖、跨阶段依赖、保存重载。20 epoch 只作为筛选，不自动宣称优于 B0。若未来要做严格架构结论，必须另跑同协议从零 B0。

立即 NO-GO 信号：状态/门控失活或饱和；去掉状态/残差后不降；观测残差下降而 PSNR/SSIM 不升；效率明显超预算；或只有单个最佳 epoch 的孤立收益。

## 7. 服务器入口

```bash
bash repro/run_n11_screen_server.sh n11/srprv1_20e_seed1
```

可通过 `DATA_ROOT=/path/to/datasets PYTHON_BIN=python GPU=0` 覆盖路径。脚本先运行 `check_n11.py`，再只训练 SRPRv1，输出 `run_manifest.json`、`console.log`、checkpoint、完整精度曲线和逐图指标，并调用 `summarize_n11.py`。

## 8. 未验证字段

当前参数、FLOPs、显存、延迟、PSNR、SSIM、状态诊断和服务器运行状态均为待测，不预填结果。
