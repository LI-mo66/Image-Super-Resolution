# N12 / SRPRv2 实验卡

## 身份

```text
候选编号：N12
候选名称：SRPRv2 (State-Routed Proximal Reconstruction v2)
分支：codex/n12-srprv2
共同基线：feature/prior-update-screen / b198de7
状态：IMPLEMENTED，待本地完整验证
```

## 核心假设

现有轻量 SISR 阶段主要串联特征变换，没有显式维护由当前观测残差驱动并回写主干的跨阶段重建状态；N12 通过 LR persistent state、固定观测一致性、学习型 proximal 更新和 stage feature writeback 改变阶段组织。如果去掉 residual、state 或 writeback 后性能不降，则假设失败。

## 与 N11 的实质区别

N11 采用 HR 48-channel state 和 average-pool surrogate，且 proximal feature 不回写原始八阶段主干。N12 将 state/feature/writeback 放在 LR 48-channel 分辨率，HR 只保留 3-channel estimate 和 3-channel backprojection；每个 proximal correction 明确写回下一 stage 的 `feat`，使用显式 bicubic analysis 与精确矩阵转置。N12 不继承 N11 代码，不修改 `lfmn.py`，不更新 `fs`/`beta/gamma`，不使用 Token、蒸馏、FFT 或额外损失。

## 公式与形状

```text
xhat_i = x_base + U(q_i)
r_i = y - D(xhat_i)
b_i = D^T r_i
a_i = A(b_i)
c_i = Phi(feat_i, h_i, a_i)
g_i = sigmoid(G(feat_i, h_i, a_i))
h_(i+1) = h_i + g_i ⊙ (c_i - h_i)
feat_(i+1) = backbone_i(feat_i) + P_i(feat_i,h_(i+1),a_i)
q_(i+1) = q_i + S_i(h_(i+1),a_i)
```

`y/r/a`: `B×3×h×w`; `feat/h`: `B×48×h×w`; `b`: `B×3×4h×4w`; `xhat`: `B×3×4h×4w`; `q`: `B×3×h×w`。

## 首轮协议

N12 从零训练 20 epoch；B0 复用 N9 已完成从零 B0：`/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/scratch/n9_pcstr_20e_seed1/baseline`。DIV2K `1-800/801-900`、x4、HR patch 256、batch 4、seed 1、L1、Adam、lr 2e-4、Cosine T_max 150、eta_min 1e-6。20 epoch 不通过不得进入 40 或 1000 epoch。

## 验证

必须通过 bicubic D/D^T float64 内积、尺寸、有限性、proximal/writeback梯度、输入扰动因果、真实batch、保存重载和效率 profile。每轮记录 PSNR/SSIM、逐图指标、参数、MAC/FLOPs、显存/延迟和逐阶段 residual/state/gate/writeback 诊断。未测字段不得填写为结果。

## 服务器入口

```bash
B0_REFERENCE=/root/autodl-tmp/Image-Super-Resolution-N9/experiment/all_runs/scratch/n9_pcstr_20e_seed1/baseline \
DATA_ROOT=/root/autodl-tmp/Image-Super-Resolution/datasets \
PYTHON_BIN=python GPU=0 \
bash repro/run_n12_srprv2_screen_server.sh n12/srprv2_20e_seed1
```

## 闸门

20 epoch 末 5 轮相对 B0 稳定为正、逐图 CI 下界为正、至少 60% 图像正收益、SSIM 不退化、机制诊断非零且效率预算合格，才允许 40 epoch；40 epoch通过后才设计 T_max=1000 的正式训练。否则 NO-GO，不用长训赌反转。
