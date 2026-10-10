# P0 / P1 / P2 原始 LFMN 问题诊断

2026-10-10 用户指定新 P 编号；本轮为固定权重问题深挖，不是结构短筛训练。

| 编号 | 分支 | 问题 | 当前执行范围 |
|---|---|---|---|
| P0 | codex/p0-lfmn-diagnostics | 原模型错误分布 | 统一量化Y评测、逐图和内容分层 |
| P1 | codex/p1-sfml-diagnostics | 浅层先验在阶段内如何作用 | beta/gamma在线统计与固定±2%微扰 |
| P2 | codex/p2-iasa-diagnostics | 内容候选与几何兼容性 | 真实排序重叠窗口、抽样per-head注意力与等权组成 |

共同原模型源提交 `1e51b2068b62a793860e7b7192a43877d9ccc2f1` 的 `lfmn.py` 与本轮主工作区原模型无差异。仅导入原模型；不继承任何旧候选。固定 `eval_refine_iters=0`、`normalize_overlap=False`、eval、FP32、无chop、无自集成。公共诊断入口独立于训练器，不接触用户未提交日志实现。

## 预注册与资源

- 首轮 DIV2K valid ×4：0801–0804选择、0805–0808检验，中心LR64；这些图不保证历史未使用，称检验划分而非新盲测。
- 五benchmark每集排序第一张中心LR64用于覆盖烟雾，绝不能记为完整benchmark成绩。全图全量命令另列。
- LR区域：固定归一化Sobel/8，5×5均值结构张量，energy≥1e-4，coherence≥.7为单方向边缘，<.3为多方向；其余单列。不能从这些掩码识别规则周期纹理或宣称全部“复杂纹理”。
- P1四个全阶段干预预先固定，不逐图选符号或在五benchmark调参；排序变化率不是聚类成员变化率。
- P2固定阶段0/3/7，每阶段24个均匀原始像素query；attention质量必须和同一候选集等权组成比较；跨标签、重复槽位和唯一源像素分别报告。
- 用户指定本机只准备检查提交，服务器执行实际诊断。服务器多GPU支持并行，每张GPU最多一进程；单GPU排队；CPU源码审计、检查和汇总可并行。日志、权重、JSON、CSV保存在ignored experiment，不提交Git。
- 未开始训练；任何结构实现/短筛须按SOP新卡及独立分支、完整训练协议、基线复用决定、训练日志烟雾。诊断完成不等于进入SCREENING。

## 固定评测口径

HR从左上modcrop到LR×scale，中心裁块严格按同坐标缩放；不重新生成bicubic。
PSNR：SR RGB255 clamp/round，Y差值系数[65.738,129.057,25.064]/256，HR裁边scale；SSIM沿用原utility的BT601系数[65.481,128.553,24.966]和Gaussian11 σ1.5。二者系数不同，明确记录，不笼统称完全相同Y实现。
辅助未量化误差不得替代标准量化PSNR。区域按图片统计，不以像素作为独立样本产生置信区间。

## 运行

以下为单入口接口示例，不在本机执行真实数据。服务器统一拉取/调度命令见《P系列_服务器执行说明.md》。各分支运行自己的 `repro/run_p_diagnostics.py`。

```powershell
# P0 首轮预注册裁块诊断
& E:/anaconda/envs/dl/python.exe repro/run_p_diagnostics.py --scheme P0
# P1 / P2 各自在自己的分支中执行
& E:/anaconda/envs/dl/python.exe repro/run_p_diagnostics.py --scheme P1 --plugin repro/p1_sfml_probe.py
& E:/anaconda/envs/dl/python.exe repro/run_p_diagnostics.py --scheme P2 --plugin repro/p2_iasa_probe.py
# 完整五benchmark，空div2k-ids表示不评DIV2K；每倍率独立目录
& E:/anaconda/envs/dl/python.exe repro/run_p_diagnostics.py --scheme P0 --scale 4 --benchmarks full --crop-lr 0 --div2k-ids
```

每次运行建立唯一目录，config记录权重/源码/数据hash、commit及dirty、精度、口径、裁块、插件hash、状态与耗时；diagnostic_log实时镜像stdout/stderr；metrics.csv及per_image.json逐样本写入。无训练指标，不伪造epochs或训练曲线。

## 解释边界

首轮是问题证据和投资排序：P1真实微扰能支持特定checkpoint敏感性；P2观察只能支持相关线索。二者不是同等证据强度。少量裁块、单权重、单图驱动均不能证明结构收益、跨倍率泛化、行业领先或整个方法族失败。完整五集、独立训练和多seed仍待执行。
