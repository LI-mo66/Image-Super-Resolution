# P0 / P1 / P2 服务器执行说明

2026-10-10：用户明确本机只负责准备、检查、提交代码；真实数据诊断、五benchmark评测及训练在服务器执行。本轮入口只有固定权重推理，不启动训练。所有实际收益和问题证据等待服务器产物。

## 分支与任务

| 分支 | 任务 | 真实forward是否改变 |
|---|---|---|
| codex/p0-lfmn-diagnostics | 原始LFMN统一逐图评测、内容分层及统一调度/汇总 | 否 |
| codex/p1-sfml-diagnostics | SFML beta/gamma统计和全阶段固定±2%单因素敏感性 | 四个临时hook干预；p0观察不改变 |
| codex/p2-iasa-diagnostics | 原IASA候选槽位、几何组成及CPU重构注意力 | 否 |

共同基础来自原模型提交1e51b2068b62a793860e7b7192a43877d9ccc2f1；P1/P2从登记的P0公共诊断基础建立，原模型blob完全一致，不继承结构候选。P编号按用户本轮要求统一管理。

## 拉取与检查

在服务器已有仓库根目录执行，先确保无未提交文件，再拉取。不要在有其他实验依赖的checkout中强制切换或重置文件。

```bash
git status --short
git fetch origin
git switch --detach origin/codex/p0-lfmn-diagnostics
git log -1 --oneline
python -c "import torch,cv2,PIL,numpy,einops,matplotlib; print(torch.__version__); print(torch.cuda.get_device_name(0))"
python repro/run_p_server.py --plan --data-root /实际数据根目录 --gpus 0
```

数据根目录应包含 `DIV2K/DIV2K_valid_HR`、`DIV2K/DIV2K_valid_LR_bicubic/X4` 和 `benchmark/{Set5,Set14,B100,Urban100,manga109}/{HR,LR_bicubic/X4}`。支持 Manga109 大写目录见下方约定。使用官方三个权重文件 `LFMN/model/scale2_model_996.pt`、`scale3_model_969.pt`、`scale4_model_939.pt`，运行自动记录SHA256并strict加载。历史权重训练seed、优化器及源提交未知；只复用冻结推理，不能当训练曲线基线。

`--plan`只解析refs与命令，不执行模型、不创建worktree。执行时为三个分支解析具体commit、核对原模型blob，并在ignored `experiment/p_server_worktrees/`建立detached工作树；已有工作树如dirty或commit不符则拒绝复用，不覆盖。

## 首轮诊断

单GPU：

```bash
python -u repro/run_p_server.py --data-root /实际数据根目录 --gpus 0
```

两张空闲GPU：

```bash
python -u repro/run_p_server.py --data-root /实际数据根目录 --gpus 0 1
```

每张指定GPU最多一个子进程，同一GPU的任务排队。GPU号必须是当前可分配的物理GPU ID，启动器通过CUDA_VISIBLE_DEVICES隔离。它不替你检查其他用户训练的占用；运行前用nvidia-smi确认空闲。不自动关机，不修改服务器环境，不启动训练。

首轮固定×4、DIV2K valid0801–0808中心LR64（前4选择、后4检验，非保证历史未用的盲测），五benchmark各第一图中心LR64只是覆盖烟雾。P1每图共五遍（p0、beta±2%、gamma±2%），P2一遍。未分类规则周期纹理，当前掩码仅平坦／单方向／多方向／其他。不要将这里五张裁块均值写成完整benchmark成绩。

每个子进程先运行奇数/非方形前向、观察或零干预与P0一致、hook移除、state含buffer不变等检查。未通过就停止，不运行该任务真实数据。诊断成本含CPU统计、同步、图像度量，不能当推理延迟benchmark。

## P0完整五benchmark

先建立原模型×2/×3/×4完整成绩；本轮不在五benchmark反复挑干预参数。

```bash
python -u repro/run_p_server.py --mode benchmark --schemes P0 --scales 2 3 4 --data-root /实际数据根目录 --gpus 0
```

多GPU可传`--gpus 0 1 2`，按倍率分配独立进程。原图无chop、无自集成；若大图OOM保留失败，不临时修改某组裁块或chop后继续比较。单卡建议先按默认×4裁块诊断确认环境，再做全量评测。脚本不假定全图适合每台服务器显存。

HR统一从左上modcrop到LR×scale；PSNR沿原utility RGB255量化、Y差值系数[65.738,129.057,25.064]/256、裁边scale；SSIM沿原不同BT601系数[65.481,128.553,24.966]和Gaussian11 σ1.5。辅助未量化误差单列。

## 产物与返回

唯一比较目录：`experiment/all_runs/P_compare_<mode>_<时间>/`。

```text
launcher_manifest.json            # refs、实际commits、GPU、命令、PID、退出状态
P0_x4_launcher.log                # 启动器stdout/stderr，终端仍实时显示
P1_x4_launcher.log
P2_x4_launcher.log
P0/x4/<唯一运行目录>/
P1/x4/<唯一运行目录>/
P2/x4/<唯一运行目录>/
    config.json                  # 权重/源码/输入/插件hash、协议、状态
    checks.json                  # 工程检查
    diagnostic_log.txt           # 推理进程stdout/stderr实时镜像
    metrics.csv                  # 逐图完整浮点PSNR/SSIM/误差
    per_image.json               # 区域误差、对齐坐标、机制统计
summary_x4.json
summary_x4.md
```

自动汇总只接受completed运行；P1/P2未干预行须与独立P0的逐图指标、输入hash、坐标、区域完全一致；完整协议不一致即拒绝。选择/检验/benchmark smoke分开，图片bootstrap不代表多训练seed。P2属于观察相关证据，P1属于当前权重敏感性证据，二者不直接排名PSNR结构收益。

请返回整个比较目录（无新权重生成，不需复制数据集）。若失败返回launcher_manifest、对应launcher log和已有config/diagnostic log，不删除失败目录。汇总器和运行入口不会把结果提交Git。

## 下一阶段

收到真实诊断结果后再定位要修改的唯一机制，明确可证伪假设、独立结构实现、统一B0训练协议、短筛预算和停止线。训练前必须通过《深度学习训练日志管理规范.md》的日志烟雾；本轮推理日志不等于训练日志规范已满足。
