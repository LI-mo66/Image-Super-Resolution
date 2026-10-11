# 原始 LFMN ×4 从零训练交付

本轮仅建立原模型控制组，不属于 F 系列创新。独立分支 `codex/original-lfmn-x4-reproduction`；不改动共同模型、历史实验或当前未提交训练入口。历史权重和旧 B0 曲线均不复用。

## 来源与可比范围

- 论文：用户本地 LFMN.pdf，III-A 和 Table II。训练 DIV2K，8 MGRB，LR64×64，Adam β=(0.9,0.999)、ε=1e-8，初始 LR2e-4，每200 Epoch减半，1000 Epoch，RTX4090。
- 官方仓库：[hehesjtu/LFMN](https://github.com/hehesjtu/LFMN)。本地保留的官方提交 `4fe1614ef24f03bee81d53f5d7ec7df91606d93b`，其编译模型和 common.py 的 Git blob 与本仓库相同。官方模型 SHA256 `cb147a6d540e384e6233a23c28dbdc54ee1421af0f6f40dece5e32817fcc37ee`。
- 该源码 option.py 默认 Batch=8、test_every=1000、L1；SRData repeat=(8×1000)//800=10，8000 patches/Epoch，1000 optimizer updates/Epoch。1000 Epoch据此推导为1,000,000 updates。论文没有明确 Batch/总Iterations，不能宣称作者实际长训命令已获证实。论文参数优先于源码中用于演示的20 Epoch、1e-4等默认值。
- 本次全部不使用 self-ensemble、x8、chop、AMP、TF32、EMA、预训练、蒸馏或新模块；模型只有一次整图前向。官方演示默认打开 x8，但论文 Table II 未明确 x8 状态，论文成绩与本次 no-x8 的差异不得直接解释为实现退化或保证相等。
- 原始模型由官方提供的 CPython3.9/Linux x86_64 编译扩展运行，**不会自动退回 Python 重实现**。Windows/Python3.10 的本地重实现仅获准用于工程诊断，不能发布为原模型性能证据。编译模型的兼容性、训练前向、GPU资源和两轮完整评测必须由服务器闸门实际验证。

## 预先固定的协议

|项目|固定值|
|---|---|
|训练数据|DIV2K 0001–0800，官方提供的 bicubic ×4 LR配对；不在线重采样|
|patch / batch|LR64×64，HR256×256；Batch8；不做梯度累积|
|优化|FP32 Adam，2e-4，β0.9/0.999，ε1e-8，weight_decay0，L1|
|进度|1000 Epoch，每轮1000更新；200/400/600/800/1000轮结束LR×0.5|
|增强|官方 paired crop/横翻/竖翻/transpose；seed1|
|验证|每轮完整 Set5(5)、Set14(14)、B100(100)、Urban100(100)、Manga109(109)|
|统一选模|五数据集各自逐图平均PSNR的等权平均最大；相等保留较早Epoch|
|PSNR|SR clamp0–255并round；float32 `(SR−HR)/255`；官方Y系数 `[65.738,129.057,25.064]/256`；HR裁边4；逐图PSNR均值|
|SSIM|SR/HR量化；HR裁边4；float64 MATLAB BT.601 Y；Gaussian11×11 σ1.5；valid区域，C1/C2使用255|

PSNR与SSIM使用官方源码的不同Y系数，不能悄悄统一成另一个公式。HR若未modcrop，只允许多出0–3像素并从右/下对齐裁到LR×4；拒绝缺失、尺寸不匹配、alpha、非uint8或非标准数量。数据检查解码全部1128对并对文件内容计算SHA256；fingerprint覆盖数据、模型、评测、训练入口和日志版本。Linux允许 `manga109` 小写目录。

五个benchmark均参与选模，这是用户要求的预注册选模标准，不是额外独立盲测。独立PSNR-best与SSIM-best Epoch只用于诊断；report中的正式best五数据集成绩全部来自同一个checkpoint。最终1000轮权重另外完整报告。

## 4090 环境与数据

租用 Linux x86_64、RTX4090，创建 **Python3.9** 环境。不要直接用Python3.10/3.11加载原扩展。原扩展为CPython扩展而非需要重新编译的CUDA核；仍需实际通过服务器导入/梯度检查。建议使用支持Python3.9的PyTorch2.5.1 CUDA12.4环境；这不是论文披露的环境，实际版本写入config。若环境报错，保留失败日志，不改模型或降低Batch绕过。

```bash
conda create -n lfmn_original python=3.9 -y
conda activate lfmn_original
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r repro/requirements-original-lfmn-py39.txt
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

安装命令来源：[PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。使用完整交付包解压后的根目录，或Git分支；不要把其他创新目录拷入运行路径。数据结构：

```text
datasets/DIV2K/DIV2K_train_HR/0001.png ... 0800.png
datasets/DIV2K/DIV2K_train_LR_bicubic/X4/0001x4.png ... 0800x4.png
datasets/benchmark/<Set5|Set14|B100|Urban100|Manga109>/HR/<name>.png
datasets/benchmark/<set>/LR_bicubic/X4/<name>x4.png
```

数据下载、解压由用户完成；交付包不包含数据、预训练或实验结果。论文/数据许可沿用原作者来源。

## 先验收，再从零长训

```bash
export DATA_ROOT=/你的数据根目录
bash repro/run_original_lfmn_x4_server.sh check-data
bash repro/run_original_lfmn_x4_server.sh preflight
```

preflight首先检查日志、指标与checkpoint/RNG一致性，然后启动原编译模型的完整第1轮，保存后退出，再完整恢复第2轮。总计2000更新；两轮均完整验证五数据集。检查CSV旧行不变、日志追加、模型/优化器/scheduler/RNG/loader恢复、latest/best/final、config和曲线，全部成功才生成 `preflight_passed.json`。模型训练期聚类含CUDA非确定性操作，不能保证重启与不中断长训位级一致；在同一已提交Epoch边界恢复完整状态，不复用已消费的partial Epoch。失败不会生成凭证，也不会自动开始长训。

通过后根据输出的**完整凭证路径**运行：

```bash
export PREFLIGHT_RECEIPT=/完整路径/preflight_passed.json
bash repro/run_original_lfmn_x4_server.sh train
```

正式训练创建全新目录和全新随机初始化；不加载preflight权重。正式入口拒绝无凭证、非官方backend、任何烟雾限制、数据/源码/GPU/PyTorch不同的凭证。新正式运行与验收必须使用相同seed和worker设置。默认8 workers。不会自行改Batch、patch、LR、迭代数来适应资源；OOM时停止并保留日志，查服务器资源。

断点续训：

```bash
export RUN_DIR=/完整路径/LFMN_original_x4_seed1_时间戳_后缀
export PREFLIGHT_RECEIPT=/原来的完整凭证路径/preflight_passed.json
bash repro/run_original_lfmn_x4_server.sh resume
```

固定从该run的latest完整状态恢复，而不是加载任意外部权重。Epoch中途中断从最后完整训练+验证+保存的Epoch重做下一轮，保留总更新预算，不伪称保留了未提交的partial Epoch。日志新session追加。latest原子提交后CSV可从checkpoint历史恢复；文件系统/磁盘错误会停止进程。best保存全状态；用于正式续训的入口仍统一选择latest以保护时间线。SIGKILL后若遗留run.lock，核实其中host/PID进程已退出后人工移除；不可在训练活跃时删锁。

## 产物与时间口径

每次训练自动创建 `experiment/all_runs/LFMN_original_x4_seed1_<time>_<unique>/`：

- `train_log.txt`：实时stdout/stderr，包括异常traceback；`config.json`：源码、环境、协议指纹、session和结果。
- `metrics.csv`：每轮loss、使用的LR/下一轮LR、五数据集当前PSNR/SSIM、各指标历史best与Epoch、统一选模分数、训练/验证/累计秒数。
- `checkpoints/latest.pt`、`best.pt`、`final.pt`：完整模型、buffers、优化器、调度器、RNG、数据generator、global_step和指标历史；final仅在固定预算完整结束时保存。
- `per_image_epoch_XXXX.json`、`data_manifest.json`、`summary.json`、`report.md`、`psnr_curves.png`。

`elapsed_seconds`为已完成Epoch训练与验证时间之和，包含加载训练batch，不含首次数据检查、模型构建、绘图/保存与暂停时间。各session起止时间在config中；preflight凭证另有整段wall_seconds。正式结论须同时说明训练/验证时间与session实际墙钟范围。1000轮×328张整图验证属于本次指定协议，可能比作者原训练更耗时；未实测4090不编造小时数。

本地已完成的工程验证与服务器未完成项见实验记录。交付代码不包含未经完成的1000轮性能结果，不保证达到论文数值。
