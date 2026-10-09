# F2 与 B0：从零独立 150 epoch Cosine 对照完整交接说明

本轮只验证一个问题：在完全匹配的 150 epoch 从零训练协议下，已锁定的 F2 是否仍优于原始 LFMN（B0）。服务器顺序完成 B0、F2 两组；本机只做必要的检查和烟雾验证，不启动 150 epoch。150 epoch 后自动结束，不延伸至 1000 epoch，不自动关机。

## 1. 交付身份、证据和授权边界

| 项目 | 固定内容 |
|---|---|
| GitHub | https://github.com/LI-mo66/Image-Super-Resolution |
| 分支 | `codex/f2-attention-budget-reallocation` |
| F2 模型交付提交 | `110f4cc` |
| 20 epoch 结果审计提交 | `08edba3` |
| 本轮完整实现提交 | `33140c1d5e252645539a1935379841a557425e29`（本轮运行固定此完整提交） |
| 运行入口 | `repro/run_f2_150_server.py` |
| F2 结构 | TAB/FFN 96 × 8；LRSA 64/80/112/128 双周期；参数量 759627；主要 MAC 与基线匹配 |
| 比较对象 | 原始 LFMN B0 与已交付 F2，各自随机初始化、独立训练 |
| 授权范围 | 两组 seed 1、每组完整 150 epoch；必要检查、烟雾、恢复和评测 |

不改原始 LFMN、不改已锁 F2 模型，不调结构、损失、优化器、学习率、增强或数据范围。历史其他候选不是本轮设计起点，也不是结果证据。本轮是 F2 的独立匹配对照验证，不开发新候选，不附加消融或多 seed。

此前 20 epoch F2 独立验证相对 B0 的 DIV2K 均值约 +0.0135 dB、Set5 均值约 +0.0593 dB，仅是短筛线索。不能保证 150 epoch 保持、扩大或复现该差值，也不能把短筛当作正式结构收益证明。20 epoch 权重、优化器或调度器状态一律不加载、不续训。旧20轮使用的也是Cosine T_max=150，其前20轮时间轴与本轮一致。本轮按用户明确要求，两组都重新从零跑满150轮，既有20轮不复用为本轮起点，因此 **Baseline reuse: NO（用户指定的新独立对照）**。

先阅读本仓库 `AGENTS.md`、`创新点开发与服务器实验SOP.md`、`深度学习训练日志管理规范.md` 和 F2 实验卡/检查报告。候选已完成既有开发阶段，本说明只管理用户批准的 150 epoch 验证；任何检查失败都不得绕过门槛进入正式训练。

## 2. 固定协议指纹

| 字段 | B0 与 F2 共同设置 |
|---|---|
| 数据根目录 | `/root/autodl-tmp/Image-Super-Resolution/datasets`（用户确认） |
| 训练数据 | DIV2K 0001–0800，完整 800 张 |
| 独立验证数据 | DIV2K 0801–0900，完整 100 张，不参与梯度更新 |
| 退化与倍率 | bicubic ×4 |
| patch / batch | HR 256 × 256、LR 64 × 64；batch 4 |
| seed / workers | seed 1；workers 4 |
| 每 epoch | 1000 batch，不擅自减步数 |
| 训练长度 | 各组 150 epoch；总更新各 150000 步 |
| 优化器 | Adam，初始 LR 2e-4，betas=(0.9, 0.999)，eps=1e-8，weight_decay=0 |
| 调度器 | Cosine，T_max=150，eta_min=1e-6；完整时间轴从头开始 |
| 损失与精度 | L1；FP32 |
| 初始化与恢复 | 新训练不加载任何预训练或旧筛选权重；仅故障恢复本轮对应组的完整状态 |
| 推理 | self-ensemble OFF；chop OFF |
| EMA | 无模型权重 EMA；模型原有 TAB 内部 EMA 保留，二者不是同一含义 |
| 执行顺序 | 同一 GPU、同一环境、同一入口串行 B0 后 F2 |
| 训练随机流 | 相同 seed、shuffle、crop、flip/rotation 流；以实际 LR/HR 输入哈希核对 |
| 评测 | 固定提交的同一评测实现；完整精度逐图 PSNR/SSIM，记录量化、颜色空间和边界裁剪口径 |

具体指标公式、rgb_range、增强实现、checkpoint 键与调度器调用时机，以固定提交的代码和机器可读协议为准，不按人工猜测修改。检查阶段需把实际字段写入 `protocol.json`/各组 `config.json`。无法确定的字段写 `null` 或 `unknown` 并报告；影响公平性的未知字段解决前不能开训。两组指纹除模型身份及其已锁结构外必须一致。

## 3. 服务器准备与数据检查

使用新源码目录，保留旧训练目录和旧环境。无需把数据复制进新 Git 仓库；通过已确认的数据根目录读取。数据与权重不提交 Git。先确认新目录不存在，以下命令会在冲突时停止。

```bash
set -euo pipefail
REPO=/root/autodl-tmp/Image-Super-Resolution-F2-150
DATA=/root/autodl-tmp/Image-Super-Resolution/datasets
OUT=/root/autodl-tmp/F2_150_experiments
COMMIT=33140c1d5e252645539a1935379841a557425e29

[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
test ! -e "$REPO"
git clone --branch codex/f2-attention-budget-reallocation \
  https://github.com/LI-mo66/Image-Super-Resolution.git "$REPO"
cd "$REPO"
git checkout --detach "$COMMIT"
test "$(git rev-parse HEAD)" = "$(git rev-parse "$COMMIT^{commit}")"
git status --short
git log -1 --format='%H %s'

# 使用已验证的服务器 Python/conda 环境，不盲目升级或重装 PyTorch。
which python
python --version
python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
nvidia-smi
test -d "$DATA"
df -h /root/autodl-tmp
python repro/run_f2_150_server.py --help
```

若提交尚未推送或找不到，报告问题并停止，不能改用分支最新未知提交。`git status --short` 出现源码修改时查明来源，不偷偷丢弃用户文件，也不带 dirty 代码开训。

检查器必须核实 HR/LR 配对、文件解码、×4 尺寸关系、训练与验证范围、Set5/Set14/B100/Urban100/Manga109 完整性。不能把缺文件的数据集改成子集、静默跳图、换数据、用训练集代替验证集，或把缺失值记为 0。缺数据要明确列出并报告；服务器数据准备可以按用户已给的数据路径核查，不能自行下载未知替代版本。

存储预算包括两组 150 个 epoch 的权重、完整恢复 checkpoint、逐图评测、日志及终点评测；不要为了腾空间擅自删除任何既有实验。空间不足时先报告。

## 4. 必须先通过检查与烟雾

```bash
cd /root/autodl-tmp/Image-Super-Resolution-F2-150
python -u repro/run_f2_150_server.py \
  --data-root /root/autodl-tmp/Image-Super-Resolution/datasets \
  --workers 4 --output-root /root/autodl-tmp/F2_150_experiments \
  --check-only

python -u repro/run_f2_150_server.py \
  --data-root /root/autodl-tmp/Image-Super-Resolution/datasets \
  --workers 4 --output-root /root/autodl-tmp/F2_150_experiments \
  --smoke-only
```

两条命令都必须非零失败即停止。`--check-only` 核查日志机制、结构不变量、协议与数据；`--smoke-only` 使用真实 B0 和 F2 完成每组 1 batch → 保存 → 恢复至 2 batch，并用 Set5 做权重重载推理核对。烟雾目录必须独立，不能成为正式训练起点；烟雾数值不能作为 PSNR 收益证据。

验收内容：前向/损失/梯度有限；参数量与锁定模型一致；严格保存重载；完整 stdout/stderr 实时镜像；训练中可读日志；重复启动不覆盖；受控异常有完整 traceback 和失败状态；config JSON 可解析；metrics CSV 顺序可靠；恢复追加 session 而不覆盖旧指标；固定 seed 单步一致性；单进程日志无竞争。把检查命令、退出码、产物路径、通过/失败项保留在检查报告。检查器、日志或真实链路任一必需项失败，先保存证据，不跳过检查开长训。

## 5. 正式运行：只有两组 150 epoch

确认检查通过、数据完整、源码固定、GPU 空闲后运行。下面 `nohup` 文件只作启动器备份；不能替代程序自身的 `train_log.txt`、`config.json`、`metrics.csv`。

```bash
cd /root/autodl-tmp/Image-Super-Resolution-F2-150
mkdir -p /root/autodl-tmp/F2_150_experiments
LAUNCH=/root/autodl-tmp/F2_150_experiments/launcher_$(date +%Y%m%d_%H%M%S).log
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
  python -u repro/run_f2_150_server.py \
  --data-root /root/autodl-tmp/Image-Super-Resolution/datasets \
  --workers 4 --output-root /root/autodl-tmp/F2_150_experiments \
  > "$LAUNCH" 2>&1 < /dev/null &
PID=$!
printf 'PID=%s\nLAUNCH=%s\n' "$PID" "$LAUNCH"
tail -F "$LAUNCH"
```

`Ctrl+C` 仅结束 `tail` 查看，不停止后台训练。另一个终端可运行 `nvidia-smi`、`pgrep -af run_f2_150_server.py`，并读取启动器打印的组内 `train_log.txt`。程序自动创建 `F2_pair150_x4_seed1_TIMESTAMP` 独立配对根目录，先完成 B0，再开始 F2；已有目录和 checkpoint 不得覆盖。

应监控 epoch、实际 LR、L1、DIV2K/Set5 PSNR/SSIM、checkpoint 保存、NaN/Inf、OOM、磁盘空间和完整 traceback。短暂曲线落后不能自行调参、早停或加模块。本轮长度预定为完整 150 epoch；发生技术故障则保留日志并处理恢复，而不是把未完成曲线当成最终结果。

## 6. 中断恢复与仅评测

`GROUP` 必须是启动器输出的 **配对根目录**，例如 `/root/autodl-tmp/F2_150_experiments/F2_pair150_x4_seed1_20261009_时间戳`，其中包含 `protocol.json`。不能传入B0/F2训练子目录，也不能传入旧20 epoch目录。恢复前确认原进程已结束，禁止两个进程写同一组。

```bash
cd /root/autodl-tmp/Image-Super-Resolution-F2-150
GROUP=/root/autodl-tmp/F2_150_experiments/F2_pair150_x4_seed1_实际时间戳
python -u repro/run_f2_150_server.py \
  --data-root /root/autodl-tmp/Image-Super-Resolution/datasets \
  --workers 4 --output-root /root/autodl-tmp/F2_150_experiments \
  --resume "$GROUP"
```

恢复必须恢复模型、optimizer、scheduler、epoch/global_step 和可靠的随机状态/数据进度；保留原 run ID、记录新 session、恢复 checkpoint SHA256 和起始位置，指标不得重复或倒退。若损坏或不兼容，报告并停下，不改成仅加载权重的“伪恢复”，不把调度器重新从 epoch 1 开始。入口自动检查两组状态：从中断组最后完整epoch继续，另一组若尚未开始则从零训练，已经完整结束的150轮不会重复训练。

只重新计算本轮已有权重的评测与汇总，不做参数更新：

```bash
python -u repro/run_f2_150_server.py \
  --data-root /root/autodl-tmp/Image-Super-Resolution/datasets \
  --workers 4 --output-root /root/autodl-tmp/F2_150_experiments \
  --eval-only "$GROUP"
```

同一份代码、数据和评测协议必须固定；仅评测不是重新选择协议的许可。

## 7. 全量产物与结果选择

配对根目录中的 `protocol.json` 和 manifest 记录运行身份、源码、环境、数据协议及各产物映射；组内目录必须含 `train_log.txt`、`config.json`、`metrics.csv`，并保存所有实际产物。具体叶文件名以固定提交的 manifest 为准，不能凭本说明猜文件名。

必须保留：

1. 两组正常或失败的完整 stdout/stderr、启动器日志、检查/烟雾报告、命令、环境和 Git 身份。
2. epoch 1–150 的模型权重与对应 SHA256；最新完整边界的 optimizer/scheduler/resume 状态（不是每个旧epoch都保存独立优化器快照）；checkpoint 保存确认。不能只留 best 或删掉低分轮。
3. 每 epoch 训练 L1、实际 LR、步数、时长及可靠训练指标；DIV2K 验证 100 张逐图 PSNR/SSIM、均值、所用权重/哈希；Set5 5 张逐图 score、均值、所用权重/哈希。
4. 两组各 epoch 首 batch 的 LR/HR 哈希及配对比对，作为输入流一致性证据；不匹配必须解释或停止公平性结论。
5. 两套终点五 benchmark 评测：固定 epoch 150；各模型仅由独立 DIV2K 验证均值选出的 best epoch。每个模型的同一 best 权重用于 Set5、Set14、B100、Urban100、Manga109 全套，不能按 benchmark 或单图拼最高分。Set5 不选权重。
6. CSV/JSON/Markdown 自动 comparison：每 epoch B0/F2 的 DIV2K、Set5、L1、LR、SSIM 及 F2−B0 差值；固定终点、last-5/last-10；配对逐图差值 mean/median/win-rate、bootstrap 置信区间；最终及 DIV2K-best 五集逐图与均值。
7. 机器可读原始完整精度数据，保留报告生成过程和所有日志；缺失值为 `null`/空白，并明确原因，不能记 0、不插值、不伪造缺失 epoch。

末 5/10 轮指同 epoch 配对差值的平均，即 epoch 146–150、141–150；不能取各模型不同轮的最高值平均。各模型 DIV2K-best 权重是预先规定的选择规则，须同时报告其 epoch、验证均值、SHA256，和固定 epoch 150 对照分开。五个 benchmark 分别报告，不能作无依据宏平均或以最高的一个替代全部结论。

同 seed 的逐图 bootstrap CI 描述当前验证图片上的配对不确定性，不能代表跨 seed 稳定性或随机训练方差。新训练在 RTX 4090/4080 上的浮点差别不是可自动解释的系统涨分；比较前需确认同机、同环境、同输入流、同指标代码。若报告效率，延迟必须在同一机器以一致计时方法实测；本轮首要判断 PSNR，参数和 MAC 匹配本身不证明延迟收益。

## 8. 预注册判定与归档

以下门槛针对固定 epoch 150 的独立 DIV2K 均值差 `Δ = F2 − B0`（dB），同时审阅 last-5/last-10、逐图分布、CI 和 SSIM；best 与 benchmark 作为单独辅助报告，不能替代主判定。

| 150 epoch DIV2K Δ | 本轮判定 |
|---|---|
| Δ ≤ 0 | 不通过“涨分”主张；记录本协议结果 |
| 0 < Δ < 0.03 | 小收益灰区，需重复验证；不宣称稳定结构收益 |
| 0.03 ≤ Δ < 0.05 | 保留候选，检查一致性后再规划后续 |
| Δ ≥ 0.05 | 优先计划多 seed 验证；仍非已证实跨 seed 结论 |

这些判定不自动授权额外训练。两组完成后只生成报告、保存和打包；多 seed、消融、1000 epoch 或新结构须后续用户明确授权。若协议不一致、输入流不匹配、日志/数据缺失或未跑满，写明不可判定项，不套用通过阈值。

打包完整结果根目录（包括检查/烟雾、失败 session、原始表、所有权重和恢复状态），不要只打 summary，不把数据集放入包：

```bash
cd /root/autodl-tmp
ARCHIVE=F2_150_experiments_$(date +%Y%m%d_%H%M%S).tar.gz
tar -czf "$ARCHIVE" F2_150_experiments
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
tar -tzf "$ARCHIVE" > "$ARCHIVE.contents.txt"
printf 'archive=%s\n' "/root/autodl-tmp/$ARCHIVE"
```

先确认训练及评测全部结束，再打包，避免归档正在写入的 checkpoint。包可能很大；提前保证空间。下载并核对 SHA256 后保留服务器原件直到用户确认。实验输出、权重、数据集、缓存不提交 Git；研究记录追加到 `创新思路与实验记录.md`，注明固定源码、模型权重、数据/倍率/评测、训练协议、设备、路径、完整精度结果、未知字段及结论。提交前检查 status、diff 和显式暂存列表，提交并推送当前独立分支；推送失败保留本地提交并报告原因。

## 9. 可完整复制给同学或 AI 的执行提示词

```text
你接手 Image-Super-Resolution 仓库的 F2/B0 150 epoch 配对验证。用户只授权原始 LFMN B0 和已锁定 F2 各自从零独立完整训练 150 epoch，随后评测、汇总和归档；不能附加消融、多 seed、1000 epoch 或关机。请先阅读仓库 AGENTS.md、创新点开发与服务器实验SOP.md、深度学习训练日志管理规范.md、F2 实验卡/检查报告和 F2_B0_150epoch_完整交接说明.md，并遵守检查门槛。

GitHub：https://github.com/LI-mo66/Image-Super-Resolution。
分支：codex/f2-attention-budget-reallocation。
固定完整实现提交：33140c1d5e252645539a1935379841a557425e29。请核对该完整提交存在并固定检出，不能用未知最新版。F2 模型交付提交110f4cc，20e审计08edba3；不加载旧20e权重/optimizer，不续旧训练。

只使用原 B0 与 F2：TAB/FFN96×8，LRSA64/80/112/128双周期，F2参数759627且主要MAC匹配。不改模型、不调参、不换损失/数据/评测，不读其他历史候选来继承结构或结论。已确认数据根目录 /root/autodl-tmp/Image-Super-Resolution/datasets。检查完整 DIV2K0001–0800训练、0801–0900独立验证，Set5/Set14/B100/Urban100/Manga109；缺数据报告缺失，不悄悄删图、替换或改子集。

共同协议：bicubic x4，HR patch256/LR64，batch4，seed1，workers4，1000batch/epoch；Adam2e-4 betas(.9,.999) eps1e-8 wd0，L1 FP32，Cosine T_max150 eta_min1e-6，从零跑满150。self-ensemble/chop OFF，无模型权重EMA，原TAB内部EMA保留。两组同GPU同环境串行B0后F2。固定源码和实际指标口径，未知字段写null或unknown，影响公平性的未知项未解决不能训练。旧20e约+0.0135dB DIV2K/+0.0593dB Set5仅短筛线索，不保证长训收益。

在新源码目录固定提交，沿用已验证环境，不盲目升级。先跑 python repro/run_f2_150_server.py --data-root /root/autodl-tmp/Image-Super-Resolution/datasets --workers 4 --output-root /root/autodl-tmp/F2_150_experiments --check-only，再同参数 --smoke-only。核查日志基础设施、结构、真实B0/F2各1batch保存并resume至2batch、Set5重载。失败时保留完整日志、traceback、config和metrics并报告，不跳检查。仅必要烟雾可在本机，150e只在服务器启动。

检查全部通过后，用同一命令去掉check/smoke标志正式运行；可nohup并实时tail。程序自动创建F2_pair150_x4_seed1_TIMESTAMP配对目录，protocol.json/manifest记录身份，各组实时train_log.txt、config.json、metrics.csv。nohup日志不能替代组内日志。不得覆盖已有目录或checkpoint。故障恢复只用本轮配对根目录（包含protocol.json）的 --resume GROUP，完整恢复optimizer/scheduler/RNG与进度，追加session；只评测用 --eval-only GROUP。开始恢复前确认旧进程已退出，不两进程竞争写。入口自动继续中断组并补齐尚未开始的另一组，跳过已完整完成150轮的组。

必须保存epoch1–150全部权重及SHA256、完整checkpoint/optimizer/resume状态、每epoch训练L1/LR/SSIM与DIV2K100图、Set5五图逐图/均值指标；记录两组首batch LR/HR哈希并验证配对输入流。固定epoch150评测完整五benchmark；另给各模型仅由独立DIV2K均值选出的best权重评测，同一模型同一best权重用于全部五集，Set5不选权重，不拼各数据集最高分。五集分别报告，不宏平均。

生成完整精度comparison CSV/JSON/MD：每epochB0/F2及delta，final、last5/last10，配对逐图mean/median/win-rate/bootstrapCI、SSIM；缺值null不是0，不插值。seed1逐图CI不代表跨seed稳定性。150e DIV2K delta≤0不通过涨分、(0,.03)重复验证灰区、[.03,.05)保留、≥.05优先多seed计划；门槛不授权加跑。检查协议/输入流公平性，否则报告不可判定。RTX4090/4080浮点差异不是系统涨分证据；效率须同机计时，本轮重点PSNR。

结束后不延长训练、不关机。打包整个F2_150_experiments根目录，含raw/all weights/checkpoints/日志/失败session/检查报告，不含数据集，生成SHA256和包文件清单。将实际命令、固定提交、GPU环境、协议、目录、两组完成情况、检查结果、所有完整结果和未知字段报告给用户。只把代码/研究摘要提交Git，输出与权重不进Git；追加创新思路与实验记录.md后适当验证、明确本地提交、可用时推送，失败报告原因。遇报错先保留证据，不能为赶进度绕过检查或改变协议。
```

## 10. 交付完成核对

- [ ] 完整实现提交已填入说明及复制提示词，已推送并在服务器固定检出。
- [ ] 检查/烟雾实际通过，CLI、GROUP语义、产物映射与固定代码一致。
- [ ] 两组源码/环境/数据/随机输入流/指标指纹匹配，无旧20e续训。
- [ ] B0与F2各完成150epoch，正常或失败日志完整，无覆盖。
- [ ] 全部epoch权重、恢复状态、逐图原始指标与比较表完整。
- [ ] 固定150与DIV2K-best两套五benchmark结果独立报告。
- [ ] 全结果包可读取，SHA256可核对，研究记录与结论保持证据范围。

本说明不是已完成训练报告。实际退出码、运行目录、数值和最终判定须从固定提交产生的真实日志与机器可读产物填写；尚未运行的结果保持未知。


## 补充：指标、锁和资源测量

固定评测口径为原utility.py：rgb_range255、quantize255；DIV2K验证PSNR用RGB/crop10，benchmark PSNR用原BT601系数/256的Y通道/crop4，SSIM沿原MATLAB风格实现。不改变任何基础评测代码。`validation_per_checkpoint.csv`保存逐图double均值，`metrics.csv`同时保留原Float32日志读数；报告以两组同一完整precision逐图均值作主要比较，极小累积差异不视为结构收益。

所有相对训练目录由`protocol.json`的`runs`映射，完整路径名称含方案、seed和时间。新增`dataset_manifest.json`记录所有LR/HR文件内容SHA256；`first_batch_hashes.csv`是每轮**首个**batch的核验，不等同逐batch全训练数据转储。禁止只凭首batch哈希宣称所有GPU运算位级确定。

同一运行组使用`.runner_lock.json`独占锁。正常退出/可捕获异常会清锁；强制杀死可能留下锁。恢复前读锁里的主机/PID并核实该进程已经不存在，确认没有训练仍在运行后才由人工处理这个单独锁文件；不要删除训练日志、权重或指标，不要同时启动两个resume。

完成正式训练后默认还测同机FP32网络forward效率：LR64/128、batch1、预热30次、重复100次、CUDA同步，保存中位数/P10/P90/波动和显存原始样本，结果在`resource_OFF_时间戳/`，`resources_latest.json`指向最新版。完整FLOPs未知，留空/null，绝不填写0。FFN卷积/线性MAC只覆盖FFN，不冒充整网成本。资源测量失败会保留PSNR产物、日志及失败状态，报告效率未知；需要只做质量对照时可明确加`--skip-resource-profile`，这不跳过训练/日志闸门。

汇总写入`comparison_reports/时间戳/`，组根目录`comparison_latest.json`指向有效报告。完整结果包括每轮comparison_metrics.csv、final150和best_div2k两套benchmark及配对逐图差值、final/last5/last10和按图片bootstrap区间。若某模型独立验证best恰好是150，允许复用同一固定150评测，避免重复测同一个权重。任何数据缺失或协议不一致均不生成新的成功汇总，不覆盖旧报告。


### 大压缩包分卷（保留所有权重，不删原件）

两组各150个权重的结果包可能超过单次上传大小。完整tar.gz生成并校验后可分成256MiB：

```bash
split -b 256M -d -a 3 "$ARCHIVE" "$ARCHIVE.part_"
sha256sum "$ARCHIVE".part_* > "$ARCHIVE.parts.sha256"
```

上传全部分卷与sha256文件；接收端按编号拼接再核对完整包SHA256。不要只传最高分轮或遗漏逐图文件。

### 权威文档与固定代码提交

本说明通过最终分支单独归档，训练代码固定为表格中的完整40位实现提交。把这份MD和第9节提示词直接交给同学；克隆后切换固定实现提交即可运行，不依赖说明文档所在的后续归档提交。再次恢复时仍保持训练时的固定提交，不能因为文档更新而切到分支新HEAD。


## 本地交付验证记录（2026-10-09，非150e性能结果）

RTX4060Laptop/Python3.10.20/PyTorch2.9.1+cu128/CUDA12.8：B0与F2各两个真实DIV2K单batch工程epoch，保存、恢复optimizer/scheduler/RNG/data generator、独立验证与Set5重载均通过；两组每轮首batch LR/HR哈希一致，公共初值fingerprint一致。CPU固定seed单步对照证明输入观察前后输出/损失/参数和buffer更新精确相等；运行组独占锁拒绝并发并正常释放。

公共日志实时stdout/stderr、期间可读、两次运行非覆盖、受控异常及resume追加已验证；150入口已有下一轮权重时拒绝覆盖，完整Traceback与failed状态保留。完整150轮/五benchmark汇总使用明确synthetic工程fixture，正常完整数据、11类缺失/混配错误拒绝及best==150精确复用均通过。fixture不是训练结果，不能把它的数字写成PSNR增益。

本机未启动正式150e；服务器会在固定提交下自动重做必要检查。真实GPU效率要等正式两组150轮后才能填入报告；完整FLOPs未实现计数时始终未知。
