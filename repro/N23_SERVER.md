# N23服务器短筛（不是保证涨点）

分支`codex/n23-hierarchical-scc`，应核对本次交付答复中的commit。使用已有CUDA PyTorch环境，不自动升级依赖，以免旧B0环境指纹失配。GPU0必须空闲，正式入口要求≥16GiB，建议24GiB；不操作已有任务。

## 1. 独立克隆

```bash
cd /root/autodl-tmp
git clone --branch codex/n23-hierarchical-scc \
  https://github.com/LI-mo66/Image-Super-Resolution.git Image-Super-Resolution-N23
cd Image-Super-Resolution-N23
git log -1 --oneline
git status --short
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

不要浅克隆；旧B0审计需要完整Git历史。仓库不包含DIV2K和历史权重，它们必须留在原路径。若GPU有N16等任务，请另用专用实例，不要终止它们。

## 先单独跑N23（2026-10-08用户新增授权）

旧B0当前不可用，可以主动选择--n23-only。此模式不访问旧B0、不训练新B0，smoke也只更新N23一步；正式scratch40轮，其它训练字段不变。仍保存数据/source/environment指纹、40权重、40×100图指标，便于之后同协议合法比较。没有B0时不GO/NO_GO、不称完整配对筛查完成、不自动延长；目标服务器相对效率闸门待补，预检查只记录N23资源，不宣称相对通过。

```bash
cd /root/autodl-tmp/Image-Super-Resolution-N23
git pull --ff-only origin codex/n23-hierarchical-scc
git log -1 --oneline
export DATA_ROOT=/root/autodl-tmp/datasets
export OUTPUT="$PWD/experiment/all_runs/n23_only_40e_seed1_r1"
nohup bash repro/run_n23_screen_server.sh \
  --n23-only --data-root "$DATA_ROOT" --output "$OUTPUT" --run \
  > n23_only_launcher.log 2>&1 < /dev/null &
echo "后台PID=$!"
tail -F n23_only_launcher.log
# 完成后，无需打包
cat "$OUTPUT/n23_epoch_metrics.csv"
cat "$OUTPUT/n23_summary.json"
```

后台启动前核对答复中的commit；输出目录必须不存在。可选自动关机仍需显式--shutdown-on-success --dedicated-instance，单组结果完整后请求关机，不等待B0；报错不关机。只收集N23不意味着旧B0原始分数现在可以合法复用。

## 2. 旧B0审计（不训练、不创建实验输出）

以下路径对应本对话此前服务器目录；按实际位置修改BASELINE/DATA_ROOT，不要填不存在的tar包。

```bash
export BASELINE=/root/autodl-tmp/Image-Super-Resolution-N21N22/experiment/all_runs/n21_n22_40e_seed1_r1/b0
export DATA_ROOT=/root/autodl-tmp/datasets
export OUTPUT="$PWD/experiment/all_runs/n23_40e_seed1_r1"
bash repro/run_n23_screen_server.sh \
  --baseline "$BASELINE" --data-root "$DATA_ROOT" --output "$OUTPUT"
```

只有`AUDIT_PASS`才考虑复用。需要旧B0 config、父manifest、initial proof、batch fingerprints、mechanism、optimizer/scheduler、log、40个model_1.pt...model_40.pt及完整precision curves。比对源码b198de7、旧私有入口、1800数据哈希、scratch/LR64/batch4/seed1/增强/Adam/L1/CosineT150/eta1e-6/40×1000、旧及新torch/CUDA/GPU/driver/Python/libraries。任一缺失/变化停止；不自动重训。旧HR batch hash/cuDNN信息缺失另行记录，不能称每个环境/数据字段有位级证明。

即使权重可复用，旧legacy全图指标也不能直接比较。运行后按新exact口径重新评测40权重×100图，重评同样耗GPU时间。若只有最终权重，只能做最终诊断，不能满足本次完整40e闸门。

## 3. 审计通过后，后台只训练N23

```bash
(
set -e
bash repro/run_n23_screen_server.sh \
  --baseline "$BASELINE" --data-root "$DATA_ROOT" --output "$OUTPUT"
nohup bash repro/run_n23_screen_server.sh \
  --baseline "$BASELINE" --data-root "$DATA_ROOT" --output "$OUTPUT" --run \
  > n23_launcher.log 2>&1 < /dev/null &
echo "后台PID=$!"
)
```

再次审计→target工程/真实单batch smoke→formal数据流零更新probe→cache-miss资源检查→旧B0统一重评→N23 scratch40轮→完整性审计→汇总。任何失败停止，无隐式resume或覆盖；重启必须选择新输出目录，先定位失败，不盲目重跑。

正式固定：DIV2K1–800/801–900、×4、HR256/batch4/seed1、Adam2e-4、L1、FP32/TF32off、CosineT_max150/eta_min1e-6、40epoch×1000更新。不是CosineT40，也不是MultiStep。正式训练不加载历史B0或smoke到N23。

## 4. 旧B0不能复用时的独立选择（增加成本，非自动回退）

如GPU/driver不同或历史证据缺失，先把audit报错发回讨论。如果你明确愿意付新B0成本，可主动执行：

```bash
export OUTPUT="$PWD/experiment/all_runs/n23_newpair_40e_seed1_r1"
nohup bash repro/run_n23_screen_server.sh \
  --train-new-baseline --data-root "$DATA_ROOT" --output "$OUTPUT" --run \
  > n23_newpair_launcher.log 2>&1 < /dev/null &
echo "后台PID=$!"
```

该模式N23→新B0各40轮，共80组轮次，另各一步smoke；不再引用旧B0分数作为主对照。旧模式与新模式二选一，不同时启动。

## 5. 进度及结果

```bash
tail -F n23_launcher.log
# 新pair改为tail -F n23_newpair_launcher.log
cat "$OUTPUT/manifest.json"
# 训练开始后
tail -F "$OUTPUT/n23_console.log"
# 完成后直接输出，无需打包
cat "$OUTPUT/epoch_comparison.csv"
cat "$OUTPUT/decision.json"
```

Ctrl+C仅结束tail。nohup使SSH断开、本机关机不终止远端训练；不能防止OOM、进程异常、服务器断电或平台关机。汇总不足40轮只诊断、不GO。完整阈值：final与末5均≥+.010dB、100图CI下界>0、median>0、win≥60%、SSIM≥−1e-4。NO_GO不自动延长。Bootstrap单位图片，仅此100图/一次运行，不等于多seed保证。

## 6. 可选自动关机

默认不关机。仅专用实例、没有N16等其它任务时，在上述nohup命令`--run`之后加：

```text
--shutdown-on-success --dedicated-instance
```

会在两组指标完整、完整性检查与汇总成功后请求整个AutoDL实例关机；方法NO_GO也属于实验正常完成。报错/缺指标不关机，实例可能持续计费，要查看日志并手工停机。helper缺失或syntax检查失败则不开始训练。无shebang shell helper用bash显式执行避免Exec format error；实际平台是否关机仍需确认，不能把发出请求等同于关机已成功。关闭整个实例有丢失其它任务风险。
