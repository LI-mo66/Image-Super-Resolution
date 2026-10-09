# V1N23C 独立服务器交付

仅V1+校准SCC写回；不是旧V1N23续训，不含N21/N22。默认从零20epoch、Cosine T_max150、eta_min1e-6、batch4、HR256、seed1、8worker、DIV2K×4、L1+C1输出KD0.1。不要改成T_max20或MultiStep再与历史V1比较。

旧V1不在当前服务器：使用collect-only，不访问或重训V1。结果只标PENDING_MATCHED_BASELINE；之后提供V1完整证据审计。历史差值不等于合法同协议结构增益。

## 1. 新目录克隆

```bash
(
set -euo pipefail
cd /root/autodl-tmp
git -c http.version=HTTP/1.1 clone --branch codex/v1-n23-calibrated-writeback \
  https://github.com/LI-mo66/Image-Super-Resolution.git Image-Super-Resolution-V1N23C
cd Image-Super-Resolution-V1N23C
git log -1 --oneline
git status --short
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install -r requirements-server.txt
python repro/test_v1n23c_server.py
)
```

使用现有PyTorch镜像（建议同前次2.5.1+cu124），requirements不安装torch。网络失败停止，不能接着训练或关闭TLS校验。已存在目标目录不要删除/重置，另选新目录；克隆失败的目录保留。

## 2. 教师：已有则复用，缺少才准备

已有教师可以把下列环境变量改为真实绝对路径，避免重复下载：

```bash
cd /root/autodl-tmp/Image-Super-Resolution-V1N23C
export DATA_ROOT=/root/autodl-tmp/datasets
export TEACHER_REPO="$PWD/repro/swinir_ref"
export TEACHER_WEIGHT="$PWD/repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
python repro/prepare_v1n23c_teacher.py --repo "$TEACHER_REPO" --checkpoint "$TEACHER_WEIGHT"
```

准备脚本不装依赖、不训练、不会reset已有教师仓库。源必须6545850fbf8df298df73d81f3e8cba638787c8bd且tracked源码干净；权重必须59611499字节、SHA129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a。下载中断保留.partial，重新执行会续传，不反复丢弃已下载尾段。已有无效最终权重不覆盖，换--checkpoint新路径或人工确认后保留备份。网络/Range失败仍可能发生，未SHA通过绝不开训。

## 3. 先审计，再后台启动

```bash
export STAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT="$PWD/experiment/all_runs/v1n23c_20e_seed1_$STAMP"
export LOG="/root/autodl-tmp/v1n23c_$STAMP.log"

if bash repro/run_v1n23c_screen_server.sh \
  --collect-only --data-root "$DATA_ROOT" \
  --teacher-repo "$TEACHER_REPO" --teacher-checkpoint "$TEACHER_WEIGHT" \
  --output "$OUTPUT"; then
  nohup bash repro/run_v1n23c_screen_server.sh \
    --collect-only --data-root "$DATA_ROOT" \
    --teacher-repo "$TEACHER_REPO" --teacher-checkpoint "$TEACHER_WEIGHT" \
    --output "$OUTPUT" --run --shutdown-on-success --dedicated-instance \
    > "$LOG" 2>&1 < /dev/null &
  echo "后台PID=$!"
  echo "结果目录=$OUTPUT"
  echo "日志文件=$LOG"
else
  echo "准备或审计失败，未启动训练；请提供报错。"
fi
```

先审计1800个数据文件、教师、源码、GPU占用和可用磁盘；无--run时无优化器，不创建结果目录。启动后顺序：无优化器layout→生产数据流→候选2步smoke及整图0859保存重载→无优化器延迟/显存→候选20轮→逐图/曲线/每轮1000更新/LR时间轴/数据源码完整性→汇总→申请关机。没有V1训练命令。

只在专用实例使用关机参数。需保留实例则去掉--shutdown-on-success和--dedicated-instance。没有GPU空闲/≥16GB、数据缺失、dirty checkout、输出已存在或延迟>1.15x等情况会阻止正式20轮。

## 4. 实时进度及导出文本

```bash
tail -F "$LOG"
```

Ctrl+C只停止查看；nohup独立后台任务不依赖本机或SSH。重新SSH需用实际日志路径，环境变量不会自动恢复。

```bash
cat /root/autodl-tmp/Image-Super-Resolution-V1N23C/experiment/all_runs/实际目录/epoch_comparison.csv
cat /root/autodl-tmp/Image-Super-Resolution-V1N23C/experiment/all_runs/实际目录/summary.json
tail -n 5 /root/autodl-tmp/Image-Super-Resolution-V1N23C/experiment/all_runs/实际目录/n23c/mechanism.jsonl
```

没有压缩或上传动作。也可运行 `python repro/summarize_v1n23c.py --output 实际完整输出路径` 重新打印结果（不加--write就不覆盖已有汇总）。

## 5. 风险与证据边界

- 成功后关机关闭整个实例，不只是当前GPU任务；须自行确认没有其它工作。
- 失败默认不关机，保留日志但会持续计费；nohup不是失败报警，需检查AutoDL控制台/日志。
- 包装脚本按ELF/shebang/无shebang Bash三种方式识别AutoDL shutdown，先检查后执行。平台接口拒绝/权限/网络仍可能使关机失败，须在平台确认停止计费；不能保证所有实例关机都成功。
- 本机仅工程验证，2.9.1+cu128/8GB与目标服务器不同；服务器preflight实际验证不能省略。训练GPU断电/租期结束/系统重启仍会中断。
- 从零20轮是T_max150轨迹的前段，不是训练完成150轮；不得据其宣称长期提升或使用best单点包装成功。
- 校准改变的是接缝而非已证实的创新。PSNR、泛化、完整MAC和顶会新颖性未知。

原549个V1 state、零gate映射、FFN写回顺序与梯度/权重重载由私有检查验证。极小LN缩放更新可能低于FP32半ULP：仅非零梯度、非零Adam moment且明确计算更新小于半ULP时记录rounding；零梯度/优化器漏参/可表示却不更新仍报错。不把日志“全部学起来”替代性能。
