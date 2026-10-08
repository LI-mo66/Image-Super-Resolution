# V1＋N21/N22/N23独立短筛交付

分支 `codex/v1-combos-server-screen` 是交付容器，不把三个机制叠在一个模型中。共同V1为982463c。代码、梯度和烟雾检查只能证明已测试路径可运行；不能保证涨PSNR、无任何错误或150轮收益。

## 三个方案各改什么

| 组合 | 在V1中的接入点 | 新增信息/响应 | 保留部分 | 参数 |
|---|---|---|---|---:|
| V1N21 | 第2/3/4、6/7/8阶段TAB全局响应 | 当前特征刷新父级原型，48→8→48映射与2个child code生成子原型；融合新旧全局响应的差量 | 原局部SDPA、分组排序、原全局分支、EMA和SRPR | 847089，+5526 |
| V1N22 | 各阶段SFML的中间12通道 | 静态LR的4先验+最近SRPR observation的绝对/相对强度，6→12的1x1投影 | 原SFML四层、原TAB/LRSA、8次SRPR及解码 | 842243，+680 |
| V1N23 | 第4/8阶段LRSA后并联修正 | 32/64窗SCC的空间/通道响应，不再八层替换 | 全部原LRSA、TAB、SFML、SRPR及解码 | 854507，+12944 |

N21使用`原响应 + 0.25*tanh(g)*(新global−原global)`，只改全局响应而不重写局部路由；父中心每图清空、与SRPR state分离、跨阶段传递时detach。零gain起点映射与V1一致，但原型传递未保证能改善细节，且两次额外global-response有实际开销。

N22使用`u*(1+0.25*tanh(g)*tanh(P(context)))`，再走原expand_a/b。最初阶段无observation，动态输入为零；后续采用上阶段既有Dᵀr的LR尺度观测，不额外解码、不输入HR/teacher特征。绝对观测经rgb255缩放较小，保留物理幅度并加入空间相对强度。静态先验含DCT低频/DC，不称为高频需求；观测也不是真实HR误差或可靠性。不通过随意加池化/残差保证PSNR。

N23使用`原LRSA输出 + 0.1*tanh(g)*SCC修正`。新支路pad→原LN→DFE投影→按真实坐标切窗→SCC→精确coverage还原→crop，无重复输入残差或FFN。仅新支路用精确coverage，原V1 legacy路径完全不动。这是N23-derived残差适配，不是原八层替换的等价组合，也不再减少参数；新增上下文可能冗余。

共同接口：LR B3HW→学生B3(4H)(4W)，内部B48HW，SRPR state B48HW、q B3HW；原八阶段、stage_probe、D/Dᵀ、写回和最终PixelShuffle均不变。C1仍为官方冻结SwinIR-M x4训练期输出KD0.1+L1；推理没有教师。

## 已验证与尚未验证

549个共同state键、共同初始化/RNG、B2 33×47等非方形/窗口不整除测试、零gain/关闭恢复V1均通过；真实DIV2K LR64/HR256+官方教师两步检查通过，第二步新增36/24/54参数张量均启动。实际framework另完成两步batch4、完整0859验证、保存日志/权重、strict加载与同设备raw输出hash一致。测试用0worker，仅工程，不作精度比较。

本机RTX4060 Laptop/PyTorch2.9.1，framework烟雾peak allocated约6.53GB。随机权重eval、LR128²、FP32、2次warmup/5次计时：N21/N22/N23延迟比约1.111/1.045/1.062，参数增加约0.66%/0.08%/1.54%。不是服务器全图或训练后benchmark；MAC/FLOPs未知。服务器会独立复测，不能据小参数保证快。

## 协议与基线

实际短筛：三个组分别从零20轮，Cosine T_max150/eta_min1e-6，LR2e-4、Adam默认betas(.9,.999)/eps1e-8、batch4/HR256/seed1、8worker、每轮1000步、DIV2K1–800/801–900、FP32、无chop/x8。20轮对应旧V1 Cosine150的前20轮，不是T_max20。

用户提供V1的20→40→150三段config，初始scratch没有resume_data_epochs属旧字段缺省0；续训必须显式20/40，按阶段解析而非误判重复。既有V1不重训、不从其150轮权重微调。完整曲线、前20轮逐图和model_20.pt存在时可输出同epoch参考差；历史data/environment/source/update-stream尚未完整审计，所有参考强制REFERENCE_ONLY，不给正式GO。不把三位小数日志当完整精度文件。

## 服务器操作

```bash
cd /root/autodl-tmp
git clone --branch codex/v1-combos-server-screen \
  https://github.com/LI-mo66/Image-Super-Resolution.git Image-Super-Resolution-V1Combos
cd Image-Super-Resolution-V1Combos
git log -1 --oneline
git status --short
```

数据/teacher/checkpoint不在Git，需要现有数据盘目录。优先用已有教师，设TEACHER_REPO和TEACHER_WEIGHT到其绝对路径。若本实例确无教师才运行`bash repro/setup_rgcrd_teacher_server.sh`（会安装requirements-server依赖及下载官方教师，不装PyTorch；依赖变化写入环境，旧V1可比性仍待审计）。不要修改CUDA/PyTorch来绕过错误。

```bash
export DATA_ROOT=/root/autodl-tmp/datasets
# 改成你实际已有教师路径；若刚在新checkout准备，则用这里的默认位置
export TEACHER_REPO="$PWD/repro/swinir_ref"
export TEACHER_WEIGHT="$PWD/repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"
export STAMP=$(date +%Y%m%d_%H%M%S)
export OUTPUT="$PWD/experiment/all_runs/v1_combos_20e_seed1_$STAMP"
export LOG="/root/autodl-tmp/v1_combos_$STAMP.log"

# 若旧V1完整文件还没在本机，明确只收集三候选；绝不会重训V1
bash repro/run_v1_combos_screen_server.sh \
  --collect-only --data-root "$DATA_ROOT" \
  --teacher-repo "$TEACHER_REPO" --teacher-checkpoint "$TEACHER_WEIGHT" \
  --output "$OUTPUT"
# 以上只有审计，没有优化器；通过后：
nohup bash repro/run_v1_combos_screen_server.sh \
  --collect-only --data-root "$DATA_ROOT" \
  --teacher-repo "$TEACHER_REPO" --teacher-checkpoint "$TEACHER_WEIGHT" \
  --output "$OUTPUT" --run \
  > "$LOG" 2>&1 < /dev/null &
echo "PID=$! OUTPUT=$OUTPUT LOG=$LOG"
tail -F "$LOG"
```

旧V1存在时把两处`--collect-only`改成`--baseline "$V1"`：

```bash
export V1=/root/autodl-tmp/Image-Super-Resolution/experiment/all_runs/n16/srpr_c1_150e_seed1_6a68328_r2/srpr_c1
test -f "$V1/config.txt"
test -f "$V1/model/model_20.pt"
test -f "$V1/per_image_metrics/epoch_0020.pt"
test -f "$V1/psnr_log.pt"
```

可用`--groups n21`等只收集一组。源码clean、1800文件hash、官方教师commit/SHA、GPU空闲/≥16GB、输出新路径和≥5GB空盘均先审计。付费训练前先核对三组生产8worker前两batch相同，三组各两步framework烟雾和冻结效率测试；全通过后串行20轮。任何错误停止，不静默删数据、改源码、跳过检查或改协议。

## 后台、关机、结果

nohup能抵抗SSH断开/本机关机，不能抵抗服务器重启、被回收、OOM或程序错误。Ctrl+C退出tail不停止后台训练。自动关机只在专用实例时给启动命令额外加`--shutdown-on-success --dedicated-instance`；这会关闭整个实例。失败保留现场不自动关机，可能继续计费；请检查AutoDL控制台。兼容无shebang的AutoDL shutdown文本helper，不把它当ELF直接执行。

```bash
tail -F "$LOG"
tail -F "$OUTPUT/n21/log.txt"  # 文件在组开始后出现
tail -F "$OUTPUT/n21/mechanism.jsonl"
watch -n 2 nvidia-smi

# 完成后直接输出，无需打包；读取已有结果不进行训练
python repro/summarize_v1_combos.py --output "$OUTPUT"
# 有历史V1完整文件，打印参考对比：
python repro/summarize_v1_combos.py --output "$OUTPUT" --baseline "$V1"
```

每组独立权重、完整曲线、20轮逐图、机制/更新数/第一batch哈希/strict重载信息；主目录manifest/summary.json/epoch_comparison.csv。比较的是组合−V1，不拿旧B0替代，也不挑best说明成功。没有匹配基线证据的组只能PENDING，不自动长训。安全关机只在所有指定组和完整性检查成功、sync保存完成后发生。
