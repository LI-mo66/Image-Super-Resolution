# 科研创新候选统一开发、基线复用与服务器实验 SOP

## 0. 适用范围与强制性

本文件不是RGCRD、Token、蒸馏或任何单一创新点的实施说明，而是本项目今后**所有创新候选**必须遵守的统一生命周期规范。具体方法的结构、公式、超参数和消融只写入该候选自己的实验卡，不写入公共流程。

无论候选属于网络结构、训练框架、损失函数、数据策略、推理机制、效率优化或它们的组合，都必须经过相同状态机：

```text
PROPOSED
  ↓ 问题与证据齐全
AUDITED
  ↓ 新颖性、基线和资源获批
APPROVED
  ↓ 独立实现完成
IMPLEMENTED
  ↓ 本地验证全部通过
VERIFIED
  ↓ 服务器短筛启动
SCREENING
  ↓ 按预注册闸门判定
DECIDED
  ├── PROMOTED：进入长训、消融和论文验证
  └── ARCHIVED：记录负结果并关闭分支
```

禁止跳级。例如：没有实验卡不得编码；没有本地验证不得上服务器；没有B0/M1短筛不得直接1000 epoch；主候选未通过不得先跑完整消融。

### 0.1 各状态的必需交付物

| 状态 | 必需交付物 | 允许进入下一状态的条件 |
|---|---|---|
| PROPOSED | 一句话假设、问题位置、预期收益 | 问题可从源码或实验观察定位 |
| AUDITED | 失败证据、近邻工作、反例、风险 | 不是已失败假设的换名重复，也不是显然模块堆叠 |
| APPROVED | 完整实验卡、协议指纹、B0复用决定、预算 | 主对照和停止线已预注册 |
| IMPLEMENTED | 独立分支与独立入口文件 | 默认路径不改变共同基线 |
| VERIFIED | 结构、数值、梯度、真实batch、保存重载报告 | 所有检查通过且无未解释NaN/失活 |
| SCREENING | 服务器commit、命令、日志、完整精度指标 | B0与M1处于同一协议 |
| DECIDED | 自动汇总、机制诊断、GO/NO-GO | 结论遵守预注册规则 |
| PROMOTED | 长训、消融、多seed、benchmark计划 | 短筛存在稳定正向证据 |
| ARCHIVED | 负结果、失败原因、关闭声明 | 不再在原分支追加补丁式模块 |

### 0.2 每个候选的标准资产包

所有候选使用占位符`<ID>`和`<slug>`形成同构资产：

```text
branch: codex/<ID>-<slug>
method: LFMN/model/lfmn_<ID>.py 或 LFMN/<ID>/
check:  repro/check_<ID>.py
run:    repro/run_<ID>_screen_server.sh
report: repro/summarize_<ID>_screen.py
card:   创新方案/<ID>_实验卡.md
record: 创新思路与实验记录.md 中的独立小节
```

候选之间只能共享经过登记的共同基础设施与合法复用的baseline，不能共享未经证明的候选机制。

## 1. 目的

本流程用于约束从“提出创新假设”到“服务器实验、结果判定、Git归档”的全过程，目标是：

1. 每个创新可以被独立实现、验证、停止和回滚；
2. 不让不同候选互相污染，避免在失败模块上继续叠加；
3. 在实验协议完全一致时复用已有基线，避免重复训练；
4. 在协议不一致时禁止伪复用，避免把训练差异误写成算法收益；
5. 所有结论都能追溯到源码提交、服务器命令、日志和完整精度指标；
6. 优先研究框架级矛盾，不以简单堆卷积、注意力、门控或频域模块代替创新。

本SOP适用于Codex、其他AI代理和人工开发者。任何AI开始任意新候选前，都应先阅读本文件和根目录`AGENTS.md`。方法名称不得改变流程要求。

---

## 2. 核心原则

### 2.1 一个创新，一个隔离分支

每个新候选从约定的共同基线提交创建新分支，不从另一个尚未证明有效的候选分支继续叠加。

推荐命名：

```text
codex/n10-short-name
codex/n11-short-name
codex/rgcrd-v2
```

若团队已有明确分支命名规则，可替换`codex/`前缀，但必须保证一个候选对应一个分支。

禁止：

```text
N10建立在失败N9代码上，再叠加N8分支
多个候选共用一个不断修改的model文件
服务器直接修改源码但不提交
```

### 2.2 一个创新，一组独立入口文件

候选不得直接覆盖`LFMN/model/lfmn.py`。建议文件结构：

```text
LFMN/model/lfmn_n10.py
LFMN/loss/n10.py                    # 只有需要独立损失时创建
LFMN/n10/                           # 多文件机制使用独立包
repro/check_n10.py                  # 静态、前向、梯度、保存重载检查
repro/run_n10_screen_server.sh      # 服务器短筛入口
repro/summarize_n10_screen.py       # 自动比较与止损
创新方案/N10_方案与实验卡.md          # 或在根目录使用唯一文件名
```

共同训练基础设施确需修改时，应单独提交，并证明：

- 默认关闭时原LFMN行为不变；
- 不改变其他候选；
- 基线与候选都使用相同修复；
- 修改属于工程修复还是算法创新已经明确区分。

### 2.3 原LFMN是控制组，不是候选

统一记号：

```text
B0：相同协议下重新训练或合法复用的原始LFMN
M1：当前完整创新候选
C1/M0：只在M1通过B0后补充的必要对照或消融
```

主假设的第一胜负线始终是：

$$
\Delta(e)=Metric_{M1}(e)-Metric_{B0}(e).
$$

先判断`M1 > B0`。若M1不成立，不应先花大量算力跑完整消融。若M1成立，再用C1/M0回答收益来自哪里。

### 2.4 框架创新优先于模块缝合

提出候选前必须回答：

1. LFMN存在什么可观测、可定位的结构或优化矛盾？
2. 该矛盾属于单层算子不足，还是跨阶段状态、训练监督、信息路由或训练—推理不一致？
3. 新方法是否用一个统一对象贯穿多个阶段，而不是并列添加多个模块？
4. 如果去掉论文名称，方法能否用一条因果链解释？
5. 哪个对照可以否定“只是参数更多、计算更多或普通KD”的解释？

模块缝合常见形式：

```text
卷积 + 注意力 + 门控 + 高频分支
```

框架级候选应更接近：

```text
发现跨阶段关系/状态缺陷
→ 定义贯穿全网络的状态或监督对象
→ 统一改变八阶段的信息演化或学习方式
→ 给出对应的可证伪预测
```

框架级叙事不能替代实验。若统一机制不超过B0，仍然判失败。

---

## 3. 从问题到创新方案的逻辑流程

### 3.1 先做失败证据审计

新方案不得只从“某论文模块效果好”出发。先汇总：

- 已失败候选及其提交号；
- 最终、最佳、末3/5轮差值；
- 是否存在梯度、门控、路由或模块失活；
- 是否出现训练损失改善但PSNR不改善；
- 失败说明哪个假设族不值得继续。

例如，多次修改SFML调制而没有收益，则下一候选不能只换一个新门控继续修改同一位置。

### 3.2 定义唯一核心矛盾

创新卡必须用一句话写出：

```text
由于________，原LFMN在________上受到限制；本候选通过________改变这一过程。
```

不允许同时写三个互不依赖的瓶颈，再分别添加三个模块。

### 3.3 建立机制链

建议使用以下结构：

```text
源码事实
→ 可测缺陷
→ 数学对象
→ 干预机制
→ 可观测中间量
→ PSNR/SSIM预测
→ 失败表现
```

每一步都必须能被代码或实验检查。

### 3.4 做最近邻工作检索

至少记录：

- 最接近的3–5项工作；
- 它们改变的对象；
- 本候选与它们的实质差异；
- 是否只是重新命名已有机制；
- 即使性能成立，新颖性是否足以支撑论文。

没有完成最近邻检索时，只能称为“候选”，不能宣称原创。

### 3.5 预注册最小对照

第一阶段只运行回答主问题所需的最少组：

```text
B0：原LFMN
M1：完整候选
```

M1通过后再补：

```text
C1：最强替代解释，例如普通KD或等参数模块
M0：去掉候选关键机制
```

提前写出：

- 通过线；
- 灰区；
- 停止线；
- 何时允许延长训练；
- 哪些结果绝不能通过挑选最佳epoch解释。

---

## 4. 基线协议指纹与复用规则

### 4.1 基线协议指纹

每个B0都必须记录以下字段：

```yaml
baseline_id:
source_commit:
model: LFMN
pretrained_checkpoint:
checkpoint_sha256:
data_train:
data_range_train:
data_range_validation:
scale:
patch_size_hr:
batch_size:
seed:
augmentation:
optimizer:
learning_rate:
scheduler:
scheduler_horizon_or_milestones:
eta_min_or_gamma:
epochs:
steps_per_epoch:
loss:
rgb_range:
evaluation_datasets:
metric_code_commit:
metric_protocol:
torch_cuda_environment:
result_directory:
psnr_log:
ssim_log:
per_image_metrics:
status:
```

这些字段共同构成“协议指纹”。文件路径相同不等于协议相同。

### 4.2 可以复用基线的情况

只有以下条件同时成立才允许复用：

1. 原LFMN源码和影响其前向/训练的共同代码相同；
2. 数据集、训练/验证范围、倍率相同；
3. HR patch、batch、epoch、每epoch步数相同；
4. seed、shuffle、裁剪和数据增强随机流相同；
5. 优化器、学习率、调度器及其完整时间轴相同；
6. L1等基线损失相同；
7. PSNR/SSIM、颜色空间、边界裁剪和量化口径相同；
8. 复用曲线包含候选需要比较的相同epoch；
9. 原日志、`.pt`完整精度曲线和逐图指标仍可读取；
10. 没有影响公平性的已知实现缺陷。

典型可复用情况：候选仅增加训练期辅助损失，B0协议与已有B0完全一致。

### 4.3 禁止复用基线的情况

以下任一字段改变，默认不得复用：

- 从Cosine改为MultiStep；
- `T_max`、milestone或学习率不同；
- batch或patch不同；
- 训练/验证图片范围不同；
- seed或数据增强不同；
- 从预训练微调改为从零训练；
- 训练轮数或每epoch步数不同；
- 修复了会影响原LFMN输出、梯度或数据顺序的代码；
- PSNR/SSIM实现或边界裁剪口径不同；
- 只有日志三位小数，没有完整精度数据，而当前判断需要千分之一dB级差异。

### 4.4 基线复用决策

开始服务器实验前必须输出：

```text
Baseline reuse: YES/NO
Baseline ID:
Compared fields:
Known differences:
Decision owner:
```

若为`YES`，候选运行脚本不得重新训练B0；应通过参数指定已有B0目录，或允许在同一screen目录追加候选组。

若为`NO`，必须说明哪个协议字段变化，不能只写“为了公平重新跑”。

---

## 5. 分支与文件工作流

### 5.1 创建候选分支

首先确定共同基线提交：

```bash
git switch feature/prior-update-screen
git pull --ff-only origin feature/prior-update-screen
git log -1 --oneline
```

然后创建独立分支：

```bash
git switch -c codex/n10-short-name
```

不得从另一个未合并候选的工作树直接创建，除非明确说明继承关系并把组合实验作为新候选。

### 5.2 最小修改范围

优先新增文件，不覆盖基线：

```text
新增：LFMN/model/lfmn_n10.py
新增：repro/check_n10.py
新增：repro/run_n10_screen_server.sh
新增：repro/summarize_n10_screen.py
新增：创新方案/N10_方案与实验卡.md
```

对`trainer.py`、`option.py`等公共文件的修改必须：

- 默认关闭；
- 不改变旧命令行为；
- 有独立检查；
- 单独解释为什么无法放入候选私有文件。

### 5.3 提交前保护用户文件

检查：

```bash
git status --short
git diff --check
```

只显式暂存本候选文件：

```bash
git add -- \
  LFMN/model/lfmn_n10.py \
  repro/check_n10.py \
  repro/run_n10_screen_server.sh \
  repro/summarize_n10_screen.py \
  创新方案/N10_方案与实验卡.md \
  创新思路与实验记录.md
```

再次检查：

```bash
git diff --cached --check
git diff --cached --stat
git status --short
```

不得暂存`.vscode/`、`.idea/`、数据集、实验输出、checkpoint、缓存、密钥或用户压缩包。

### 5.4 提交和推送

```bash
git commit -m "feat: add N10 short description"
git push -u origin codex/n10-short-name
```

最终答复必须报告：

```text
branch:
commit:
push status:
files changed:
tests passed:
unverified claims:
```

---

## 6. 实现后的本地验证门槛

### Gate 0：结构不变量

至少检查：

- 参数量；
- 新增参数列表；
- 关闭候选时是否恢复基线输出；
- 训练态与评测态返回协议；
- 推理是否包含训练期模块；
- checkpoint严格保存和重载；
- baseline权重能否按预期加载；
- 候选是否意外改变原模型键名。

### Gate 1：数值与梯度

使用合成输入和至少一个真实DIV2K batch检查：

- 所有输出、损失、梯度有限；
- 新增参数获得非零梯度；
- 辅助损失确实影响目标层；
- 门控、路由、Token或关系矩阵不塌缩；
- 辅助梯度/L1梯度范数比；
- 梯度余弦是否持续强冲突；
- AMP、FP32和目标服务器环境差异。

### Gate 2：端到端烟雾测试

必须完成：

```text
真实数据单batch训练
→ 验证集至少1张图
→ 保存checkpoint与指标
→ 重载checkpoint
→ 再次推理
```

烟雾测试数值不作为性能结果。

### Gate 3：短筛

先运行B0与M1。常见检查点：

```text
epoch 1/5/10/20/40
最终差值
末5轮平均差值
相对差值斜率
逐图胜率
bootstrap置信区间
SSIM
机制中间量
```

### Gate 4：正式验证

短筛通过后才允许：

- 150/200 epoch确认；
- 多seed；
- 五benchmark；
- 完整1000 epoch；
- 消融与等计算量对照；
- 统一设备上的参数、FLOPs、显存和延迟测试。

不得因为短筛失败而直接用1000 epoch“赌后期反转”。

---

## 7. 服务器拉取与训练流程

### 7.1 首次拉取候选分支

```bash
cd /root/autodl-tmp/Image-Super-Resolution-N9

git fetch origin
git switch --track origin/codex/n10-short-name
git log -1 --oneline
```

已有本地分支：

```bash
git switch codex/n10-short-name
git pull --ff-only origin codex/n10-short-name
git log -1 --oneline
```

服务器必须核对commit，不能只核对文件名。

### 7.2 运行服务器前检查

```bash
python repro/check_n10.py
nvidia-smi
git status --short
```

若检查脚本失败，不得开始付费长训。

### 7.3 后台运行

```bash
nohup env \
GPU=0 \
EPOCHS=40 \
GROUPS="m1" \
bash repro/run_n10_screen_server.sh \
n10/screen_40e_seed1 \
> n10_screen_launcher.log 2>&1 < /dev/null &

echo $!
```

`nohup`日志、组内`console.log`和框架生成的`log.txt/.pt`都必须保留。

### 7.4 实时监控

```bash
tail -F n10_screen_launcher.log
watch -n 2 nvidia-smi
pgrep -af "run_n10|python main.py"
```

按`Ctrl+C`只停止查看，不停止后台训练。

### 7.5 自动关机

AutoDL可在包装脚本末尾使用：

```bash
/usr/bin/shutdown
```

包装脚本必须先：

1. 保存退出状态；
2. `sync`日志与指标；
3. 写入完成时间和commit；
4. 再执行关机。

自动关机会关闭整个实例。确认没有其他并行实验后才能启用。

---

## 8. 结果分析与停止决策

### 8.1 只比较同epoch、同协议

禁止：

```text
候选epoch 40 对比 baseline epoch 20
MultiStep候选对比Cosine baseline
seed 1候选对比未知seed baseline
候选最佳epoch对比baseline最终epoch
```

至少报告：

```text
final delta
last-5 mean delta
positive-epoch ratio
paired per-image mean/median/win-rate
95% bootstrap CI
SSIM delta
```

### 8.2 结果解释优先级

1. M1是否稳定超过B0；
2. 中间机制是否按假设工作；
3. 收益是否来自目标数据类型；
4. 是否优于最简单替代解释；
5. 是否值得延长训练；
6. 最后才讨论论文叙事。

### 8.3 失败后不得原地堆叠

若M1失败：

- 记录结果与失败原因；
- 关闭该假设；
- 不在同一分支继续加入第二、第三个模块；
- 新假设从共同基线新建分支；
- 若要组合两个已分别成立的候选，必须创建新的组合分支和新实验卡。

---

## 9. 实验结果与Git归档

实验输出、模型权重、数据集和缓存不提交Git。应提交：

- 方法代码；
- 检查脚本；
- 运行脚本；
- 汇总脚本；
- 实验卡；
- `创新思路与实验记录.md`中的结果摘要。

结果摘要必须注明：

```text
branch和commit
服务器设备
数据集和范围
倍率
训练协议
评测口径
baseline来源及是否复用
结果目录
最终/最佳/末N轮
逐图统计
未知字段
Go/No-Go结论
```

实验结束后在开发机：

```bash
git switch codex/n10-short-name
git pull --ff-only
# 只更新实验记录，不复制服务器输出到Git
git add -- 创新方案/N10_方案与实验卡.md 创新思路与实验记录.md
git commit -m "docs: record N10 screening result"
git push
```

---

## 10. 交给其他AI时的固定提示

可将下面文字直接交给其他AI：

```text
请先完整阅读仓库根目录AGENTS.md、创新点开发与服务器实验SOP.md和创新点实验卡模板.md。
本次候选必须从指定共同基线提交创建独立分支，不得覆盖共同基线的核心实现文件，不得继承其他未证明有效的候选代码。先完成失败证据审计、最近邻检索、协议指纹和B0复用判断，再按<ID>创建独立实现、检查、运行、汇总和实验卡文件。第一阶段只运行B0与完整候选M1；协议完全一致时必须复用已有B0，不得重复训练。所有性能主张必须来自机器可读完整精度指标和逐样本结果，未验证结果只能表述为假设。修改后完成结构、梯度、真实单batch、评测、保存重载检查，显式暂存相关文件，提交并推送，报告branch、commit、测试和服务器命令。
```

---

## 11. 每个候选的完成定义

一个候选只有满足以下条件才算完成一轮：

- [ ] 独立分支；
- [ ] 独立方案/实验卡；
- [ ] 明确的框架级问题或合理的模块级定位；
- [ ] 最近邻与新颖性风险；
- [ ] 基线协议指纹及复用决定；
- [ ] 独立实现文件；
- [ ] 自动检查脚本；
- [ ] 服务器运行脚本；
- [ ] 自动汇总与止损；
- [ ] 本地验证通过；
- [ ] Git提交并推送；
- [ ] 服务器commit核对；
- [ ] 完整精度结果；
- [ ] Go/No-Go结论；
- [ ] 实验记录提交；
- [ ] 失败候选没有继续原地堆叠。
