# N9 PCSTR 独立 M1 实验卡

## 1. 身份信息
N9 / Prior-Conditioned Soft Token Router；结构候选；状态 VERIFIED（2026-10-07用户授权实现，独立实现与本地验证完成；服务器短筛尚未启动）。
分支 codex/n9-pcstr-core-m1；共同基线 b198de7；复用已登记 N9 实现，不将同一方法重新命名为 N17。实现提交见 Git HEAD。

## 2. 核心假设
原 TAB 使用硬分组及训练更新、测试冻结的 EMA 中心；以 Fs+Fm 生成可微的逐图分配，可能改善精度/效率权衡。瓶颈与收益尚未证实。

## 3. 证据
lfmn.py 的 TAB 与 lfmnpcstr.py；N5 中心细化只有 +0.003380dB、SSIM下降，不能据此保证收益。先验更新、状态记忆已有失败；本次只替换 TAB。N9 已有结构和单batch验证，无正式性能结果。历史详见 N9_TAB替换结构设计评审.md。

## 4. 机制
Fm B×48×H×W，Fs B×32×H×W；LN后投影至16维，GELU后产生 A B×HW×32，沿32 softmax；V只来自Fm，48维。
T=AᵀV/sum_pixels(A) 为 B×32×48；4头注意力 Q/K24、V48；A×T恢复 B×HW×48，原两段残差、ConvFFN保留。FP32质量累加与除法；八阶段不共享，每阶段重新生成；训练/评测同构；仅L1。

## 5. 差异
只替换 EMA/argmax/sort/IASA/IRCA；SFML、LRSA、ESA、decoder及bilinear图像残差保留。既有模型文件原样复用，无新增门控、状态、KD。

## 6. 近邻及风险
CATANet CVPR2025：内容聚合/训练期中心，本方法改为 Fs 条件逐图软路由。
https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CATANet_Efficient_Content-Aware_Token_Aggregation_for_Lightweight_Image_Super-Resolution_CVPR_2025_paper.html
ATD CVPR2024：自适应字典/类别交互；本方法无外部字典，但自适应Token不新。
https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Transcending_the_Limit_of_Local_Window_Advanced_Super-Resolution_Transformer_with_CVPR_2024_paper.html
SPIN ICCV2023：超像素聚合/区域交互；本方法Fs+Fm全图软分配，但有重叠风险。
https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Lightweight_Image_Super-Resolution_with_Superpixel_Token_Interaction_ICCV_2023_paper.html
不可声称首次可微路由、已证明TAB瓶颈、已满足顶会新颖性。旧TokenLearner仅作优先权参考，不是本次近期核心来源。

## 7. 预算
B0 759627，M1 733259（-26368，-3.47%）。路由29.655M MAC/阶段/LR64，不是全网FLOPs；服务器延迟/显存待测，预算均不恶化超过10%。

## 8. 证伪
同epoch精度不超过B0即不支持；有效Token<8或质量比>100为退化信号；软聚合可能平滑细节。机制生效不代表性能成立。

## 9. 协议指纹
baseline_id prior_proxy_scratch_seed1/baseline；产出source_commit未写入原日志，时间/脚本对应bf68bfe（推断，非已证实）；共同代码b198de7对B0的默认路径审计，数据/指标实现一致。
DIV2K 1-800/801-900；x4；HR256、batch4、线程8、seed1、增强开启、ext img、FP32、rgb255；Adam .9/.999 eps1e-8、wd0、clip0；lr2e-4；cosine T_max150 eta_min1e-6；L1；1000batch/epoch；从零训练。
B0完成150epoch；M1先运行前20epoch，epochs只控制终止，T_max显式150、数据repeat由test_every控制，前20epoch时间轴相同。超过20不自动续训。
PSNR：量化RGB裁10；SSIM：量化Y裁4再11x11有效窗；无chop/self-ensemble。
旧环境未完整记录，服务器运行环境写manifest；checkpoint SHA256由入口计算。

## 10. 复用决定
Baseline reuse: YES，条件是配置/状态/源码默认路径审计及复评全100图与历史曲线一致。入口在M1训练前复评固定20轮，末5轮使用旧全精度曲线，偏差超过PSNR 0.0002或SSIM 0.00001则停止，不悄悄重训或继续。复评不覆盖旧B0。来源提交未知仍列为追溯局限；复评一致不能单独证明训练来源。

## 11. 最小矩阵
B0旧权重仅复评；M1=LFMNPCSTR从零；C0等参数mixer与Fm-only仅M1通过后进行。

## 12. 闸门
Gate0：尺寸、参数、train/eval、重载；Gate1：有限非零梯度、Fs/Fm因果、AMP、路由；Gate2：真实batch4 HR256训练一步、0801验证、保存重载。
20轮EXTEND：末5均值>=.01dB、逐图中位>0、差值斜率>=-.001、SSIM>=-.0001；灰区：(0,.01)、斜率>=.001、末5至少4正，只允许一次到40。
末5<=0且斜率<=0或SSIM<-0.0001停止；其他不自动延长。40轮须末5>=.02、CI下界>0、胜率>=60%、高纹理非负且效率预算合格；之后才150、多seed(>=3)、冻结结构后五benchmark。

## 13. 实际服务器协议
未启动；入口 repro/run_n9_m1_server.py；GPU/commit/数据hash/旧权重hash/环境由manifest记录。无自动关机。

## 14. 结果
性能结果未知；本地检查和烟雾报告写独立输出，不作为精度证据。

## 15. 决策
等待服务器短筛；不得保证提升。实际状态以追加验证记录为准。

## 2026-10-07 验证追加
已通过参数/尺寸/Fs与Fm因果/模块梯度/train-eval/AMP/严格重载；真实DIV2K batch4 HR256训练一步、0801整图验证及保存已完成（n9_local_verify_20261007_fp32fix），不作为精度证据。旧bf68bfe与当前B0在CPU eval输出逐元素相同、train输出及梯度匹配，数据/loader/L1源码相同。CUDA训练scatter累加可能令硬聚类对照产生微小差异，故源码等价测试使用CPU；这不证明旧服务器未提交过修改。
修复PCSTR autocast下bmm显式float仍降精度的问题：聚合区显式关闭autocast，FP32正式路径不变，参数不变。汇总回归测试通过正收益/零收益/调度不匹配/逐图曲线不匹配案例。
最终检查目录 n9_local_verify_20261007_final：全模型所有参数在真实batch中梯度有限且非零，最小梯度绝对和约1.52e-8。旧/当前quantize、PSNR、SSIM AST一致，旧150轮与新20轮/T_max150的前20次Adam/cosine学习率一致。
诊断脚本已在本地旧B0与单步PCSTR权重上通过链路；此计时仅工程检查，不是已训练M1的性能结论。正式全100图B0回放及20轮M1由新服务器完成。
