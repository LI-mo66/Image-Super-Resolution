"""Write the research review from an already audited JSON; no training or reevaluation."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args = p.parse_args()
    a = json.loads(args.audit.read_text(encoding='utf-8'))
    names = ('Set5','Set14','B100','Urban100','Manga109')
    lines = ['# 原始 LFMN ×4 从零训练1000轮结果审查', '', '审查日期：2026-10-24。', '',
             '完整训练和收敛记录核验通过，但无自集成成绩尚未达到论文Table II。'
             '统一选模第984轮，不能用各数据集独立最佳拼成一组模型成绩。', '',
             '## 1. 来源与核验', '']
    for key in ('metrics','summary'):
        source = a['sources'][key]
        lines += ['- `%s`：`%s`；SHA256 `%s`。' % (key,source['path'],source['sha256'])]
    lines += ['', '1000行，Epoch1–1000连续无重复；global_step=Epoch×1000；训练LR与下一轮LR、'
              '五数据集等权均值、is_best标记、逐数据集PSNR/SSIM历史最佳值和首次Epoch全部核验通过；'
              'summary中统一best、final、独立best与CSV一致。原始输入未改写。', '',
              '元数据登记官方Linux/Python3.9编译LFMN，8阶段，×4，DIV2K1–800，seed1，'
              'Batch8/HRpatch256，FP32 L1/Adam LR2e-4，200轮减半，100万次更新；'
              '无预训练、无自集成/chop/AMP/TF32/权重EMA。PSNR：量化SR、官方Y系数/256、裁边4；'
              'SSIM：量化、BT.601 Y、Gaussian11 sigma1.5、裁边4。', '',
              '这些是返回记录及历史服务器输出支持的结论，不是独立重跑。当前未收到checkpoint、'
              '权重SHA256、config.json、完整stdout及数据清单，不能独立验证实际权重内容、'
              '数据逐图版本或真实墙钟起止；不得据此宣布与作者所有未披露配置完全一致。', '',
              '## 2. 统一best与最终结果', '',
              '|数据集|984轮PSNR/SSIM|1000轮PSNR/SSIM|论文PSNR/SSIM|best−论文PSNR|',
              '|---|---|---|---|---|']
    for n in names:
        c = a['comparison'][n]
        b,f,paper = c['best'],c['final'],c['paper']
        lines.append('|%s|%.6f / %.6f|%.6f / %.6f|%.2f / %.4f|%+.6f|' %
                     (n,b['psnr'],b['ssim'],f['psnr'],f['ssim'],paper[0],paper[1],c['psnr_gap']))
    lines += ['', '统一best五集等权PSNR均值 %.8f dB；final %.8f dB，差 %.8f dB。'
              '最大best→final回落来自Manga109，约0.04006 dB；其余四集约0.00117–0.00911 dB。'
              '这是选模指标的轻微末轮回落，不足以证明整体过拟合或训练崩溃。' %
              (a['best_mean5'],a['final_mean5'],a['final_mean5']-a['best_mean5']), '',
              '论文值来自本项目此前核对的本地论文Table II，亦保存在原summary.paper_x4；'
              '论文自集成状态与实际Batch/总更新数未明确披露，不能把差值全部归因于训练实现。', '',
              '## 3. 固定阶段PSNR', '',
              '以下每行均取该Epoch的真实指标，不取此前历史最佳。单位dB。', '',
              '|Epoch|训练LR|Set5|Set14|B100|Urban100|Manga109|',
              '|---|---|---|---|---|---|---|']
    for r in a['milestones']:
        lines.append('|%d|%.5g|%s|' % (r['epoch'],r['learning_rate'],
                     '|'.join('%.5f' % r[n+'_psnr'] for n in names)))
    lines += ['', '## 4. 收敛判断', '',
              '- 1–200轮提升最快；之后随学习率降低，收益逐步减小，曲线存在偶发回落后恢复。'
              '没有非有限指标、持续发散或大幅长期下降的日志证据。',
              '- 200→400轮Urban100/Manga109分别提升约0.21454/0.22213 dB；'
              '800→1000轮仅约0.02121/0.02832 dB。',
              '- 901–1000轮均值对比701–800轮，五数据集提升依次约'
              '0.01204、0.00760、0.00628、0.02330、0.02847 dB。后期趋于平台，仍有小幅改善；'
              '不能断言绝对收敛，也不能推导延长训练必定补齐论文差距。',
              '- 末100轮五集均值的Epoch间标准差约0.00587 dB，只描述本次轨迹波动，'
              '不是多seed置信区间；不要把相邻Epoch当作独立实验样本。', '',
              '## 5. 独立PSNR最佳（仅诊断）', '',
              '|数据集|独立最高PSNR|Epoch|统一best落后该值|', '|---|---|---|---|']
    for n in names:
        c = a['comparison'][n]
        i = c['independent']
        lines.append('|%s|%.8f|%d|%.8f|' %
                     (n,i['psnr'],i['psnr_epoch'],i['psnr']-c['best']['psnr']))
    lines += ['', '即使看各集独立最高值也没有达到论文值。统一选模带来的差额只有'
              '0–0.01509 dB，不能解释目前主要差距。独立PSNR最高Epoch的SSIM必须从CSV同一行读取，'
              'summary独立best里的SSIM可能属于另一个Epoch，不能配成同次测量。', '',
              '## 6. 时间与基线用途', '',
              '记录的训练 %.3f 小时，验证 %.3f 小时，合计 %.3f 小时（约4.22天）；'
              '验证约占25.03%%。这是训练/验证累计耗时，不含首次数据检查、保存/绘图与暂停时间，'
              '不能当作完整计费时长。' % (a['timing_hours']['train_seconds'],
              a['timing_hours']['validation_seconds'],a['timing_hours']['epoch_seconds']), '',
              '可以登记为“本项目官方模型、seed1、无自集成完整训练基线”，不标为“论文数值完全复现”。'
              '后续候选只可在同数据/训练/评测/选模协议下比较，优先报告统一best和固定1000轮两种口径。'
              '若新候选依赖本地Python重实现，应先证明其与本次官方编译模型的数值及训练行为等价，'
              '否则必须使用同一Python骨架另训控制组，不能默认本次B0可直接复用。', '',
              '定位差距时优先核查同一best权重的无自集成逐图复评、官方发布权重在同一评测代码下的结果、'
              '数据退化与作者实际训练命令/迭代预算。自集成只能作为另外登记的诊断口径，'
              '不得反向替换本次no-x8主结果。当前不修改模型，不启动额外训练，不宣称差距来自某个结构机制。', '']
    with args.report.open('x',encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
