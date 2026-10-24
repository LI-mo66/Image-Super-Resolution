"""Audit returned LFMN CSV/JSON and plot curves without touching source artifacts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

SETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, required=True)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New output directory, must not exist')
    args = p.parse_args()
    blobs = {name: path.read_bytes() for name, path in [('metrics', args.metrics), ('summary', args.summary)]}
    rows = list(csv.DictReader(blobs['metrics'].decode('utf-8-sig').splitlines()))
    summary = json.loads(blobs['summary'].decode('utf-8-sig'))
    assert len(rows) == 1000 and [int(r['epoch']) for r in rows] == list(range(1, 1001))
    assert summary['complete'] and not summary['engineering_smoke'] and summary['final_epoch'] == 1000
    proto = summary['protocol']
    assert proto['backend'] == 'official' and proto['scale'] == 4
    assert proto['pretrained_checkpoint'] is None and not proto['self_ensemble_enabled']
    running, running_epochs, top = {}, {}, -math.inf
    for row in rows:
        epoch = int(row['epoch'])
        assert int(row['global_step']) == epoch*1000
        assert abs(float(row['learning_rate']) - 2e-4 * .5**((epoch-1)//200)) < 1e-15
        assert abs(float(row['next_learning_rate']) - 2e-4 * .5**(epoch//200)) < 1e-15
        assert all(math.isfinite(float(row[k])) for k in row if k not in ('is_best',))
        score = sum(float(row[n+'_psnr']) for n in SETS)/5
        assert abs(score-float(row['selection_psnr'])) < 1e-12
        assert (row['is_best'] == 'True') == (score > top)
        top = max(top, score)
        for name in SETS:
            for metric in ('psnr', 'ssim'):
                key = name+'_'+metric
                value = float(row[key])
                if value > running.get(key, -math.inf):
                    running[key], running_epochs[key] = value, epoch
                assert abs(running[key]-float(row[name+'_best_'+metric])) < 1e-12
                assert running_epochs[key] == int(row[name+'_best_'+metric+'_epoch'])
        assert abs(float(row['epoch_seconds'])-float(row['train_seconds'])-float(row['validation_seconds'])) < 1e-6
    best = max(rows, key=lambda r: float(r['selection_psnr']))
    assert int(best['epoch']) == summary['best_epoch']
    for name in SETS:
        for metric in ('psnr', 'ssim'):
            key = name+'_'+metric
            assert abs(float(best[key])-summary['selected_checkpoint_metrics'][name][metric]) < 1e-12
            assert abs(float(rows[-1][key])-summary['final_metrics'][name][metric]) < 1e-12
            assert abs(running[key]-summary['independent_dataset_bests'][name][metric]) < 1e-12
            assert running_epochs[key] == summary['independent_dataset_bests'][name][metric+'_epoch']
        assert summary['selected_checkpoint_metrics'][name]['count'] == dict(zip(SETS,(5,14,100,100,109)))[name]
    assert abs(float(rows[-1]['elapsed_seconds'])-summary['elapsed_seconds']) < 1e-9
    times = {k: sum(float(r[k]) for r in rows)/3600 for k in ('train_seconds','validation_seconds','epoch_seconds')}
    milestones = [{k: float(r[k]) for k in ('epoch','learning_rate','train_loss','selection_psnr',
                  *[n+'_psnr' for n in SETS])} for r in rows if int(r['epoch']) in
                  (1,2,5,10,20,50,100,150,200,300,400,500,600,700,800,900,984,1000)]
    windows = {}
    for low, high in [(1,200),(201,400),(401,600),(601,800),(801,1000),(701,800),(901,1000),(951,1000)]:
        window = rows[low-1:high]
        windows['%d-%d' % (low,high)] = {key: dict(mean=st.mean(float(r[key]) for r in window),
              stdev=st.stdev(float(r[key]) for r in window), minimum=min(float(r[key]) for r in window),
              maximum=max(float(r[key]) for r in window)) for key in ['selection_psnr','train_loss',*[n+'_psnr' for n in SETS]]}
    output = dict(checks='1000 epochs, steps, LR, finite fields, selection/best histories, summary consistency PASS',
                  sources={key:dict(path=str(path.resolve()),sha256=hashlib.sha256(blobs[key]).hexdigest())
                           for key,path in [('metrics',args.metrics),('summary',args.summary)]},
                  best_epoch=int(best['epoch']), best_mean5=float(best['selection_psnr']),
                  final_mean5=float(rows[-1]['selection_psnr']), timing_hours=times,
                  milestones=milestones,windows=windows,comparison={},best_to_final={})
    for name in SETS:
        b, f = summary['selected_checkpoint_metrics'][name],summary['final_metrics'][name]
        paper = summary['paper_x4'][name]
        output['comparison'][name] = dict(best=b,final=f,paper=paper,psnr_gap=b['psnr']-paper[0],
                                        ssim_gap=b['ssim']-paper[1],independent=summary['independent_dataset_bests'][name])
        output['best_to_final'][name] = f['psnr']-b['psnr']
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    epochs = list(range(1,1001))
    for ax, name in zip(axes.flat, SETS):
        ax.plot(epochs,[float(r[name+'_psnr']) for r in rows],color='#176c9c',lw=1,label='No ensemble')
        ax.axhline(summary['paper_x4'][name][0],color='#b26c22',ls='--',lw=1,label='Paper (TTA unspecified)')
        ax.scatter([int(best['epoch'])],[float(best[name+'_psnr'])],color='#a52a40',s=22,zorder=4,label='Common best epoch')
        for milestone in (200,400,600,800):
            ax.axvline(milestone,color='gray',alpha=.2,lw=.8)
        ax.set(title=name,xlabel='Epoch',ylabel='PSNR (dB)')
        ax.grid(alpha=.18)
    ax = axes.flat[-1]
    ax.plot(epochs,[float(r['selection_psnr']) for r in rows],color='#176c9c',lw=1,label='Mean of five dataset PSNRs')
    ax.axvline(int(best['epoch']),color='#a52a40',ls=':',label='Best epoch %s' % best['epoch'])
    ax.set(title='Fixed selection criterion',xlabel='Epoch',ylabel='Mean PSNR (dB)')
    axes.flat[0].legend(fontsize=8)
    ax.legend(fontsize=8)
    fig.suptitle('Original LFMN x4, scratch, seed 1, 1,000,000 updates',fontsize=14)
    fig.savefig(args.output/'convergence.png',dpi=160)
    plt.close(fig)
    # Late-stage view prevents early gains from obscuring the remaining improvement.
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name in SETS:
        start = float(rows[599][name+'_psnr'])
        axes[0].plot(epochs[599:],[float(r[name+'_psnr'])-start for r in rows[599:]],lw=1,label=name)
    axes[0].set(title='Gain relative to epoch 600',xlabel='Epoch',ylabel='PSNR gain (dB)')
    axes[0].legend(fontsize=8)
    axes[1].plot(epochs[599:],[float(r['selection_psnr']) for r in rows[599:]],lw=.8,color='#176c9c')
    axes[1].axvline(int(best['epoch']),color='#a52a40',ls=':')
    axes[1].set(title='Late-stage selection score',xlabel='Epoch',ylabel='Mean PSNR (dB)')
    for ax in axes:
        ax.grid(alpha=.18)
    fig.savefig(args.output/'late_stage.png',dpi=160)
    plt.close(fig)
    assert all(path.read_bytes()==blobs[key] for key,path in [('metrics',args.metrics),('summary',args.summary)])
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
