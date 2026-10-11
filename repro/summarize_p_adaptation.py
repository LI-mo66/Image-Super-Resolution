"""Strict same-budget P0/P3/P4 adaptation comparison; single-seed investment gate."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

SCHEMES = {'P0', 'P3', 'P4'}
EPOCHS = list(range(21))
VALID_IDS = {f'{i:04d}' for i in range(811,831)}

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def average(values):
    return statistics.mean(values) if values else None

def paired_stats(pairs):
    values = [value for _,value in pairs]
    if not values: raise ValueError('Empty comparison')
    rng=random.Random(20261011)
    samples=sorted(average(rng.choices(values,k=len(values))) for _ in range(5000))
    index=max(range(len(values)),key=lambda i: abs(values[i]))
    return {'images':len(values),'mean':average(values),'median':statistics.median(values),
            'win_rate':sum(v>0 for v in values)/len(values),'zero_rate':sum(v==0 for v in values)/len(values),
            'bootstrap95_ci':[samples[124],samples[4874]],'bootstrap_seed':20261011,'bootstrap_replicates':5000,
            'max_influence_image':pairs[index][0],'max_influence_delta':values[index],
            'leave_one_out_mean':average([v for i,v in enumerate(values) if i!=index]),
            'unit':'image; no cross-training-seed stability claim'}

def row_key(row):
    return (row['split'],row['dataset'],str(row['image']))

def comparable(fp):
    # Only the declared scientific differences are excluded, never data/budget/source fields.
    return {key:value for key,value in fp.items() if key not in {'scheme','loss','loss_name','candidate_parameter_count','new_parameter_count'}}

def load_run(path):
    config=read_json(Path(path)/'config.json')
    if config.get('status')!='completed' or config.get('exit_code')!=0:
        raise ValueError(f'Incomplete/failed run cannot support comparison: {path}')
    config['scheme']=config.get('scheme',config.get('experiment_name'))
    if config.get('scheme') not in SCHEMES:
        raise ValueError('Unknown scheme')
    required=('protocol_fingerprint','batch_plan_sha256','initial_checkpoint_sha256','data_stream_sha256')
    if any(key not in config for key in required):
        raise ValueError(f'Missing required evidence in config: {path}')
    fp=config['protocol_fingerprint']
    if not isinstance(fp,dict) or not fp:
        raise ValueError('Empty protocol fingerprint')
    for key,expected in {'scale':4,'steps':2000,'steps_per_epoch':100,'batch_size':4,'patch_lr':48,'patch_hr':192,'seed':1,'precision':'FP32','train_range':[1,800],'validation_ids':list(range(811,831)),'validation_crop_lr':96,'optimizer':'Adam','betas':[.9,.999],'learning_rate':1e-5,'scheduler':'constant','self_ensemble':False,'chop':False,'normalize_overlap':False,'eval_refine_iters':0}.items():
        if fp.get(key)!=expected:
            raise ValueError(f'Unregistered protocol {key}: {path}')
    for key in ('model_sha256','runner_sha256','metric_source_sha256','metric_protocol','gpu','torch','augmentation'):
        if key not in fp or fp[key] is None:
            raise ValueError(f'Missing common protocol {key}')
    expected_weight='44999471d8cc2d5f7dbf10d354766e08a5d23a84200a1060dfa9a9ed7a7711dd'
    if config['initial_checkpoint_sha256']!=expected_weight or fp.get('initial_checkpoint_sha256')!=expected_weight or fp.get('batch_plan_sha256')!=config['batch_plan_sha256']:
        raise ValueError('Initial checkpoint or plan differs from registered evidence')
    metrics=read_csv(Path(path)/'metrics.csv')
    if [int(r['epoch']) for r in metrics]!=EPOCHS or [int(r['global_step']) for r in metrics]!=[e*100 for e in EPOCHS]:
        raise ValueError(f'Incomplete/duplicated/out-of-order metrics epochs: {path}')
    for row in metrics:
        for key in ('validation_psnr','validation_ssim'):
            if not math.isfinite(float(row[key])): raise ValueError('Nonfinite epoch metrics')
    stream=read_csv(Path(path)/'batch_hashes.csv')
    if [int(r['global_step']) for r in stream]!=list(range(1,2001)):
        raise ValueError('Actual data-stream ledger must contain all 2000 steps once')
    if any(len(r.get('sha256',''))!=64 for r in stream):
        raise ValueError('Invalid batch SHA256')
    stream_digest=hashlib.sha256()
    for row in stream:
        try: bytes.fromhex(row['sha256'])
        except ValueError: raise ValueError('Batch hash is not hexadecimal')
        stream_digest.update(row['sha256'].encode())
    if stream_digest.hexdigest()!=config['data_stream_sha256']:
        raise ValueError('Data stream digest disagrees with step ledger')
    rows=read_json(Path(path)/'per_image.json')
    timeline={e:{} for e in EPOCHS}; benchmarks={}
    for row in rows:
        epoch=row['epoch']; step=row['global_step']
        if epoch not in EPOCHS or step!=epoch*100 or not all(finite(row.get(k)) for k in ('psnr','ssim')):
            raise ValueError('Invalid per-image epoch, step or metrics')
        if not row.get('alignment') or not row.get('lr_hash') or not row.get('hr_hash'):
            raise ValueError('Per-image input hashes required')
        key=row_key(row)
        if row['split'] in ('validation','inspection','adaptation_validation'):
            if key in timeline[epoch]: raise ValueError('Duplicate validation image')
            timeline[epoch][key]=row
        elif row['split']=='benchmark_final':
            if epoch!=20 or key in benchmarks: raise ValueError('Invalid final benchmark record')
            benchmarks[key]=row
        else:
            raise ValueError(f'Unexpected split {row["split"]}')
    first=set(timeline[0])
    if len(first)!=20 or {key[2].zfill(4) for key in first}!=VALID_IDS or {k[1] for k in first}!={'DIV2K_valid'}:
        raise ValueError('Validation must contain exactly DIV2K 0811–0830')
    for epoch in EPOCHS:
        if set(timeline[epoch])!=first: raise ValueError('Missing validation image or changed split/dataset')
        for key,row in timeline[epoch].items():
            base=timeline[0][key]
            if (row['lr_hash'],row['hr_hash'])!=(base['lr_hash'],base['hr_hash']):
                raise ValueError('Validation inputs changed across epochs')
        for metric,column in [('psnr','validation_psnr'),('ssim','validation_ssim')]:
            aggregate=average([r[metric] for r in timeline[epoch].values()])
            if abs(aggregate-float(metrics[epoch][column]))>1e-8:
                raise ValueError('metrics.csv disagrees with complete per-image metrics')
    return {'path':str(path),'config':config,'metrics':metrics,'timeline':timeline,'stream':stream,'benchmarks':benchmarks}

def compare_rows(candidate,baseline,metric):
    if set(candidate)!=set(baseline): raise ValueError('Pair image sets differ')
    pairs=[]
    for key in sorted(baseline):
        a,b=candidate[key],baseline[key]
        if a.get('alignment')!=b.get('alignment'):
            raise ValueError('Paired crop alignment differs')
        if (a['lr_hash'],a['hr_hash'])!=(b['lr_hash'],b['hr_hash']):
            raise ValueError('Paired LR/HR input hashes differ')
        pairs.append(('/'.join(key),a[metric]-b[metric]))
    return paired_stats(pairs)

def summarize(paths):
    runs={}
    for path in paths:
        run=load_run(path); scheme=run['config']['scheme']
        if scheme in runs: raise ValueError('Duplicate scheme run')
        runs[scheme]=run
    if set(runs)!=SCHEMES: raise ValueError('Exactly P0/P3/P4 completed runs required')
    p0=runs['P0']; c0=p0['config']; first=p0['timeline'][0]
    for scheme,run in runs.items():
        c=run['config']
        if comparable(c['protocol_fingerprint'])!=comparable(c0['protocol_fingerprint']):
            raise ValueError(f'Common protocol fingerprint mismatch: {scheme}')
        for key in ('dataset','batch_plan_sha256','initial_checkpoint_sha256','data_stream_sha256'):
            if c[key]!=c0[key]: raise ValueError(f'Common {key} mismatch: {scheme}')
        if [(r['global_step'],r['sha256']) for r in run['stream']]!=[(r['global_step'],r['sha256']) for r in p0['stream']]:
            raise ValueError(f'Actual training tensor stream mismatch: {scheme}')
        initial=run['timeline'][0]
        compare_rows(initial,first,'psnr')
        for key in first:
            if abs(initial[key]['psnr']-first[key]['psnr'])>1e-5 or abs(initial[key]['ssim']-first[key]['ssim'])>1e-6:
                raise ValueError(f'Initial zero-bias output metrics outside registered tolerance: {scheme} {key}')
    result={'scope':'2000-step fixed-checkpoint adaptation, one seed; no from-scratch convergence, novelty or industry-leading claim.',
            'protocol_fingerprint':comparable(c0['protocol_fingerprint']),
            'initial_checkpoint_sha256':c0['initial_checkpoint_sha256'],'batch_plan_sha256':c0['batch_plan_sha256'],
            'data_stream_sha256':c0['data_stream_sha256'],'runs':{},'comparisons':{},'benchmarks':{}}
    for scheme,run in runs.items():
        means=[average([r['psnr'] for r in run['timeline'][e].values()]) for e in EPOCHS]
        best=max(range(len(means)),key=lambda e:means[e])
        result['runs'][scheme]={'path':run['path'],'commit':run['config'].get('git_commit'),
            'scientific_type':'training strategy' if scheme=='P3' else ('structure' if scheme=='P4' else 'matched-budget L1 control'),
            'epoch0_psnr':means[0],'final_psnr':means[-1],'last3_psnr':average(means[-3:]),
            'best_psnr_descriptive_only':means[best],'best_epoch_descriptive_only':best,
            'final_ssim':average([r['ssim'] for r in run['timeline'][20].values()]),
            'vs_initial_final_psnr':compare_rows(run['timeline'][20],run['timeline'][0],'psnr'),
            'vs_initial_final_ssim':compare_rows(run['timeline'][20],run['timeline'][0],'ssim')}
    for scheme in ('P3','P4'):
        run=runs[scheme]
        final=compare_rows(run['timeline'][20],p0['timeline'][20],'psnr')
        ssim=compare_rows(run['timeline'][20],p0['timeline'][20],'ssim')
        delta_epochs={e:average([run['timeline'][e][key]['psnr']-p0['timeline'][e][key]['psnr'] for key in first]) for e in EPOCHS}
        last3=average([delta_epochs[e] for e in (18,19,20)])
        verdict='GRAY'
        if final['mean']>0 and last3>0 and final['median']>0 and final['win_rate']>=.6 and final['bootstrap95_ci'][0]>0 and ssim['mean']>=-1e-4:
            verdict='PROVISIONAL_GO'
        elif final['mean']<=0 and last3<=0: verdict='NO_GO'
        result['comparisons'][scheme]={'control':'P0 matched extra L1 budget','type':result['runs'][scheme]['scientific_type'],
            'final_psnr_delta':final,'last3_mean_psnr_delta':last3,'final_ssim_delta':ssim,
            'epoch_psnr_delta':delta_epochs,'decision':verdict,'decision_scope':'Investment ranking only; no automatic longer training.'}
    flags={bool(run['config'].get('benchmark_final')) for run in runs.values()}
    if len(flags)!=1:
        raise ValueError('Benchmark-final flag differs across groups')
    if True in flags and not all(run['benchmarks'] for run in runs.values()):
        raise ValueError('Requested final benchmarks are incomplete')
    if any(run['benchmarks'] for run in runs.values()):
        expected_counts={'Set5':5,'Set14':14,'B100':100,'Urban100':100,'manga109':109}
        for scheme,run in runs.items():
            b=run['benchmarks']; counts={name:sum(k[1]==name for k in b) for name in expected_counts}
            if counts!=expected_counts or len(b)!=328:
                raise ValueError('Explicit benchmark-final requires all five full datasets for every scheme')
            result['benchmarks'][scheme]={name:{'images':count,'psnr':average([r['psnr'] for k,r in b.items() if k[1]==name]),'ssim':average([r['ssim'] for k,r in b.items() if k[1]==name])} for name,count in counts.items()}
            compare_rows(b,p0['benchmarks'],'psnr')
    return result

def markdown(report):
    lines=['# P0 / P3 / P4 配对适配结果','',report['scope'],'',
           '| 组别 | 类型 | step0 PSNR | final PSNR | last3 PSNR | best（仅描述） |', '|---|---|---:|---:|---:|---:|']
    for scheme,row in sorted(report['runs'].items()):
        lines.append(f"| {scheme} | {row['scientific_type']} | {row['epoch0_psnr']:.9f} | {row['final_psnr']:.9f} | {row['last3_psnr']:.9f} | {row['best_psnr_descriptive_only']:.9f} @ {row['best_epoch_descriptive_only']} |")
    lines += ['', '主比较是同预算 P0；相对官方初始化的变化只作为额外训练效果描述，不能归因于候选。', '',
              '| 候选 | final ΔPSNR | last3 ΔPSNR | 中位数 | 胜率 | 图片bootstrap95%CI | ΔSSIM | 决策 |', '|---|---:|---:|---:|---:|---|---:|---|']
    for scheme,row in sorted(report['comparisons'].items()):
        d=row['final_psnr_delta']
        lines.append(f"| {scheme} | {d['mean']:.9f} | {row['last3_mean_psnr_delta']:.9f} | {d['median']:.9f} | {d['win_rate']:.3f} | {d['bootstrap95_ci']} | {row['final_ssim_delta']['mean']:.9f} | {row['decision']} |")
        lines += ['', f"{scheme} 最大影响图：{d['max_influence_image']}；Δ={d['max_influence_delta']:+.9f}；leave-one-out mean={d['leave_one_out_mean']:.9f}。", '']
    lines += ['P3 属于训练目标／优化量纲变化；P4 属于位置建模结构变化，分别解释。单 seed 图片 CI 不能证明跨 seed 稳定性；不自动延长预算。', '']
    if report['benchmarks']:
        lines += ['## 最终权重五 benchmark（独立外测，不参与筛选）','', '| 组别 | 数据集 | 图片数 | PSNR | SSIM |','|---|---|---:|---:|---:|']
        for scheme,datasets in sorted(report['benchmarks'].items()):
            for name,row in datasets.items():
                lines.append(f"| {scheme} | {name} | {row['images']} | {row['psnr']:.9f} | {row['ssim']:.9f} |")
    return '\n'.join(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); target=Path(args.output).resolve()
    if target.suffix!='.json': parser.error('--output must be .json')
    root=Path(__file__).resolve().parents[1]
    for path in (target,target.with_suffix('.md')):
        if subprocess.run(['git','check-ignore','--quiet',str(path)],cwd=root).returncode:
            parser.error('Report outputs must be git-ignored')
    report=summarize(args.runs)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    target.with_suffix('.md').write_text(markdown(report),encoding='utf-8')
    print(target)

if __name__=='__main__': main()
