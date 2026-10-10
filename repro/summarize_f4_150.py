"""Full150 absolute results; no comparison against an epoch20 baseline."""
import csv
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import write_json
from backfill_f4_set5 import digest,read_evaluation,aggregate
from run_f4_screen_server import BENCHMARKS


def validate_completion(group):
    import torch
    from run_f4_150_server import check_boundary,assert_parent_unchanged
    group=Path(group);train=group/'F4_x4_seed1'
    curve=check_boundary(train,150);assert_parent_unchanged(group)
    cfg=json.loads((train/'config.json').read_text(encoding='utf-8'))
    actual=cfg['resolved_arguments']
    if (cfg['status']!='completed' or actual['epochs']!=150 or actual['scheduler_t_max']!=150
        or actual['self_ensemble'] or actual['chop'] or actual['data_test']!=['DIV2K']
        or cfg['training_mode']!='CONTINUATION' or cfg['checkpoint_selection']!='fixed epoch150'):
        raise ValueError('actual continuation protocol differs')
    for e in range(1,151):
        per=torch.load(train/'per_image_metrics'/f'epoch_{e:04d}.pt',map_location='cpu',weights_only=True)
        aggregate(per,'DIV2K',100)
        for k in ['validation_psnr','validation_ssim']:
            if not math.isfinite(float(curve[e-1][k])):raise ValueError('nonfinite DIV2K curve')
    monitor=json.loads((group/'set5_paths.json').read_text(encoding='utf-8'))
    if set(monitor)!={str(e) for e in range(1,151)}:raise ValueError('need all150 Set5 epochs')
    rows=[]
    for e,r in enumerate(curve,1):
        checkpoint=train/'model'/f'model_{e}.pt';entry=monitor[str(e)]
        score=aggregate(read_evaluation(entry['directory'],checkpoint),'Set5',5)
        if entry['checkpoint_sha256']!=digest(checkpoint):raise ValueError('Set5 curve checkpoint differs')
        rows.append(dict(epoch=e,DIV2K_PSNR=float(r['validation_psnr']),DIV2K_SSIM=float(r['validation_ssim']),
                         Set5_PSNR=score['psnr'],Set5_SSIM=score['ssim']))
    checkpoint=train/'model/model_150.pt';sha=digest(checkpoint)
    info=json.loads((group/'final_evaluation.json').read_text(encoding='utf-8'))
    if info['checkpoint_sha256']!=sha:raise ValueError('final benchmark checkpoint differs')
    per=read_evaluation(info['directory'],checkpoint)
    benchmarks={n:aggregate(per,n,count) for n,count in BENCHMARKS.items()}
    for k in ['psnr','ssim']:
        if abs(benchmarks['Set5'][k]-rows[-1]['Set5_'+k.upper()])>1e-4:
            raise ValueError('fixed150 Set5 final/monitor discrepancy')
    resources=json.loads((group/'resources150.json').read_text(encoding='utf-8'))
    if resources['checkpoint_sha256']!=sha or resources['parameters']!=766859 or resources['self_ensemble']:
        raise ValueError('resources checkpoint/protocol differs')
    if set(resources['results'])!={'LR64x64','LR128x128','LR96x160'}:
        raise ValueError('three resource sizes required')
    for item in resources['results'].values():
        if item['counted_flops']<=0 or any(not math.isfinite(item[k]) for k in ['median_ms','p90_ms','peak_allocated_mib']):
            raise ValueError('incomplete/nonfinite resources')
    return rows,benchmarks,sha,per


def main():
    group=Path(sys.argv[1]).resolve()
    rows,benchmarks,sha,per=validate_completion(group)
    last5={k:sum(r[k] for r in rows[-5:])/5 for k in rows[-1] if k!='epoch'}
    report=dict(status='COMPLETE_F4_150_OFF_BASELINE_PENDING',checkpoint_sha256=sha,
        epochs=rows,final=rows[-1],last5=last5,benchmarks=benchmarks,
        total_epochs=150,scheduler_t_max=150,initial_phase_epochs=20,additional_epochs=130,
        baseline_comparison='PENDING: require matched B0 total150, never B0_20',
        checkpoint_selection='fixed epoch150',self_ensemble=False)
    lines=['F4 | scratch trajectory continued20→150 | T_max150 | seed1 | OFF | fixed epoch150']
    lines += ['EPOCH {epoch:03d} DIV2K PSNR={DIV2K_PSNR:.9f} SSIM={DIV2K_SSIM:.9f} '
              'Set5 PSNR={Set5_PSNR:.9f} SSIM={Set5_SSIM:.9f}'.format(**r) for r in rows]
    lines += ['LAST5 '+json.dumps(last5)]
    lines += ['EPOCH150 {} PSNR={psnr:.9f} SSIM={ssim:.9f}'.format(n,**s) for n,s in benchmarks.items()]
    lines += ['Checkpoint SHA256: '+sha,'B0 total150 comparison pending. Original20 results preserved. No architecture gain claim.']
    write_json(group/'summary150.json',report)
    text='\n'.join(lines);(group/'summary150.txt').write_text(text,encoding='utf-8')
    with (group/'benchmark150_per_image.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per[0]));w.writeheader();w.writerows(per)
    print(text,flush=True)


if __name__=='__main__':main()
