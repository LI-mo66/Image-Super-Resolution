"""Fail-closed paired full-precision summary, or bounded integration-smoke report."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

GROUPS = ('n21','n22','b0')

def config(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ': ' in line:
            key,value = line.split(': ',1)
            if key in result:
                raise ValueError('Duplicate config field')
            result[key]=value
    return result

def summarize(output):
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    protocol=manifest['protocol']
    epochs=protocol['epochs']
    expected_names={f'{i:04d}' for i in protocol['validation_ids']}
    curves,rows,proofs,batch_fingerprints={}, {}, {}, {}
    configs={g:config(output/g/'config.txt') for g in GROUPS}
    excluded={'model','save'}
    for group in GROUPS:
        if configs[group]['model'] != {'n21':'lfmn_n21','n22':'lfmn_n22','b0':'LFMN'}[group]:
            raise ValueError('Model identity mismatch')
        for key,value in configs['b0'].items():
            if key not in excluded and configs[group].get(key)!=value:
                raise ValueError(f'Config mismatch {group}: {key}')
        for key,value in protocol['expected_config'].items():
            if configs[group].get(key)!=str(value):
                raise ValueError(f'Unexpected training config {group}: {key}')
        proofs[group]=json.loads((output/group/'initial_state_proof.json').read_text())
        curves[group]={}
        rows[group]={}
        for metric in ('psnr','ssim'):
            tensor=torch.load(output/group/f'{metric}_log.pt',map_location='cpu',weights_only=True)
            if tuple(tensor.shape)!=(epochs,1,1) or not torch.isfinite(tensor).all():
                raise ValueError(f'Incomplete/nonfinite {group} {metric} curve')
            curves[group][metric]=tensor[:,0,0].double().numpy()
        for epoch in range(1,epochs+1):
            payload=torch.load(output/group/'per_image_metrics'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
            indexed={r['filename']:r for r in payload}
            if len(indexed)!=len(payload) or set(indexed)!=expected_names:
                raise ValueError('Per-image identities/count mismatch')
            if any(r['dataset']!='DIV2K' or r['scale']!=4 for r in payload):
                raise ValueError('Dataset/scale mismatch')
            for metric,tol in [('psnr',2e-4),('ssim',1e-5)]:
                values=np.array([r[metric] for r in payload])
                if not np.isfinite(values).all() or abs(values.mean()-curves[group][metric][epoch-1])>tol:
                    raise ValueError('Per-image / curve mismatch')
            rows[group][epoch]=indexed
        scheduler=torch.load(output/group/'scheduler.pt',map_location='cpu',weights_only=True)
        if scheduler['T_max']!=150 or scheduler['last_epoch']!=epochs:
            raise ValueError('Scheduler time-axis mismatch')
        steps=[json.loads(line) for line in (output/group/'mechanism.jsonl').read_text().splitlines()]
        if len(steps)!=epochs or any(r['epoch']!=i+1 or r['updates']!=protocol['updates_per_epoch'] for i,r in enumerate(steps)):
            raise ValueError('Training update count mismatch')
        state=torch.load(output/group/'model'/f'model_{epochs}.pt',map_location='cpu',weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
            raise ValueError('Nonfinite final weights')
        import importlib
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'LFMN'))
        net=importlib.import_module('model.'+configs[group]['model'].lower()).Net(scale=4)
        net.load_state_dict(state,strict=True)
        batch_fingerprints[group]=[json.loads(line) for line in (output/group/'batch_fingerprints.jsonl').read_text().splitlines()]
        if len(batch_fingerprints[group]) != epochs:
            raise ValueError('Incomplete batch fingerprints')
    if len({v['common_initial_state_sha256'] for v in proofs.values()})!=1:
        raise ValueError('Unmatched common initialization')
    if not batch_fingerprints['n21']==batch_fingerprints['n22']==batch_fingerprints['b0']:
        raise ValueError('First-batch crop/augmentation fingerprints differ across groups')
    result={'protocol':protocol,'integrity':'PASS','comparisons':{}}
    names=sorted(expected_names)
    for group in ('n21','n22'):
        d=curves[group]['psnr']-curves['b0']['psnr']
        delta=np.array([rows[group][epochs][n]['psnr']-rows['b0'][epochs][n]['psnr'] for n in names])
        rng=np.random.default_rng(2122)
        bootstrap=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
        ci=np.quantile(bootstrap,[0.025,0.975]).tolist()
        ssim=float(curves[group]['ssim'][-1]-curves['b0']['ssim'][-1])
        tail=float(d[-5:].mean())
        if protocol['smoke']:
            decision='INTEGRATION_ONLY'
        elif d[-1]>=0.01 and tail>=0.01 and ci[0]>0 and np.median(delta)>0 and (delta>0).mean()>=0.6 and ssim>=-1e-4:
            decision='GO_ELIGIBLE: efficiency audit and independent confirmation required'
        elif d[-1]<=0 and tail<=0 or ssim < -1e-4:
            decision='NO_GO'
        else:
            decision='GRAY: no automatic extension'
        result['comparisons'][group]={'decision':decision,'final_delta':float(d[-1]),'last5_mean_delta':tail,
            'positive_epoch_ratio':float((d>0).mean()),'per_image_mean':float(delta.mean()),
            'per_image_median':float(np.median(delta)),'win_rate':float((delta>0).mean()),
            'bootstrap95':ci,'ssim_delta':ssim,'epoch_deltas':d.tolist()}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    print(json.dumps(summarize(ap.parse_args().output.resolve()),indent=2))
