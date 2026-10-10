"""Engineering-only paired continuation: one batch per path, no formal epochs."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from run_f4_150_server import snapshot,check_boundary,training_command,continuation_config,safe_state


def equal(a,b):
    import torch
    import numpy as np
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-smoke',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);a=p.parse_args()
    source=a.source_smoke.resolve();before=snapshot(source)
    check_boundary(source,2,steps_per_epoch=1)
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
    args=SimpleNamespace(data_root=a.data_root.resolve(),workers=4,cpu=True,data_fingerprint=cfg['data_fingerprint'])
    group=ROOT/'experiment'/('F4_150_paired_smoke_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    group.mkdir(parents=True,exist_ok=False)
    subprocess.run([sys.executable,str(ROOT/'repro/check_training_run_logging.py')],cwd=ROOT,check=True)
    outputs=[]
    for kind in ['original','monitor']:
        parent=group/kind;parent.mkdir()
        directory=parent/'F4_x4_seed1';shutil.copytree(source,directory)
        write_json(parent/'continuation_manifest.json',dict(parent_group=str(source.parent),start_epoch=2))
        c=training_command(args,directory,2,target=3,smoke=True)
        if kind=='original':c[2]=str(ROOT/'repro/f4_train_entry.py')
        new=continuation_config(args,c,source,2,target=3)
        new.update(data_range='1-1/801-801',stop_epoch=3,test_every=1,max_train_batches=1,
                   kind='ENGINEERING_ONLY_ONE_BATCH',checkpoint_selection='engineering only')
        launch(c,directory,new,ROOT/'LFMN',resume=True)
        check_boundary(directory,3,steps_per_epoch=1)
        outputs.append(directory)
    for name in ['model/model_3.pt','optimizer.pt','scheduler.pt','resume_state.pt','psnr_log.pt','ssim_log.pt','loss_log.pt']:
        if not equal(safe_state(outputs[0]/name),safe_state(outputs[1]/name)):
            raise AssertionError('monitor changed training trajectory: '+name)
    if snapshot(source)!=before:raise AssertionError('original smoke source changed')
    cfg2=json.loads((outputs[1]/'config.json').read_text(encoding='utf-8'))
    assert cfg2['sessions'][-1]['resume'] and cfg2['resume_start_epoch']==2
    assert 'Restored RNG/DataLoader state at epoch 2' in (outputs[1]/'train_log.txt').read_text(encoding='utf-8')
    monitor=json.loads((outputs[1].parent/'set5_paths.json').read_text(encoding='utf-8'))
    assert set(monitor)=={'3'}
    write_json(group/'checks.json',dict(status='PASS',candidate_engineering_batches=2,baseline_optimizer_steps=0,
        parameters_Adam_scheduler_RNG_exact=True,original_source_unchanged=True,
        scope='one batch per resumed F4 path; engineering only, no PSNR evidence'))
    print('PAIRED CONTINUATION PASS:',group,flush=True)


if __name__=='__main__':main()
