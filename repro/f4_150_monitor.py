"""Set5 monitoring in separate eval-only processes, preserving parent RNG."""
import contextlib
import csv
import datetime
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
from run_logging import launch,write_json
from run_f4_screen_server import config
from backfill_f4_set5 import command,read_evaluation,aggregate,digest



def validate_continuation_args(args,cfg):
    if (not args.load or args.test_only or args.model!='LFMNF4' or args.data_test!=['DIV2K']
        or args.self_ensemble or args.chop or args.scheduler!='cosine' or args.scheduler_t_max!=150
        or args.precision!='single' or args.scale!=[4] or args.rgcrd_mode!='off'):
        raise ValueError('150 entry is restored F4 / DIV2K / FP32 / T150 / OFF only')
    if not args.resume < args.epochs <= 150:
        raise ValueError('continuation epoch budget must end by150')
    if cfg.get('kind')!='ENGINEERING_ONLY_ONE_BATCH' and args.epochs!=150:
        raise ValueError('formal continuation must end at total150')


@contextlib.contextmanager
def preserve_rng():
    import numpy as np
    import torch
    py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state()
    gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu)
        if gpu is not None:torch.cuda.set_rng_state_all(gpu)


def ensure_set5(group,epoch,args):
    group=Path(group);checkpoint=group/'F4_x4_seed1/model'/f'model_{epoch}.pt'
    path=group/'set5_paths.json'
    cache=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    key=str(epoch);entry=cache.get(key)
    if entry:
        rows=read_evaluation(entry['directory'],checkpoint)
        score=aggregate(rows,'Set5',5)
        if entry['checkpoint_sha256']!=digest(checkpoint):raise ValueError('cached Set5 checkpoint differs')
    else:
        parent=json.loads((group/'continuation_manifest.json').read_text(encoding='utf-8'))
        origin=Path(parent['parent_group'])
        reused=None
        if epoch<=parent['start_epoch']:
            for candidate in sorted(origin.glob(f'posthoc_Set5_*/Set5_epoch{epoch:02d}_OFF'),reverse=True):
                try:
                    candidate_cfg=json.loads((candidate/'config.json').read_text(encoding='utf-8'))
                    if candidate_cfg.get('data_fingerprint')!=args.data_fingerprint:continue
                    rows=read_evaluation(candidate,checkpoint)
                    aggregate(rows,'Set5',5);reused=candidate;break
                except (OSError,ValueError,KeyError):continue
        if reused is None:
            stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            directory=group/'epoch_monitor'/f'Set5_epoch{epoch:03d}_OFF_{stamp}'
            directory.parent.mkdir(exist_ok=True)
            before=digest(checkpoint)
            c=command(args,checkpoint,directory,['Set5'])
            cfg=config(args,c,test_only=True,checkpoint=str(checkpoint),kind='F4_150_SET5_MONITOR',
                       stop_epoch=150,checkpoint_selection='monitor only; fixed epoch150 final',
                       monitored_epoch=epoch)
            launch(c,directory,cfg,ROOT/'LFMN')
            rows=read_evaluation(directory,checkpoint)
            if digest(checkpoint)!=before:raise ValueError('monitor changed checkpoint')
        else:directory=reused
        score=aggregate(rows,'Set5',5)
        entry=dict(directory=str(directory.resolve()),checkpoint_sha256=digest(checkpoint),**score)
        cache[key]=entry;write_json(path,cache)
    print('SET5_MONITOR epoch={} PSNR={:.9f} SSIM={:.9f} OFF (observation only)'.format(
        epoch,score['psnr'],score['ssim']),flush=True)
    return score


def update_table(group):
    group=Path(group)
    with (group/'F4_x4_seed1/metrics.csv').open(encoding='utf-8') as f:curve=list(csv.DictReader(f))
    monitor=json.loads((group/'set5_paths.json').read_text(encoding='utf-8'))
    rows=[]
    for row in curve:
        e=int(row['epoch']);s=monitor.get(str(e))
        rows.append(dict(epoch=e,learning_rate=row['learning_rate'],train_loss=row['train_loss'],
            DIV2K_PSNR=row['validation_psnr'],DIV2K_SSIM=row['validation_ssim'],
            Set5_PSNR=s['psnr'] if s else '',Set5_SSIM=s['ssim'] if s else ''))
    target=group/'epochs_DIV2K_Set5.csv';temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temporary.replace(target)
    return rows


def after_epoch(directory,epoch):
    directory=Path(directory);group=directory.parent
    cfg=json.loads((directory/'config.json').read_text(encoding='utf-8'))
    args=SimpleNamespace(data_root=Path(cfg['data_root']),workers=cfg['workers'],
                         cpu=cfg['resolved_arguments']['cpu'],data_fingerprint=cfg['data_fingerprint'])
    with preserve_rng():
        ensure_set5(group,epoch,args);update_table(group)
