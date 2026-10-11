"""Server gate: one full epoch + resume to epoch two, complete five-set evaluation.

Writes a source/data/GPU/environment-bound receipt only after all checks pass.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from lfmn_baseline_logging import atomic_json
from original_lfmn_x4 import ROOT, SETS, load_checkpoint, sha


def validate(run):
    a = load_checkpoint(run/'checkpoints/latest.pt')
    b = load_checkpoint(run/'checkpoints/best.pt')
    c = load_checkpoint(run/'checkpoints/final.pt')
    assert a['epoch'] == c['epoch'] == 2 and a['global_step'] == 2000
    assert b['epoch'] == a['best_snapshot']['epoch']
    assert a['scheduler']['last_epoch'] == 2
    assert a['optimizer']['state'] and a['rng']['loader'].numel()
    assert all(int(s['step']) == 2000 for s in a['optimizer']['state'].values())
    assert all(a['metrics'][n]['count'] == count for n,count in SETS.items())
    rows = list(csv.DictReader((run/'metrics.csv').open(encoding='utf-8')))
    assert [int(r['epoch']) for r in rows] == [1,2]
    assert [int(r['global_step']) for r in rows] == [1000,2000]
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    assert config['status'] == 'complete' and config['resume_start_epoch'] == 2
    log = (run/'train_log.txt').read_text(encoding='utf-8')
    assert 'SESSION RESUME' in log and 'optimizer/scheduler/RNG/loader restored' in log
    assert len(config['sessions']) == 2
    assert (run/'psnr_curves.png').stat().st_size > 0
    return a,config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--output',default=str(ROOT/'experiment/all_runs'))
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=1)
    args = p.parse_args()
    subprocess.run([sys.executable,str(ROOT/'repro/check_lfmn_baseline_logging.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'repro/check_original_lfmn_x4.py'),
                    '--infrastructure-only'],check=True)
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    before = set(output.iterdir())
    command = [sys.executable,'-u',str(ROOT/'repro/original_lfmn_x4.py'),'--data',args.data,
               '--output',str(output),'--workers',str(args.workers),'--seed',str(args.seed),'--smoke']
    started = time.time()
    subprocess.run(command+['--stop-after','1'],check=True)
    runs = [r for r in set(output.iterdir())-before if (r/'config.json').exists()]
    assert len(runs) == 1
    run = runs[0]
    old_log = (run/'train_log.txt').read_bytes()
    first = load_checkpoint(run/'checkpoints/latest.pt')
    first_row = list(csv.DictReader((run/'metrics.csv').open(encoding='utf-8')))[0]
    assert first['epoch'] == 1 and first['global_step'] == 1000
    subprocess.run(command+['--resume',str(run)],check=True)
    assert (run/'train_log.txt').read_bytes().startswith(old_log)
    assert list(csv.DictReader((run/'metrics.csv').open(encoding='utf-8')))[0] == first_row
    state,config = validate(run)
    assert state['history'][0] == first['history'][0]
    receipt = dict(status='passed',fingerprint=state['fingerprint'],run_directory=str(run),
                   gpu_name=config['gpu_name'],torch_version=config['torch_version'],
                   latest_sha256=sha(run/'checkpoints/latest.pt'),wall_seconds=time.time()-started,
                   checks=['logging infrastructure','metric equivalence','checkpoint/RNG roundtrip',
                           '800 training pairs + 328 benchmark pairs decoded/hashed',
                           'two full 1000-step epochs','all five complete datasets each epoch',
                           'full-state resume epoch1->2','old CSV/log preserved','latest/best/final','plots/report'])
    path = run/'preflight_passed.json'
    atomic_json(path,receipt)
    print('PASS SERVER PREFLIGHT receipt=%s' % path,flush=True)
    print('Start fresh formal run: python -u repro/original_lfmn_x4.py --data "%s" '
          '--workers %d --seed %d --preflight-receipt "%s"' % (args.data,args.workers,args.seed,path))


if __name__ == '__main__':
    main()
