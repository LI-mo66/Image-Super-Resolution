"""Explicit Linux shutdown after verified F4 completion; never run in tests."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def shutdown(group, invoke=False):
    group=Path(group).resolve()
    summary=json.loads((group/'summary.json').read_text(encoding='utf-8'))
    resources=json.loads((group/'resources.json').read_text(encoding='utf-8'))
    config=json.loads((group/'F4_x4_seed1/config.json').read_text(encoding='utf-8'))
    checkpoint=group/'F4_x4_seed1/model/model_20.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (summary['status']!='COMPLETE_F4_ONLY_BASELINE_PENDING' or
        summary['protocol']['scheduler_t_max']!=150 or
        int(summary['final']['epoch'])!=20 or config['status']!='completed' or
        set(summary['benchmarks'])!={'Set5','Set14','B100','Urban100','Manga109'} or
        len(resources['results'])!=3 or
        digest!=summary['checkpoint_sha256'] or digest!=resources['checkpoint_sha256']):
        raise ValueError('completion/benchmark/resources/checkpoint integrity failed: no shutdown')
    if not invoke:
        print('Completion verified; shutdown not invoked',flush=True)
        return
    if sys.platform!='linux':raise RuntimeError('Shutdown is Linux server only')
    # F4 children have exited. Any remaining GPU compute PID is another job.
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True)
    if any(line.strip().isdigit() for line in query.stdout.splitlines()):
        raise RuntimeError('Other GPU compute process detected: shutdown refused')
    marker=group/'shutdown_status.json'
    marker.write_text(json.dumps(dict(status='requested',group=str(group),
        completed_at=datetime.datetime.now().astimezone().isoformat(),git_commit=config.get('git_commit')),indent=2),encoding='utf-8')
    print('F4 complete and saved. Requesting server shutdown now.',flush=True)
    os.sync()
    try:
        # Bash also supports platforms whose shutdown helper has no shebang.
        subprocess.run(['bash','-c','shutdown -h now'],check=True)
    except BaseException as error:
        marker.write_text(json.dumps(dict(status='failed',error=str(error)),indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group',type=Path,required=True)
    p.add_argument('--invoke',action='store_true')
    a=p.parse_args();shutdown(a.group,a.invoke)
