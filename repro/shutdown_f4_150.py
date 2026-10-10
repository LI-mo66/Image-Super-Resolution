"""Linux shutdown after verified complete150; mocked in tests only."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from summarize_f4_150 import validate_completion


def shutdown(group,invoke=False):
    group=Path(group).resolve();validate_completion(group)
    summary=json.loads((group/'summary150.json').read_text(encoding='utf-8'))
    if summary['status']!='COMPLETE_F4_150_OFF_BASELINE_PENDING' or summary['total_epochs']!=150:
        raise ValueError('not a completed150 summary')
    if not invoke:
        print('Complete150 validated; shutdown not invoked',flush=True);return
    if sys.platform!='linux':raise RuntimeError('shutdown only on Linux server')
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True)
    if any(line.strip().isdigit() for line in query.stdout.splitlines()):
        raise RuntimeError('Other GPU task remains: shutdown refused')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cfg=json.loads((group/'F4_x4_seed1/config.json').read_text(encoding='utf-8'))
    marker=group/f'shutdown150_status_{stamp}.json'
    marker.write_text(json.dumps(dict(status='requested',completed_at=stamp,git_commit=cfg.get('git_commit'))),encoding='utf-8')
    print('F4 total150 fully saved. Requesting server shutdown.',flush=True)
    os.sync()
    try:subprocess.run(['bash','-c','shutdown -h now'],check=True)
    except BaseException as error:
        marker.write_text(json.dumps(dict(status='failed',error=str(error))),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',type=Path,required=True);p.add_argument('--invoke',action='store_true')
    a=p.parse_args();shutdown(a.group,a.invoke)
