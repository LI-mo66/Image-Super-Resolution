"""Shutdown only after complete frozen gate report; mocked in tests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from probe_f4_gates import SHA150,IDS,MODES,sha


def shutdown(output,invoke=False):
    out=Path(output)
    r=json.loads((out/'gate_report.json').read_text(encoding='utf-8'))
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    if (r['status']!='COMPLETE_F4_GATE_DIAGNOSTIC' or r['ids']!=IDS or r['modes']!=MODES
        or r['formal_forwards']!=240 or r['optimizer_steps']!=0 or not r['state_unchanged']
        or not r['checkpoint_unchanged'] or not r['flags_restored'] or cfg['status']!='completed'
        or r['checkpoint_sha256']!=SHA150 or sha(cfg['checkpoint'])!=SHA150):
        raise ValueError('incomplete gate diagnostic; no shutdown')
    if not invoke:return
    if sys.platform!='linux':raise RuntimeError('Linux only')
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    if any(line.strip().isdigit() for line in q.stdout.splitlines()):raise RuntimeError('other GPU task: no shutdown')
    print('Gate report complete and saved. Requesting server shutdown.',flush=True)
    os.sync();subprocess.run(['bash','-c','shutdown -h now'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--invoke',action='store_true')
    a=p.parse_args();shutdown(a.output,a.invoke)
