"""Synthetic audit of stage selection, ties, corrupt histories and read-only behavior."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from report_original_lfmn_stages import SETS


def main():
    root = Path(__file__).resolve().parents[1] / 'experiment' / 'baseline_checks'
    root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('report_original_lfmn_stages.py')
    with tempfile.TemporaryDirectory(dir=root) as temp:
        run = Path(temp) / 'formal'
        run.mkdir()
        config = dict(smoke=False, backend='official', scale=4, model='LFMN',
                      status='complete', last_completed_epoch=1000, best_epoch=500,
                      self_ensemble_enabled=False, pretrained_checkpoint=None)
        (run/'config.json').write_text(json.dumps(config), encoding='utf-8')
        rows = []
        for epoch in range(1, 1001):
            values = [30.0 + (2 if epoch in (500, 501) else 0)] * 5
            if epoch == 700:
                values = [35.0, 28.0, 28.0, 28.0, 28.0]
            row = dict(epoch=epoch, global_step=epoch*1000, learning_rate=.0002,
                       elapsed_seconds=epoch*10, selection_psnr=sum(values)/5)
            for name, value in zip(SETS, values):
                row.update({name+'_psnr': value, name+'_ssim': .8})
            rows.append(row)
        def save():
            with (run/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
                writer = csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        def invoke(*args):
            return subprocess.run([sys.executable,str(script),*args], capture_output=True,text=True)
        save()
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
        good = invoke('--root',temp)
        assert good.returncode == 0, good.stderr
        assert 'ONE epoch 500' in good.stdout
        assert 'Set5,32.00000000,0.80000000,30.00000000,0.80000000,35.00000000,700' in good.stdout
        assert 'PHASE 801-1000 END:' in good.stdout
        assert before == {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir()}
        rows[-1]['epoch'] = 999
        save()
        assert invoke('--run',str(run)).returncode != 0
        rows[-1]['epoch'] = 1000
        rows[-1]['selection_psnr'] = 999
        save()
        assert invoke('--run',str(run)).returncode != 0
        config['smoke'] = True
        (run/'config.json').write_text(json.dumps(config),encoding='utf-8')
        assert invoke('--run',str(run)).returncode != 0
        assert invoke('--root',temp).returncode != 0
    print('PASS: fixed milestones, common best/earliest tie, independent diagnostic, '
          'read-only, duplicate epochs/score corruption/smoke/no-run rejection')


if __name__ == '__main__':
    main()
