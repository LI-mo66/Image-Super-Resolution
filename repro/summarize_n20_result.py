"""Post-run sensitivity audit; never changes preregistered N20 gates."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=json.loads((args.directory/'report.json').read_text(encoding='utf-8'))
    log=[json.loads(line) for line in (args.directory/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in log]==list(range(1,201))
    assert result['steps_per_group']==200
    assert all(len(row['batch'])==2 and len(row['batch_sha256'])==64 for row in log)
    assert all(row['stats']['b']['kd']==0 and np.isfinite(list(row['stats']['b'].values())).all()
               and np.isfinite(list(row['stats']['k'].values())).all() for row in log)
    assert [r['image'] for r in result['start']]==list(range(843,859))
    rng=np.random.default_rng(2002)
    stats={}
    for mode in ('l1','kd'):
        assert [r['image'] for r in result[mode]]==list(range(843,859))
        delta=np.array([r['psnr']-b['psnr'] for r,b in zip(result[mode],result['start'])])
        boot=delta[rng.integers(0,16,(10000,16))].mean(1)
        maximum=int(delta.argmax())
        stats[mode]={'mean':float(delta.mean()),'median':float(np.median(delta)),
                     'wins':int((delta>0).sum()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
                     'largest_gain_image':843+maximum,'largest_gain':float(delta[maximum]),
                     'mean_excluding_largest_gain':float(np.delete(delta,maximum).mean())}
    audit={'training_log_complete':True,'stats_vs_start':stats,'preregistered_summary':result['summary'],
           'limitation':'Largest-gain exclusion is post-hoc sensitivity, not a new selection rule or performance endpoint.'}
    (args.directory/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
