#!/usr/bin/env python3
"""CPU smoke test for the N16 efficiency profiler."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))

from model.lfmn import Net as BaselineNet
from model.lfmnsrprv2 import Net as SRPRv2Net


def main():
    root = ROOT / '.n16_efficiency_check'
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    try:
        source, output = root / 'source', root / 'output'
        (source / 'c1/model').mkdir(parents=True)
        (source / 'srpr_c1/model').mkdir(parents=True)
        (source / 'wrapper_exit_status.txt').write_text('0\n')
        (source / 'summary_150e.json').write_text(json.dumps({
            'decision': 'PROMOTE_TO_LONG_RUN_VALIDATION'
        }))
        torch.save(BaselineNet(scale=4).state_dict(), source / 'c1/model/model_150.pt')
        torch.save(SRPRv2Net(scale=4).state_dict(), source / 'srpr_c1/model/model_150.pt')
        subprocess.run([
            sys.executable, str(ROOT / 'repro/profile_n16_efficiency.py'),
            '--source', str(source), '--output', str(output), '--sizes', '28',
            '--warmup', '1', '--repeats', '2', '--device', 'cpu',
        ], cwd=ROOT, check=True)
        report = json.loads((output / 'efficiency_profile.json').read_text(encoding='utf-8'))
        assert report['parameters'] == {'c1': 759627, 'srpr_c1': 841563}
        for name in ('c1', 'srpr_c1'):
            assert report['results']['28'][name]['output_finite']
            assert report['results']['28'][name]['output_shape'] == [1, 3, 112, 112]
        assert (output / 'efficiency_profile.txt').is_file()
        print(json.dumps({
            'n16_efficiency_profile_smoke': 'passed',
            'strict_checkpoint_load': 'passed',
            'parameter_counts': report['parameters'],
            'finite_outputs': 'passed', 'report_generation': 'passed',
        }, indent=2))
    finally:
        shutil.rmtree(root)


if __name__ == '__main__':
    main()
