"""Summarize engineering evidence; intentionally never issues a PSNR GO."""
import argparse
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('report',type=Path)
    args=ap.parse_args()
    r=json.loads(args.report.read_text(encoding='utf-8'))
    assert r['id']=='n21' and r['status']=='PASS'
    assert r['performance_claim'] is False
    assert r['strict_reload_max_error']==0
    assert all(x['zero_gain_max_error']<1e-4 for x in r['shapes'])
    assert r.get('autocast_scaled_backward_finite') or r['device']=='cpu'
    print(f"Engineering PASS: {r['candidate_params']} parameters (+{r['candidate_params']-r['baseline_params']}).")
    print('PSNR effectiveness: UNKNOWN. No training-screen approval or GO is implied.')

if __name__=='__main__':
    main()
