"""N23 collection summary only: deliberately no B0 deltas or GO decision."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from summarize_n23_screen import (checked_curve, checked_images, trainer_float32_mean,
                                  read_config, normalized_config_value, audit_manifest)


def single_report(psnr, ssim):
    p, s = checked_curve(psnr, 'n23/psnr'), checked_curve(ssim, 'n23/ssim')
    if len(p) != 40 or len(s) != 40:
        raise ValueError('N23 must complete exactly 40 epochs; no completion/shutdown on missing metrics')
    return dict(decision='BASELINE_PENDING_NO_COMPARISON', performance_claim=False,
                completed_epochs=40, final_psnr=float(p[-1]), best_psnr=float(p.max()),
                best_psnr_epoch=int(p.argmax()+1), last5_mean_psnr=float(p[-5:].mean()),
                final_ssim=float(s[-1]), last5_mean_ssim=float(s[-5:].mean()),
                note='N23 absolute results only; no evidence of improvement over B0')


def summarize(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('run_mode') != 'N23_ONLY' or manifest['comparison_audit']['status'] != 'BASELINE_PENDING':
        raise ValueError('Not an explicitly registered N23-only collection')
    protocol = audit_manifest(manifest, complete=False)
    group = root/'n23'
    curves = {k: torch.load(group/f'{k}_log.pt', map_location='cpu', weights_only=True)
              for k in ('psnr', 'ssim')}
    report = single_report(curves['psnr'], curves['ssim'])
    config = read_config(group/'config.txt')
    assert config['model'] == 'lfmn_n23'
    for key, value in protocol['expected_config'].items():
        if normalized_config_value(config[key]) != normalized_config_value(value):
            raise ValueError(f'N23 protocol config mismatch: {key}')
    mechanism = [json.loads(s) for s in (group/'mechanism.jsonl').read_text().splitlines()]
    batches = [json.loads(s) for s in (group/'batch_fingerprints.jsonl').read_text().splitlines()]
    assert len(mechanism) == len(batches) == 40
    for epoch, row in enumerate(mechanism, 1):
        assert row['epoch'] == epoch and row['updates'] == 1000 and row['total_updates'] == epoch*1000
        assert batches[epoch-1]['epoch'] == epoch
    checkpoint_hashes = {}
    for epoch in range(1, 41):
        rows = torch.load(group/'per_image_metrics'/f'epoch_{epoch:04d}.pt', map_location='cpu', weights_only=True)
        images = checked_images(rows, f'n23/epoch{epoch}')
        for column, metric in enumerate(('psnr', 'ssim')):
            assert trainer_float32_mean(images[:, column]) == float(curves[metric][epoch-1, 0, 0])
        checkpoint = group/'model'/f'model_{epoch}.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state and all(bool(torch.isfinite(value).all()) for value in state.values())
        checkpoint_hashes[str(epoch)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report.update(protocol=protocol, checkpoint_sha256=checkpoint_hashes,
                  final_per_image_psnr_mean_float64=float(images[:, 0].mean()),
                  final_per_image_ssim_mean_float64=float(images[:, 1].mean()),
                  complete_per_image_epochs=40, data_hashes=manifest['data_hashes'],
                  source_commit=manifest['commit'], environment=manifest['environment'])
    with (root/'n23_epoch_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('epoch', 'N23_PSNR', 'N23_SSIM'))
        for epoch in range(1, 41):
            writer.writerow((epoch, f'{float(curves["psnr"][epoch-1, 0, 0]):.10f}',
                             f'{float(curves["ssim"][epoch-1, 0, 0]):.10f}'))
    (root/'n23_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    print(json.dumps(summarize(args.root), indent=2, allow_nan=False))
