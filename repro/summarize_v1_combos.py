"""Full-precision V1 combo output; historical V1 is reference only pending provenance."""
import argparse
import csv
import io
import json
from pathlib import Path


def metrics(directory, epochs=20):
    import numpy as np
    import torch
    curves = []
    for name in ('psnr_log.pt', 'ssim_log.pt'):
        values = torch.load(directory / name, map_location='cpu', weights_only=True)
        if values.ndim != 3 or tuple(values.shape[1:]) != (1, 1) or len(values) < epochs:
            raise ValueError(f'Incomplete/invalid curve: {directory/name}')
        curve = values[:epochs, 0, 0].double().numpy()
        if not np.isfinite(curve).all():
            raise ValueError('Nonfinite curve')
        curves.append(curve)
    last = None
    for epoch in range(1, epochs+1):
        rows = torch.load(directory/'per_image_metrics'/f'epoch_{epoch:04d}.pt', map_location='cpu', weights_only=True)
        if len(rows) != 100 or [r['filename'] for r in rows] != [f'{n:04d}' for n in range(801, 901)]:
            raise ValueError('Incomplete/duplicate/out-of-order DIV2K per-image rows')
        if any(r['dataset'] != 'DIV2K' or r['scale'] != 4 for r in rows):
            raise ValueError('Wrong metric dataset/scale')
        for index, metric in enumerate(('psnr', 'ssim')):
            acc = torch.tensor(0., dtype=torch.float32)
            for row in rows:
                if not np.isfinite(row[metric]):
                    raise ValueError('Nonfinite per-image metric')
                acc += row[metric]
            acc /= len(rows)
            if abs(float(acc)-curves[index][epoch-1]) > 1e-6:
                raise ValueError('Curve/per-image ordered FP32 accumulation mismatch')
        last = rows
    return curves[0], curves[1], last


def summarize(output, groups, baseline=None, write=False):
    import numpy as np
    reference = metrics(baseline) if baseline else None
    results, columns = {}, {'epoch': list(range(1, 21))}
    if reference:
        columns['V1_PSNR'], columns['V1_SSIM'] = reference[:2]
    for group in groups:
        p, s, rows = metrics(output/group)
        columns[group+'_PSNR'], columns[group+'_SSIM'] = p, s
        item = dict(final_psnr=float(p[-1]), final_ssim=float(s[-1]),
            best_psnr=float(p.max()), best_epoch=int(p.argmax()+1), last5_psnr=float(p[-5:].mean()),
            decision='PENDING_MATCHED_BASELINE', epochs=20)
        if reference:
            d = np.array([r['psnr']-b['psnr'] for r,b in zip(rows,reference[2])])
            rng = np.random.default_rng(20261008)
            means = d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
            columns[group+'_reference_delta'] = p-reference[0]
            item.update(reference_final_delta=float(p[-1]-reference[0][-1]),
                reference_last5_delta=float((p-reference[0])[-5:].mean()),
                reference_positive_epoch_ratio=float(((p-reference[0])>0).mean()),
                reference_per_image_mean=float(d.mean()), reference_median=float(np.median(d)),
                reference_win_rate=float((d>0).mean()), reference_bootstrap95=np.quantile(means,[.025,.975]).tolist(),
                reference_ssim_delta=float(s[-1]-reference[1][-1]),
                decision='REFERENCE_ONLY_PROVENANCE_PENDING')
        results[group] = item
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(columns)
    for row in zip(*columns.values()):
        writer.writerow(row)
    payload = dict(purpose='20_EPOCH_CANDIDATE_COLLECTION', baseline_reuse='NOT_APPROVED',
        baseline=str(baseline) if baseline else None, results=results,
        warning='Reference deltas are not audited structural gains; no automatic GO or extension')
    if write:
        for name, content in [('summary.json',json.dumps(payload,indent=2)), ('epoch_comparison.csv',text.getvalue())]:
            path = output/name
            if path.exists():
                raise FileExistsError(path)
            path.write_text(content,encoding='utf-8')
    print(text.getvalue(),flush=True)
    print(json.dumps(payload,indent=2),flush=True)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--groups',nargs='+',choices=('n21','n22','n23'),default=['n21','n22','n23'])
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--write',action='store_true')
    args = parser.parse_args()
    if args.baseline:
        from v1_combo_protocol import validate_v1_config
        validate_v1_config(args.baseline/'config.txt')
    summarize(args.output,args.groups,args.baseline,args.write)
