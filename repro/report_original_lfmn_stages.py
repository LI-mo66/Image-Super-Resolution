"""Read-only LFMN x4 milestone report; standard library only, no GPU/weights loading."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

SETS = ('Set5', 'Set14', 'B100', 'Urban100', 'Manga109')
MILESTONES = (1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def formal(config):
    return (config.get('smoke') is False and config.get('backend') == 'official'
            and config.get('scale') == 4 and config.get('model') == 'LFMN')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='experiment/all_runs')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    candidates = []
    for config_path in sorted(Path(args.root).glob('*/config.json')):
        config = read_json(config_path)
        if formal(config):
            candidates.append((config_path.parent, config))
    if args.list:
        for run, config in candidates:
            print('%s status=%s last_epoch=%s' %
                  (run.resolve(), config.get('status'), config.get('last_completed_epoch')))
        return
    if args.run:
        run = args.run.resolve()
        config = read_json(run / 'config.json')
    else:
        complete = [(p, c) for p, c in candidates if c.get('status') == 'complete'
                    and c.get('last_completed_epoch') == 1000]
        if len(complete) != 1:
            parser.error('Found %d completed formal runs; use --list then --run FULL_PATH. '
                         'Never select the two-epoch preflight.' % len(complete))
        run, config = complete[0]
        run = run.resolve()
    if not formal(config) or config.get('self_ensemble_enabled') is not False or \
            config.get('pretrained_checkpoint') not in (None, ''):
        raise ValueError('Not the official x4 scratch/no-ensemble formal run')
    if config.get('status') != 'complete' or config.get('last_completed_epoch') != 1000:
        raise ValueError('Run is not marked complete at epoch 1000')
    raw = (run / 'metrics.csv').read_bytes()
    rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if [int(r['epoch']) for r in rows] != list(range(1, 1001)):
        raise ValueError('CSV must contain exactly epochs 1..1000 without gaps/duplicates')
    import math
    for row in rows:
        if int(row['global_step']) != 1000 * int(row['epoch']):
            raise ValueError('Unexpected global_step at epoch ' + row['epoch'])
        for name in SETS:
            for metric in ('psnr', 'ssim'):
                if not math.isfinite(float(row[name + '_' + metric])):
                    raise ValueError('Nonfinite metric')
        if abs(float(row['selection_psnr']) - sum(float(row[n+'_psnr']) for n in SETS)/5) > 1e-8:
            raise ValueError('Selection score inconsistent with five dataset means')
    score = lambda row: float(row['selection_psnr'])
    best = max(rows, key=score)  # earliest row wins exact ties, same as trainer
    final = rows[-1]
    if int(config['best_epoch']) != int(best['epoch']):
        raise ValueError('config best_epoch differs from CSV; return both files for audit')
    summary_path = run / 'summary.json'
    if summary_path.exists():
        saved = read_json(summary_path)
        if saved.get('best_epoch') != int(best['epoch']) or saved.get('final_epoch') != 1000:
            raise ValueError('summary.json epoch mismatch')
        for name in SETS:
            for metric in ('psnr', 'ssim'):
                for key, row in [('selected_checkpoint_metrics', best), ('final_metrics', final)]:
                    if abs(float(saved[key][name][metric])-float(row[name+'_'+metric])) > 1e-8:
                        raise ValueError('summary.json metrics differ from CSV')
    print('=== LFMN_X4_STAGE_REPORT_BEGIN ===')
    print('RUN:', run)
    print('COMMIT:', config.get('git_commit'), 'PROTOCOL:', config.get('protocol_fingerprint'))
    print('DATA:', config.get('data_fingerprint'), 'GPU:', config.get('gpu_name'))
    print('CSV_SHA256:', hashlib.sha256(raw).hexdigest())
    print('CHECK: official binary; scratch; x4; no self-ensemble; 1000 epochs / 1000000 updates')
    print('\n[Fixed milestone PSNR: actual scores at that epoch, NOT historical best]')
    print('epoch,global_step,LR,' + ','.join(SETS) + ',mean5,elapsed_hours')
    def emit(row):
        print(','.join([row['epoch'], row['global_step'], row['learning_rate']] +
                       ['%.8f' % float(row[n+'_psnr']) for n in SETS] +
                       ['%.8f' % score(row), '%.4f' % (float(row['elapsed_seconds'])/3600)]))
    for epoch in MILESTONES:
        emit(rows[epoch-1])
    print('\n[LR phases: endpoint and one common best epoch per phase; descriptive only]')
    for low in (1, 201, 401, 601, 801):
        high = low+199
        phase = rows[low-1:high]
        print('PHASE %d-%d END:' % (low, high))
        emit(phase[-1])
        print('PHASE %d-%d COMMON_BEST:' % (low, high))
        emit(max(phase, key=score))
    print('\n[Unified best: all five datasets from ONE epoch %s]' % best['epoch'])
    print('dataset,best_psnr,best_ssim,final_psnr,final_ssim,independent_psnr_best,epoch,ssim_at_that_epoch')
    for name in SETS:
        independent = max(rows, key=lambda row: float(row[name+'_psnr']))
        print('%s,%.8f,%.8f,%.8f,%.8f,%.8f,%s,%.8f' %
              (name, float(best[name+'_psnr']), float(best[name+'_ssim']),
               float(final[name+'_psnr']), float(final[name+'_ssim']),
               float(independent[name+'_psnr']), independent['epoch'], float(independent[name+'_ssim'])))
    print('\n[Checkpoint files: presence/size only, contents not deserialized]')
    for name in ('latest.pt', 'best.pt', 'final.pt'):
        path = run / 'checkpoints' / name
        print(name, path.stat().st_size if path.exists() else 'MISSING')
    print('Training+validation hours:', float(final['elapsed_seconds'])/3600)
    print('Timing excludes initial data checks, checkpoint/plot overhead and pauses.')
    print('Independent dataset bests are diagnostic; do not splice them into one model result.')
    print('Intermediate epoch scores come from CSV; intermediate model weights may not be retained.')
    print('=== LFMN_X4_STAGE_REPORT_END ===')


if __name__ == '__main__':
    main()
