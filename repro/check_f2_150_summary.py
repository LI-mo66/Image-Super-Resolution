#!/usr/bin/env python3
"""Synthetic engineering checks only: no training or measured research results."""
import json
import math
from pathlib import Path
import tempfile
import torch
from f2_150_common import ROOT, MODELS, BENCHMARKS, PROTOCOL, read_csv, write_csv, sha256_file
from summarize_f2_150 import summarize, write_json


def fixture(group):
    group.mkdir()
    common = dict(protocol=PROTOCOL, commit='synthetic-engineering-commit',
                  source_sha256={'synthetic.py': 'a' * 64}, data_root='SYNTHETIC_NOT_REAL_DATA', workers=4,
                  environment={'python': 'synthetic', 'torch': 'synthetic', 'gpu': 'synthetic-single-gpu'},
                  dataset_fingerprint='d' * 64, engineering_only=False)
    manifest = dict(common, runs={label: label + '_train_x4_seed1_SYNTHETIC' for label in MODELS})
    write_json(group / 'protocol.json', manifest)
    evaluation_paths = {}
    for label, model in MODELS.items():
        directory = group / manifest['runs'][label]
        (directory / 'model').mkdir(parents=True)
        (directory / 'per_image_metrics').mkdir()
        write_json(directory / 'config.json', dict(common, status='completed', model=model,
                   parameter_count=759627, initial_unchanged_state_sha256='i' * 64,
                   authorized_experiment='F2_B0_INDEPENDENT_COS150', synthetic_fixture=True))
        metrics, validation, set5, details, batches, hashes = [], [], [], [], [], {}
        for epoch in range(1, 151):
            checkpoint = directory / 'model' / ('model_{}.pt'.format(epoch))
            torch.save({'synthetic_placeholder': torch.tensor([epoch])}, checkpoint)
            digest = hashes[epoch] = sha256_file(checkpoint)
            # Independent DIV2K best is147; observed Set5 best is150.
            gain = .06 if label == 'F2' else 0
            div_value = 30 - abs(epoch - 147) * .001 + gain
            five_value = 28 + epoch * .001 + gain
            raw = [dict(dataset='DIV2K', scale=4, filename=str(i).zfill(4),
                        psnr=div_value + (i - 850) * .0001, ssim=.85 + gain * .001) for i in range(801, 901)]
            raw += [dict(dataset='Set5', scale=4, filename='set5_{}'.format(i),
                         psnr=five_value + i * .0001, ssim=.8 + gain * .001) for i in range(5)]
            torch.save(raw, directory / 'per_image_metrics' / ('epoch_{:04d}.pt'.format(epoch)))
            div = raw[:100]; five = raw[100:]
            dpsnr = sum(r['psnr'] for r in div) / 100
            dssim = sum(r['ssim'] for r in div) / 100
            fpsnr = sum(r['psnr'] for r in five) / 5
            fssim = sum(r['ssim'] for r in five) / 5
            info = dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                        scale=4, self_ensemble=False)
            validation.append(dict(info, validation_psnr=dpsnr, validation_ssim=dssim, image_count=100))
            set5.append(dict(info, set5_psnr=fpsnr, set5_ssim=fssim, image_count=5))
            details.extend(dict(info, filename=r['filename'], psnr=r['psnr'], ssim=r['ssim']) for r in five)
            lr = PROTOCOL['eta_min'] + (PROTOCOL['lr'] - PROTOCOL['eta_min']) * (1 + math.cos(math.pi * (epoch - 1) / 150)) / 2
            metrics.append(dict(epoch=epoch, learning_rate=lr, train_loss=.1,
                                validation_psnr=float(torch.tensor(dpsnr, dtype=torch.float32)),
                                validation_ssim=float(torch.tensor(dssim, dtype=torch.float32)), elapsed_seconds=1))
            batches.append(dict(epoch=epoch, lr_sha256='l' * 64, hr_sha256='h' * 64,
                                batch_size=4, lr_shape='(4, 3, 64, 64)', hr_shape='(4, 3, 256, 256)'))
        for name, rows in (('metrics', metrics), ('validation_per_checkpoint', validation),
                           ('set5_per_checkpoint', set5), ('set5_per_image', details), ('first_batch_hashes', batches)):
            write_csv(directory / (name + '.csv'), rows)
        evaluation_paths[label] = {}
        for selection, epoch in (('final150', 150), ('best_div2k', 147)):
            target = group / (label + '_' + selection)
            target.mkdir()
            digest = hashes[epoch]
            write_json(target / 'config.json', dict(common, status='completed', model=model,
                       test_only=True, epoch=epoch, checkpoint_sha256=digest, selection=selection, synthetic_fixture=True))
            images, means = [], []
            for dataset, count in BENCHMARKS.items():
                rows = [dict(dataset=dataset, scale=4, filename=dataset + '_{}'.format(i),
                             psnr=29 + gain + i * .0001, ssim=.85 + gain * .001) for i in range(count)]
                images.extend(rows)
                means.append(dict(dataset=dataset, image_count=count, epoch=epoch, selection=selection,
                                  checkpoint_sha256=digest, psnr=sum(r['psnr'] for r in rows) / count,
                                  ssim=sum(r['ssim'] for r in rows) / count))
            write_csv(target / 'per_image_metrics.csv', images)
            write_csv(target / 'benchmark_means.csv', means)
            evaluation_paths[label][selection] = dict(directory=target.name, epoch=epoch, checkpoint_sha256=digest)
    write_json(group / 'evaluation_paths.json', evaluation_paths)
    return manifest


def main():
    parent = ROOT / 'experiment'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC_f2_150_summary_', dir=parent) as temporary:
        group = Path(temporary) / 'group'
        manifest = fixture(group)
        report, report_directory = summarize(group)
        assert report['status'] == 'validated_complete150'
        assert report['phase150_gate'] == 'PRIORITY_MULTISEED'
        assert all(r['best_div2k_epoch'] == 147 for r in report['identities'])
        assert len(read_csv(report_directory / 'comparison_metrics.csv')) == 150
        latest = (group / 'comparison_latest.json').read_bytes()
        report_count = len(list((group / 'comparison_reports').iterdir()))
        print('PASS synthetic complete150, matched data/environment/seed/batches; DIV2K best147 despite Set5 best150', flush=True)

        def reject(name, path, mutate):
            original = path.read_bytes()
            try:
                mutate(path)
                try:
                    summarize(group)
                except (ValueError, FileNotFoundError, KeyError, RuntimeError):
                    pass
                else:
                    raise AssertionError('Accepted invalid fixture: ' + name)
                assert (group / 'comparison_latest.json').read_bytes() == latest
                assert len(list((group / 'comparison_reports').iterdir())) == report_count
                print('PASS reject ' + name + '; preserved successful report', flush=True)
            finally:
                path.write_bytes(original)

        def csv_change(path, key, value):
            rows = read_csv(path); rows[0][key] = value; write_csv(path, rows)

        b0 = group / manifest['runs']['B0']
        f2 = group / manifest['runs']['F2']
        reject('missing epoch', b0 / 'metrics.csv', lambda p: write_csv(p, read_csv(p)[:-1]))
        reject('wrong checkpoint hash', b0 / 'validation_per_checkpoint.csv', lambda p: csv_change(p, 'checkpoint_sha256', 'x' * 64))
        reject('nonfinite metric', b0 / 'metrics.csv', lambda p: csv_change(p, 'train_loss', 'nan'))
        reject('paired first batch differs', f2 / 'first_batch_hashes.csv', lambda p: csv_change(p, 'lr_sha256', 'z' * 64))

        def json_change(path, key, value):
            value_json = json.loads(path.read_text(encoding='utf-8')); value_json[key] = value; write_json(path, value_json)

        reject('environment mismatch', f2 / 'config.json', lambda p: json_change(p, 'environment', {'gpu': 'different'}))
        reject('seed mismatch', f2 / 'config.json', lambda p: json_change(p, 'protocol', dict(PROTOCOL, seed=2)))
        reject('dataset mismatch', f2 / 'config.json', lambda p: json_change(p, 'dataset_fingerprint', 'x' * 64))

        def wrong_best(path):
            data = json.loads(path.read_text(encoding='utf-8'))
            data['F2']['best_div2k'] = data['F2']['final150']
            write_json(path, data)

        reject('Set5-best/wrong epoch selected', group / 'evaluation_paths.json', wrong_best)
        reject('mixed benchmark checkpoint', group / 'F2_final150' / 'benchmark_means.csv',
               lambda p: csv_change(p, 'checkpoint_sha256', 'x' * 64))
        reject('missing checkpoint', b0 / 'model' / 'model_150.pt', lambda p: p.unlink())
        reject('wrong cosine schedule', b0 / 'metrics.csv', lambda p: csv_change(p, 'learning_rate', '.0001'))
        # Move only the two independent DIV2K values so best becomes150,
        # then reuse exactly the primary evaluation artifacts.
        for label in MODELS:
            directory = group / manifest['runs'][label]
            valid = read_csv(directory / 'validation_per_checkpoint.csv')
            logged = read_csv(directory / 'metrics.csv')
            raw_paths = {e: directory / 'per_image_metrics' / ('epoch_{:04d}.pt'.format(e)) for e in (147, 150)}
            raws = {e: torch.load(p, weights_only=True) for e, p in raw_paths.items()}
            first = [r['psnr'] for r in raws[147] if r['dataset'] == 'DIV2K']
            second = [r['psnr'] for r in raws[150] if r['dataset'] == 'DIV2K']
            for epoch, values in ((147, second), (150, first)):
                rows = [r for r in raws[epoch] if r['dataset'] == 'DIV2K']
                for row, value in zip(rows, values):
                    row['psnr'] = value
                torch.save(raws[epoch], raw_paths[epoch])
                value = sum(values) / 100
                valid[epoch - 1]['validation_psnr'] = value
                logged[epoch - 1]['validation_psnr'] = float(torch.tensor(value, dtype=torch.float32))
            write_csv(directory / 'validation_per_checkpoint.csv', valid)
            write_csv(directory / 'metrics.csv', logged)
        paths = json.loads((group / 'evaluation_paths.json').read_text(encoding='utf-8'))
        for label in MODELS:
            paths[label]['best_div2k'] = dict(paths[label]['final150'])
        write_json(group / 'evaluation_paths.json', paths)
        reused, _ = summarize(group)
        assert all(reused['secondary_reused_final'].values())
        assert all(row['best_div2k_epoch'] == 150 for row in reused['identities'])
        print('PASS exact final150 reuse when independent DIV2K best is150', flush=True)
    print('PASS all synthetic engineering fixtures cleaned; these are not measured PSNR results', flush=True)

if __name__ == '__main__':
    main()
