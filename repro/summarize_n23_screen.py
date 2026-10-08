#!/usr/bin/env python3
"""Strict N23/B0 short-screen summary; engineering repairs are not SR gains."""

import argparse
import ast
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


EPOCHS = 40
MEAN_TOLERANCE = 1e-5
BOOTSTRAP_SEED = 23
BOOTSTRAP_REPETITIONS = 10000
IMAGE_IDS = tuple(f'{i:04d}' for i in range(801, 901))
THRESHOLDS = {
    'final_delta': 0.010,
    'last5_mean_delta': 0.010,
    'bootstrap95_lower_strictly_above': 0.0,
    'per_image_median_strictly_above': 0.0,
    'win_rate': 0.60,
    'ssim_delta': -1e-4,
}


def checked_curve(value, label):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[1:] != (1, 1):
        raise ValueError(f'{label}: expected [epochs,1,1], found {value.shape}')
    if len(value) > EPOCHS or not np.isfinite(value).all():
        raise ValueError(f'{label}: excess epochs or non-finite metrics')
    return value[:, 0, 0].astype(np.float64)


def checked_images(rows, label):
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError(f'{label}: expected a list of 100 image records')
    result = {}
    for row in rows:
        if row.get('dataset') != 'DIV2K' or row.get('scale') != 4:
            raise ValueError(f'{label}: dataset/scale mismatch')
        name = row.get('filename')
        if name not in IMAGE_IDS or name in result:
            raise ValueError(f'{label}: unexpected/duplicate image ID {name!r}')
        metrics = (float(row['psnr']), float(row['ssim']))
        if not np.isfinite(metrics).all():
            raise ValueError(f'{label}: non-finite per-image metrics')
        result[name] = metrics
    if set(result) != set(IMAGE_IDS):
        raise ValueError(f'{label}: missing expected image IDs')
    return np.asarray([result[name] for name in IMAGE_IDS], dtype=np.float64)


def bootstrap_ci(delta):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(delta), (BOOTSTRAP_REPETITIONS, len(delta)))
    means = delta[indices].mean(axis=1)
    return np.quantile(means, (0.025, 0.975)).tolist()


def trainer_float32_mean(values):
    """Reproduce Trainer.test's ordered CPU float32 accumulation, not f64 mean."""
    accumulator = torch.zeros((), dtype=torch.float32)
    for value in values:
        accumulator += float(value)
    accumulator /= len(values)
    return float(accumulator.item())


def compare_curves(curves, b0_images=None, n23_images=None):
    """Pure metric comparison; callers must separately audit protocols."""
    checked = {}
    for group in ('b0', 'n23'):
        checked[group] = {
            metric: checked_curve(curves[group][metric], f'{group}/{metric}')
            for metric in ('psnr', 'ssim')
        }
        if len(checked[group]['psnr']) != len(checked[group]['ssim']):
            raise ValueError(f'{group}: PSNR/SSIM epoch count mismatch')
    counts = {group: len(values['psnr']) for group, values in checked.items()}
    result = {
        'decision': 'DIAGNOSTIC_INCOMPLETE',
        'completed_epochs': counts,
        'required_epochs': EPOCHS,
        'direction': 'n23_minus_b0',
        'thresholds': THRESHOLDS.copy(),
        'performance_claim': False,
    }
    common = min(counts.values())
    if common:
        delta = checked['n23']['psnr'][:common] - checked['b0']['psnr'][:common]
        result['paired_epochs'] = common
        result['epoch_deltas'] = delta.tolist()
    if any(count != EPOCHS for count in counts.values()):
        result['reason'] = 'Both groups must finish exactly 40 epochs; no promotion.'
        return result, checked
    if b0_images is None or n23_images is None:
        raise ValueError('complete screen requires epoch_0040 per-image records')
    b0 = checked_images(b0_images, 'b0/images')
    n23 = checked_images(n23_images, 'n23/images')
    rounding = {}
    for group, images in (('b0', b0), ('n23', n23)):
        rounding[group] = {}
        for column, metric in enumerate(('psnr', 'ssim')):
            reference = float(checked[group][metric][-1])
            mean64 = float(images[:, column].mean())
            original = curves[group][metric]
            if isinstance(original, torch.Tensor) and original.dtype == torch.float32:
                reconstructed = trainer_float32_mean(images[:, column])
                if reconstructed != reference:
                    raise ValueError(
                        f'{group}: ordered Trainer float32 {metric} mean/curve mismatch '
                        f'{reconstructed} != {reference}')
                rounding[group][metric] = {
                    'validation': 'EXACT_ORDERED_TRAINER_FLOAT32',
                    'reconstructed_curve': reconstructed,
                    'float64_mean_minus_stored_curve': mean64 - reference,
                }
            else:
                error = abs(mean64 - reference)
                if error > MEAN_TOLERANCE:
                    raise ValueError(f'{group}: per-image {metric} mean/curve mismatch {error}')
                rounding[group][metric] = {
                    'validation': 'FLOAT64_MEAN_TOLERANCE',
                    'float64_mean_minus_stored_curve': mean64 - reference,
                }
    image_delta = n23[:, 0] - b0[:, 0]
    ci = bootstrap_ci(image_delta)
    result.update({
        'b0_final_psnr': float(checked['b0']['psnr'][-1]),
        'n23_final_psnr': float(checked['n23']['psnr'][-1]),
        'final_delta': float(delta[-1]),
        'last5_mean_delta': float(delta[-5:].mean()),
        'positive_epoch_ratio': float((delta > 0).mean()),
        'per_image_mean': float(image_delta.mean()),
        'per_image_median': float(np.median(image_delta)),
        'win_rate': float((image_delta > 0).mean()),
        'bootstrap95': ci,
        'ssim_delta': float(checked['n23']['ssim'][-1] - checked['b0']['ssim'][-1]),
        'per_image_ssim_delta': float((n23[:, 1] - b0[:, 1]).mean()),
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_repetitions': BOOTSTRAP_REPETITIONS,
        'bootstrap_unit': 'image; conditional on these 100 images and one run',
        'metric_rounding': rounding,
    })
    checks = {
        'final_delta': result['final_delta'] >= THRESHOLDS['final_delta'],
        'last5_mean_delta': result['last5_mean_delta'] >= THRESHOLDS['last5_mean_delta'],
        'bootstrap95_lower': ci[0] > 0,
        'per_image_median': result['per_image_median'] > 0,
        'win_rate': result['win_rate'] >= THRESHOLDS['win_rate'],
        'ssim_delta': result['ssim_delta'] >= THRESHOLDS['ssim_delta'],
    }
    result['gates'] = checks
    result['decision'] = 'GO' if all(checks.values()) else 'NO_GO'
    result['failed_gates'] = [name for name, passed in checks.items() if not passed]
    result['reason'] = 'Resource allocation only; no automatic extension or paper claim.'
    return result, checked


def audit_manifest(manifest, complete):
    protocol = manifest.get('protocol')
    if not isinstance(protocol, dict):
        raise ValueError('missing shared protocol manifest')
    expected = {'epochs': EPOCHS, 'overlap': 'exact_coverage_v1', 'mode': 'scratch'}
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'protocol {key} mismatch: {protocol.get(key)!r}')
    status = manifest.get('comparison_audit', {}).get('status')
    if status == 'FAIL' or (complete and status != 'PASS'):
        raise ValueError('complete GO/NO_GO requires comparison_audit.status=PASS')
    return protocol


def load_images(directory):
    folder = directory / 'per_image_metrics'
    pt = folder / 'epoch_0040.pt'
    if pt.is_file():
        return torch.load(pt, map_location='cpu', weights_only=True)
    for name in ('epoch_0040.json', 'epoch_40.json'):
        path = folder / name
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
    raise FileNotFoundError(pt)


def comparison_csv(checked):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(('epoch', 'B0_PSNR', 'N23_PSNR', 'N23_delta',
                     'B0_SSIM', 'N23_SSIM', 'N23_SSIM_delta'))
    maximum = max(len(values['psnr']) for values in checked.values())
    for index in range(maximum):
        values = {}
        for group in ('b0', 'n23'):
            for metric in ('psnr', 'ssim'):
                curve = checked[group][metric]
                values[group, metric] = float(curve[index]) if index < len(curve) else None
        paired = all(values[group, 'psnr'] is not None for group in ('b0', 'n23'))
        psnr_delta = values['n23', 'psnr'] - values['b0', 'psnr'] if paired else None
        ssim_delta = values['n23', 'ssim'] - values['b0', 'ssim'] if paired else None
        row = (index + 1, values['b0', 'psnr'], values['n23', 'psnr'], psnr_delta,
               values['b0', 'ssim'], values['n23', 'ssim'], ssim_delta)
        writer.writerow([row[0]] + ['' if value is None else f'{value:.10f}' for value in row[1:]])
    return stream.getvalue()


def normalized_config_value(value):
    """Safely compare literal configs with JSON's tuple-to-list conversion."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, (tuple, list)):
        return [normalized_config_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized_config_value(item) for key, item in value.items()}
    return value


def read_config(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        if key in result:
            raise ValueError(f'{path}: duplicate config key {key!r}')
        result[key] = value
    return result


def summarize(root, b0_dir=None, n23_dir=None, write=True):
    root = Path(root)
    directories = {'b0': Path(b0_dir) if b0_dir else root / 'b0',
                   'n23': Path(n23_dir) if n23_dir else root / 'n23'}
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    curves = {}
    for group, directory in directories.items():
        curves[group] = {}
        for metric in ('psnr', 'ssim'):
            path = directory / f'{metric}_log.pt'
            curves[group][metric] = (
                torch.load(path, map_location='cpu', weights_only=True)
                if path.is_file() else torch.empty(0, 1, 1)
            )
    complete = all(checked_curve(values[metric], f'{group}/{metric}').size == EPOCHS
                   for group, values in curves.items() for metric in ('psnr', 'ssim'))
    protocol = audit_manifest(manifest, complete)
    for group, directory in directories.items():
        path = directory / 'evaluation_protocol.json'
        if not path.is_file():
            if complete and group == 'b0':
                raise FileNotFoundError(path)
            continue
        recorded = json.loads(path.read_text(encoding='utf-8'))
        if recorded.get('protocol') != protocol:
            raise ValueError(f'{group}: evaluation_protocol differs from shared manifest')
    # N23 has the actual training config; reused B0's original config is not
    # rewritten to pretend it used a new evaluation protocol.
    expected_config = protocol.get('expected_config', {})
    config_path = directories['n23'] / 'config.txt'
    if complete and expected_config:
        config = read_config(config_path)
        for key, value in expected_config.items():
            if key == 'model':
                continue  # Models differ intentionally; shared training fields do not.
            if key not in config or normalized_config_value(config[key]) != normalized_config_value(value):
                raise ValueError(f'n23/config: shared field {key} mismatch')
    images = {group: load_images(directory) for group, directory in directories.items()} if complete else {}
    result, checked = compare_curves(curves, images.get('b0'), images.get('n23'))
    result['protocol'] = protocol
    result['comparison_audit'] = manifest.get('comparison_audit', {})
    result['sources'] = {group: str(directory.resolve()) for group, directory in directories.items()}
    if write:
        with (root / 'epoch_comparison.csv').open('w', encoding='utf-8', newline='') as handle:
            handle.write(comparison_csv(checked))
        (root / 'decision.json').write_text(json.dumps(
            result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


class SummaryTests(unittest.TestCase):
    @staticmethod
    def fixture(delta=0.02, epochs=40, ssim_delta=0.0):
        curve = lambda value: np.full((epochs, 1, 1), value, dtype=np.float64)
        curves = {'b0': {'psnr': curve(28.0), 'ssim': curve(0.8)},
                  'n23': {'psnr': curve(28.0 + delta), 'ssim': curve(0.8 + ssim_delta)}}
        records = lambda p, s: [{'dataset': 'DIV2K', 'scale': 4, 'filename': name,
                                'psnr': p, 'ssim': s} for name in IMAGE_IDS]
        return curves, records(28.0, 0.8), records(28.0 + delta, 0.8 + ssim_delta)

    def test_positive_complete(self):
        result, checked = compare_curves(*self.fixture())
        self.assertEqual(result['decision'], 'GO')
        self.assertEqual(len(comparison_csv(checked).splitlines()), 41)

    def test_negative(self):
        self.assertEqual(compare_curves(*self.fixture(-0.02))[0]['decision'], 'NO_GO')

    def test_ssim_gate(self):
        result = compare_curves(*self.fixture(ssim_delta=-0.001))[0]
        self.assertIn('ssim_delta', result['failed_gates'])

    def test_tail_gate(self):
        curves, b0, n23 = self.fixture()
        curves['n23']['psnr'][-5:-1] = 28.0
        result = compare_curves(curves, b0, n23)[0]
        self.assertIn('last5_mean_delta', result['failed_gates'])
        self.assertTrue(result['gates']['final_delta'])

    def test_win_rate_gate(self):
        curves, b0, n23 = self.fixture(delta=0.0225)
        for index, row in enumerate(n23):
            row['psnr'] = 28.05 if index < 50 else 27.995
        result = compare_curves(curves, b0, n23)[0]
        self.assertIn('win_rate', result['failed_gates'])
        self.assertTrue(result['gates']['final_delta'])

    def test_zero_signal(self):
        result = compare_curves(*self.fixture(delta=0.0))[0]
        self.assertEqual(result['decision'], 'NO_GO')
        self.assertFalse(result['gates']['bootstrap95_lower'])
        self.assertFalse(result['gates']['per_image_median'])

    def test_incomplete(self):
        self.assertEqual(compare_curves(self.fixture(epochs=39)[0])[0]['decision'],
                         'DIAGNOSTIC_INCOMPLETE')

    def test_malformed_curve(self):
        for invalid in (np.zeros((40,)), np.zeros((41, 1, 1)), np.full((40, 1, 1), np.nan)):
            with self.assertRaises(ValueError):
                checked_curve(invalid, 'test')

    def test_bad_images(self):
        curves, b0, n23 = self.fixture()
        n23[0]['filename'] = '0802'
        with self.assertRaises(ValueError):
            compare_curves(curves, b0, n23)

    def test_mean_mismatch(self):
        curves, b0, n23 = self.fixture()
        n23[0]['psnr'] += 0.01
        with self.assertRaises(ValueError):
            compare_curves(curves, b0, n23)

    def test_nonfinite_image(self):
        curves, b0, n23 = self.fixture()
        n23[0]['ssim'] = float('inf')
        with self.assertRaises(ValueError):
            compare_curves(curves, b0, n23)

    def test_trainer_ordered_rounding_not_float64_mean(self):
        curves, b0, n23 = self.fixture()
        rng = np.random.default_rng(23100)
        for index in range(100):
            b0[index]['psnr'] = float(rng.uniform(20.0, 40.0))
            b0[index]['ssim'] = float(rng.uniform(0.65, 0.95))
            n23[index]['psnr'] = b0[index]['psnr'] + 0.02
            n23[index]['ssim'] = b0[index]['ssim']
        for group, rows in (('b0', b0), ('n23', n23)):
            for metric in ('psnr', 'ssim'):
                value = trainer_float32_mean([row[metric] for row in rows])
                curves[group][metric] = torch.full((40, 1, 1), value, dtype=torch.float32)
        result = compare_curves(curves, b0, n23)[0]
        self.assertEqual(result['decision'], 'GO')
        self.assertNotEqual(result['metric_rounding']['b0']['psnr']
                            ['float64_mean_minus_stored_curve'], 0.0)
        curves['n23']['psnr'][-1, 0, 0] += 1e-5
        with self.assertRaises(ValueError):
            compare_curves(curves, b0, n23)

    def test_protocol_guard(self):
        manifest = {'protocol': {'epochs': 40, 'overlap': 'exact_coverage_v1', 'mode': 'scratch'},
                    'comparison_audit': {'status': 'PASS'}}
        audit_manifest(manifest, True)
        manifest['protocol']['overlap'] = 'legacy'
        with self.assertRaises(ValueError):
            audit_manifest(manifest, True)
        manifest['protocol']['overlap'] = 'exact_coverage_v1'
        manifest['comparison_audit']['status'] = 'PENDING'
        with self.assertRaises(ValueError):
            audit_manifest(manifest, True)

    def test_file_integration_json_tuple_roundtrip(self):
        # Synthetic I/O test only: never experimental evidence or a trained run.
        curves, b0, n23 = self.fixture()
        scratch = Path(__file__).resolve().parents[1] / 'experiment' / 'all_runs'
        scratch.mkdir(parents=True, exist_ok=True)
        # Explicit workspace scratch avoids platform-dependent sandbox TEMP.
        with tempfile.TemporaryDirectory(prefix='n23_summary_unit_', dir=scratch) as temporary:
            root = Path(temporary)
            if not root.resolve().is_relative_to(scratch.resolve()):
                raise RuntimeError('temporary test path escaped workspace scratch')
            protocol = {'epochs': 40, 'overlap': 'exact_coverage_v1', 'mode': 'scratch',
                        'expected_config': {'epochs': 40, 'betas': (0.9, 0.999),
                                            'scale': [4], 'model': 'lfmn_exact_overlap'}}
            manifest = {'protocol': protocol, 'comparison_audit': {'status': 'PASS'}}
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            for group, images in (('b0', b0), ('n23', n23)):
                folder = root / group
                (folder / 'per_image_metrics').mkdir(parents=True)
                for metric in ('psnr', 'ssim'):
                    value = trainer_float32_mean([row[metric] for row in images])
                    torch.save(torch.full((40, 1, 1), value, dtype=torch.float32),
                               folder / f'{metric}_log.pt')
                torch.save(images, folder / 'per_image_metrics/epoch_0040.pt')
                (folder / 'evaluation_protocol.json').write_text(
                    json.dumps({'protocol': protocol}), encoding='utf-8')
            config_path = root / 'n23/config.txt'
            config_text = 'epochs: 40\nbetas: (0.9, 0.999)\nscale: [4]\nmodel: lfmn_n23\n'
            config_path.write_text(config_text, encoding='utf-8')
            result = summarize(root)
            self.assertEqual(result['decision'], 'GO')
            self.assertGreater(abs(result['metric_rounding']['n23']['psnr']
                                   ['float64_mean_minus_stored_curve']), MEAN_TOLERANCE)
            self.assertEqual(len((root / 'epoch_comparison.csv').read_text().splitlines()), 41)
            self.assertEqual(json.loads((root / 'decision.json').read_text())['decision'], 'GO')
            wrong = json.loads(json.dumps(protocol))
            wrong['overlap'] = 'legacy'
            (root / 'b0/evaluation_protocol.json').write_text(
                json.dumps({'protocol': wrong}), encoding='utf-8')
            with self.assertRaises(ValueError):
                summarize(root)
            (root / 'b0/evaluation_protocol.json').write_text(
                json.dumps({'protocol': protocol}), encoding='utf-8')
            config_path.write_text(config_text + 'betas: (0.9, 0.999)\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path)
    parser.add_argument('--b0-dir', type=Path)
    parser.add_argument('--n23-dir', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if args.root is None:
        parser.error('root is required unless --self-test is used')
    try:
        result = summarize(args.root, args.b0_dir, args.n23_dir)
    except (ValueError, FileNotFoundError, KeyError, TypeError) as error:
        raise SystemExit(f'INTEGRITY_ERROR_NO_DECISION: {error}') from error
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
