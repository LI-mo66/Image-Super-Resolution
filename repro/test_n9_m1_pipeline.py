"""Exercise summary gates and reject invalid evidence using synthetic metrics."""
import argparse
from pathlib import Path
import tempfile
import torch
from run_n9_m1_server import EXPECTED, audit_baseline, curve, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    audit_baseline(baseline)
    with tempfile.TemporaryDirectory(prefix='n9_pipeline_') as temp:
        output = Path(temp)
        m1 = output / 'm1'
        (m1 / 'per_image_metrics').mkdir(parents=True)
        (output / 'b0_eval/per_image_metrics').mkdir(parents=True)
        config = {**EXPECTED, 'model': 'LFMNPCSTR', 'epochs': '20', 'scheduler_t_max': '150'}
        config_path = m1 / 'config.txt'
        config_path.write_text('\n'.join(f'{k}: {v}' for k, v in config.items()), encoding='utf-8')
        base_p = curve(baseline, 'psnr_log.pt')[:20]
        base_s = curve(baseline, 'ssim_log.pt')[:20]

        def rows(psnr, ssim):
            return [{'dataset': 'DIV2K', 'scale': 4, 'filename': f'{i:04d}',
                     'psnr': float(psnr), 'ssim': float(ssim)} for i in range(801, 901)]

        torch.save(rows(base_p[19], base_s[19]),
                   output / 'b0_eval/per_image_metrics/epoch_0020.pt')
        for delta, expected in [(.02, 'EXTEND_ELIGIBLE'), (0, 'STOP')]:
            for metric, values in [('psnr', base_p + delta), ('ssim', base_s)]:
                torch.save(torch.tensor(values).reshape(20, 1, 1), m1 / f'{metric}_log.pt')
            for epoch in range(16, 21):
                torch.save(rows(base_p[epoch - 1] + delta, base_s[epoch - 1]),
                           m1 / f'per_image_metrics/epoch_{epoch:04d}.pt')
            result = summarize(output, baseline, args.data_root)
            assert result['decision'].startswith(expected)
        config['scheduler_t_max'] = '20'
        config_path.write_text('\n'.join(f'{k}: {v}' for k, v in config.items()), encoding='utf-8')
        try:
            summarize(output, baseline, args.data_root)
        except ValueError as error:
            assert 'protocol mismatch' in str(error)
        else:
            raise AssertionError('scheduler mismatch was accepted')
        config['scheduler_t_max'] = '150'
        config_path.write_text('\n'.join(f'{k}: {v}' for k, v in config.items()), encoding='utf-8')
        torch.save(rows(base_p[19] + 1, base_s[19]), m1 / 'per_image_metrics/epoch_0020.pt')
        try:
            summarize(output, baseline, args.data_root)
        except ValueError as error:
            assert 'mismatch' in str(error)
        else:
            raise AssertionError('forged per-image metrics were accepted')
    print('Synthetic pipeline tests PASS; no performance evidence produced')


if __name__ == '__main__':
    main()
