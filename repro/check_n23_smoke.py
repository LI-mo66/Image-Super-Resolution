"""Opt-in real-data optimizer smoke, not a PSNR experiment or ablation."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parsed = parser.parse_args()
    output = parsed.output.resolve()
    allowed = (ROOT / 'experiment' / 'all_runs').resolve()
    # No broad/root outputs or accidental reuse of completed experiments.
    output.relative_to(allowed)
    if output == allowed or output.exists():
        raise FileExistsError(f'Output must be a NEW child of {allowed}: {output}')
    data_root = parsed.data_root.resolve()
    for number in (1, 2):
        for relative in (f'DIV2K/DIV2K_train_HR/{number:04d}.png',
                         f'DIV2K/DIV2K_train_LR_bicubic/X4/{number:04d}x4.png'):
            if not (data_root / relative).is_file():
                raise FileNotFoundError(data_root / relative)
    for relative in ('DIV2K/DIV2K_valid_HR/0859.png',
                     'DIV2K/DIV2K_valid_LR_bicubic/X4/0859x4.png'):
        if not (data_root / relative).is_file():
            raise FileNotFoundError(data_root / relative)
    output.mkdir(parents=True)
    groups = {'b0': 'lfmn_exact_overlap', 'n23': 'lfmn_n23'}
    for group, model_name in groups.items():
        command = [sys.executable, str(ROOT / 'repro/n23_train_entry.py'),
                   '--model', model_name, '--dir_data', str(data_root),
                   '--experiment_root', str(output), '--save', group,
                   '--scale', '4', '--rgb_range', '255', '--patch_size', '256',
                   '--batch_size', '4', '--seed', '1', '--n_threads', '0',
                   '--n_GPUs', '1', '--data_train', 'DIV2K', '--data_test', 'DIV2K',
                   '--data_range', '1-2/859-859', '--ext', 'img', '--epochs', '1',
                   '--max_train_batches', '1', '--test_every', '1000',
                   '--scheduler', 'cosine', '--scheduler_t_max', '150', '--eta_min', '1e-6',
                   '--lr', '2e-4', '--precision', 'single', '--loss', '1*L1',
                   '--save_per_image_metrics', '--print_every', '1']
        if parsed.cpu:
            command.append('--cpu')
        with (output / f'{group}_console.log').open('x', encoding='utf-8') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f'{group}: framework one-update/full-image validation complete', flush=True)

    import numpy as np
    import imageio.v2 as imageio
    import torch
    sys.path.insert(0, str(ROOT / 'LFMN'))
    from data import common
    import utility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    device = torch.device('cpu' if parsed.cpu else 'cuda')
    pair = [imageio.imread(data_root / p) for p in (
        'DIV2K/DIV2K_valid_LR_bicubic/X4/0859x4.png', 'DIV2K/DIV2K_valid_HR/0859.png')]
    lr_np, hr_np = pair
    hr_np = hr_np[:lr_np.shape[0]*4, :lr_np.shape[1]*4]
    lr, hr = common.np2Tensor(*common.set_channel(lr_np, hr_np, n_channels=3), rgb_range=255)
    lr, hr = lr[None].to(device), hr[None].to(device)
    summaries = {}
    proofs, batches = [], []
    with torch.no_grad():
        for group, model_name in groups.items():
            directory = output / group
            proof = json.loads((directory / 'initial_state_proof.json').read_text())
            batch = json_lines(directory / 'batch_fingerprints.jsonl')
            mechanism = json_lines(directory / 'mechanism.jsonl')
            capture = json_lines(directory / 'eval_output_fingerprints.jsonl')
            assert len(batch) == len(mechanism) == len(capture) == 1
            assert batch[0]['lr_shape'] == [4, 3, 64, 64]
            assert batch[0]['hr_shape'] == [4, 3, 256, 256]
            assert mechanism[0]['updates'] == mechanism[0]['total_updates'] == 1
            if group == 'n23':
                assert mechanism[0]['scc_nonzero_gradients_last_step'] == 208
                assert mechanism[0]['scc_parameters_changed'] > 0
            checkpoint = directory / 'model/model_1.pt'
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            net = importlib.import_module(f'model.{model_name}').Net(scale=4)
            net.load_state_dict(state, strict=True)
            net = net.to(device).eval()
            raw = net(lr)
            assert torch.isfinite(raw).all() and raw.shape == hr.shape
            assert digest(lr) == capture[0]['input_sha256']
            assert digest(raw) == capture[0]['output_sha256'], 'Saved model does not reproduce framework output'
            quantized = utility.quantize(raw, 255)
            psnr = float(utility.calc_psnr(quantized, hr, 4, 255))
            ssim = float(utility.calc_ssim(quantized, hr, 4, 255))
            saved = torch.load(directory / 'per_image_metrics/epoch_0001.pt',
                               map_location='cpu', weights_only=True)
            assert len(saved) == 1 and saved[0]['filename'] == '0859'
            assert saved[0]['psnr'] == psnr and saved[0]['ssim'] == ssim
            for filename, metric in [('psnr_log.pt', psnr), ('ssim_log.pt', ssim)]:
                curve = torch.load(directory / filename, map_location='cpu', weights_only=True)
                assert curve.shape == (1, 1, 1) and torch.isfinite(curve).all()
                assert float(curve.item()) == float(np.float32(metric))
            # A second independent strict load checks state roundtrip, not just repeated inference.
            reloaded = importlib.import_module(f'model.{model_name}').Net(scale=4)
            reloaded.load_state_dict(state, strict=True)
            reloaded = reloaded.to(device).eval()
            duplicate = reloaded(lr)
            assert torch.equal(raw, duplicate), 'Independent strict reload mismatch'
            summaries[group] = {'updates': 1, 'strict_reload_raw_max_error': 0.0,
                                'framework_output_and_metrics_reproduced': True,
                                'mechanism': mechanism[0]}
            proofs.append(proof)
            batches.append(batch[0])
            del net, reloaded, raw, duplicate, quantized, state
            if not parsed.cpu:
                torch.cuda.empty_cache()
    for key in ('common_initial_state_sha256', 'retained_state_sha256', 'cpu_rng_sha256'):
        assert proofs[0][key] == proofs[1][key], key
    for key in ('lr_sha256', 'hr_sha256'):
        assert batches[0][key] == batches[1][key], key
    report = {'status': 'REAL_FRAMEWORK_SMOKE_PASS_NOT_ACCURACY_EVIDENCE',
              'data_range': '1-2/859-859', 'device': str(device), 'groups': summaries}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
