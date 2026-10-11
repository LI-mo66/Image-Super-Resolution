"""CPU-only logger smoke gates; no LFMN or real training data are used."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout, redirect_stderr

from p_adapt_logging import RunLogger, REQUIRED_CONFIG, COLUMNS


def _config():
    return dict(model='CPU_Toy', model_source='check_p_adapt_logging.py', device='cpu',
                epochs=2, steps_per_epoch=1, batch_size=1, patch_size=1, optimizer='SGD',
                learning_rate=.01, scheduler='constant', loss='MSE', world_size=1,
                protocol_fingerprint='toy-fixed-protocol', batch_plan_sha256='toy-plan',
                initial_checkpoint_sha256='unknown', dataset='synthetic CPU tensors',
                train_range='synthetic', validation_range='synthetic', degradation='none',
                metric_protocol='engineering only', ema_enabled=False, self_ensemble_enabled=False)


def _read(path):
    return json.loads((path / 'config.json').read_text(encoding='utf-8'))


def run_checks(output_root=None):
    import torch
    root = Path(output_root) if output_root else Path(tempfile.mkdtemp(prefix='p_adapt_logging_'))
    root.mkdir(parents=True, exist_ok=True)
    passed, directories = [], []
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        logger = RunLogger(root, 'P0_logger_smoke', 4, 1, _config())
        with logger:
            first = logger.directory
            directories.append(str(first))
            print('STDOUT_LIVE_SENTINEL')
            print('STDERR_LIVE_SENTINEL', file=sys.stderr)
            live = (first / 'train_log.txt').read_text(encoding='utf-8')
            assert 'STDOUT_LIVE_SENTINEL' in live and 'STDERR_LIVE_SENTINEL' in live
            assert 'STDOUT_LIVE_SENTINEL' in stdout.getvalue()
            assert 'STDERR_LIVE_SENTINEL' in stderr.getvalue()
            passed.append('stdout/stderr terminal and immediate live-file capture')
            sys.stderr.write('\rPROGRESS_OLD')
            sys.stderr.write('\rPROGRESS_FINAL\n')
            progress_log = (first / 'train_log.txt').read_text(encoding='utf-8')
            assert 'PROGRESS_FINAL' in progress_log and 'PROGRESS_OLD' not in progress_log
            passed.append('carriage-return progress refreshes folded in file')
            logger.record(dict(epoch=0, global_step=0, lr=.01, train_loss=None,
                               valpsnr=20.1234567890123, valssim=.9, is_best=True, elapsed=0.0))
            logger.record(dict(epoch=1, global_step=1, learning_rate=.01, train_loss=.12,
                               validation_psnr=20.2345678901234, validation_ssim=.91,
                               is_best=True, elapsed_seconds=.01))
            try:
                logger.record(dict(epoch=1, global_step=1))
            except ValueError:
                passed.append('duplicate epoch rejected')
            else:
                raise AssertionError('Duplicate epoch accepted')
            # Check single-owner lock with an independent process.
            code = ('import sys; from p_adapt_logging import RunLogger; '
                    'RunLogger(sys.argv[1],"P0_logger_smoke",4,1,{},resume_dir=sys.argv[2]).__enter__()')
            child = subprocess.run([sys.executable, '-c', code, str(root), str(first)],
                                   cwd=str(Path(__file__).resolve().parent), capture_output=True, text=True)
            assert child.returncode != 0 and 'writer' in child.stderr
            passed.append('concurrent writer rejected')
        original = {p.name: p.read_bytes() for p in first.iterdir() if p.is_file()}
        with RunLogger(root, 'P0_logger_smoke', 4, 1, _config()) as second_logger:
            second = second_logger.directory
            directories.append(str(second))
            assert second != first
        assert all((first / name).read_bytes() == data for name, data in original.items())
        passed.append('unique directories without modifying previous run')
        cfg = _read(first)
        assert set(REQUIRED_CONFIG) <= set(cfg) and cfg['status'] == 'completed'
        assert cfg['protocol_fingerprint'] == 'toy-fixed-protocol'
        assert cfg['batch_plan_sha256'] == 'toy-plan'
        assert cfg['initial_checkpoint_sha256'] == 'unknown'
        with (first / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == COLUMNS
            rows = list(reader)
        assert rows[0]['train_loss'] == '' and float(rows[1]['validation_psnr']) == 20.2345678901234
        passed.append('required manifest fields, full precision and missing-value CSV')
        checkpoint = first / 'checkpoints' / 'toy_checkpoint.pt'
        checkpoint.write_bytes(b'logger-only-placeholder; not loaded as torch model')
        resume_cfg = _config()
        resume_cfg.update(resume_checkpoint=str(checkpoint), resume_start_epoch=2, resume_start_step=1)
        before_log = (first / 'train_log.txt').read_bytes()
        before_csv = (first / 'metrics.csv').read_bytes()
        with RunLogger(root, 'P0_logger_smoke', 4, 1, resume_cfg, resume_dir=first) as resumed:
            resumed.record(dict(epoch=2, global_step=2, lr=.01, train_loss=.1,
                                valpsnr=20.4, valssim=.92, is_best=True, elapsed=.02))
        assert (first / 'train_log.txt').read_bytes().startswith(before_log)
        assert (first / 'metrics.csv').read_bytes().startswith(before_csv)
        cfg = _read(first)
        assert cfg['run_id'] == first.name and len(cfg['sessions']) == 2
        assert cfg['sessions'][-1]['resume_start_step'] == 1
        assert cfg['sessions'][-1]['resume_start_epoch'] == 2
        assert str(checkpoint.resolve()) == cfg['resume_checkpoint']
        assert 'SESSION 2 RESUME' in (first / 'train_log.txt').read_text(encoding='utf-8')
        passed.append('resume preserves run identity and appends session/metrics')
        bad = dict(resume_cfg, learning_rate=.02)
        try:
            with RunLogger(root, 'P0_logger_smoke', 4, 1, bad, resume_dir=first):
                pass
        except ValueError as exc:
            assert 'learning_rate' in str(exc)
        else:
            raise AssertionError('Changed resume protocol accepted')
        passed.append('resume protocol mismatch rejected')
        exact = root / 'exact_unique_dir'
        with RunLogger(root, 'P0_logger_smoke', 4, 1, _config(), output_directory=exact) as run:
            assert run.directory == exact.resolve()
        try:
            with RunLogger(root, 'P0_logger_smoke', 4, 1, _config(), output_directory=exact):
                pass
        except FileExistsError:
            passed.append('exact output directory cannot overwrite')
        else:
            raise AssertionError('Existing exact directory accepted')
        rank_before = os.environ.get('RANK')
        try:
            os.environ['RANK'] = '1'
            try:
                RunLogger(root, 'P0_logger_smoke', 4, 1, _config())
            except RuntimeError:
                passed.append('nonzero DDP rank refused')
            else:
                raise AssertionError('Rank 1 was allowed')
        finally:
            if rank_before is None:
                os.environ.pop('RANK', None)
            else:
                os.environ['RANK'] = rank_before
        # Identical CPU toy training with/without logger. No actual SR model/data.
        def step(use_logger):
            random.seed(731)
            torch.manual_seed(731)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            x = torch.tensor([[.1, .2, .3]])
            target = torch.tensor([[.2, -.1]])
            rng_before = torch.get_rng_state().clone()
            py_before = random.getstate()
            logger = RunLogger(root, 'P0_rng_check', 4, 731, _config()) if use_logger else None
            if logger:
                logger.__enter__()
            try:
                out = model(x)
                loss = (out - target).square().mean()
                loss.backward()
                optimizer.step()
                if logger:
                    logger.record(dict(epoch=1, global_step=1, lr=.01, train_loss=loss.item()))
                assert torch.equal(rng_before, torch.get_rng_state())
                assert py_before == random.getstate()
                return out.detach().clone(), loss.detach().clone(), [p.detach().clone() for p in model.parameters()]
            finally:
                if logger:
                    logger.__exit__(None, None, None)
        bare, logged = step(False), step(True)
        assert torch.equal(bare[0], logged[0]) and torch.equal(bare[1], logged[1])
        assert all(torch.equal(a, b) for a, b in zip(bare[2], logged[2]))
        passed.append('CPU toy single-step output/loss/parameters and RNG unchanged')
    # Exception must propagate to the subprocess exit status.
    child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fail-child', str(root)],
                           capture_output=True, text=True)
    assert child.returncode != 0
    failed_dir = Path(next(line[9:] for line in child.stdout.splitlines() if line.startswith('FAIL_DIR=')))
    failed = _read(failed_dir)
    failed_log = (failed_dir / 'train_log.txt').read_text(encoding='utf-8')
    assert failed['status'] == 'failed' and failed['exit_code'] != 0
    assert 'Traceback (most recent call last)' in failed_log and 'CONTROLLED_FAILURE' in failed_log
    passed.append('controlled exception retains traceback and nonzero process exit')
    with redirect_stdout(stdout), redirect_stderr(stderr):
        interrupted = RunLogger(root, 'P0_interrupt_check', 4, 1, _config())
        try:
            with interrupted:
                raise KeyboardInterrupt('CONTROLLED_INTERRUPT')
        except KeyboardInterrupt:
            pass
        assert _read(interrupted.directory)['status'] == 'interrupted'
        passed.append('interrupt recorded and rethrown')
    return dict(passed=True, checks=passed, output_root=str(root.resolve()),
                run_directories=directories, failed_directory=str(failed_dir),
                scope='pure CPU toy logging engineering; no SR/GPU performance evidence')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root')
    parser.add_argument('--fail-child')
    args = parser.parse_args()
    if args.fail_child:
        with RunLogger(args.fail_child, 'P0_controlled_failure', 4, 1, _config()) as logger:
            print('FAIL_DIR=' + str(logger.directory))
            raise RuntimeError('CONTROLLED_FAILURE')
    else:
        print(json.dumps(run_checks(args.output_root), ensure_ascii=False, indent=2))
