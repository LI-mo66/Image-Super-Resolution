"""Single-process paired-adaptation run logging without tensor or RNG operations."""
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

REQUIRED_CONFIG = (
    'run_id experiment_name model model_source start_time status command cwd '
    'git_commit git_dirty git_diff_summary dataset train_range validation_range '
    'scale degradation epochs steps_per_epoch batch_size patch_size optimizer '
    'learning_rate scheduler loss seed device gpu_name world_size ema_enabled '
    'self_ensemble_enabled pretrained_checkpoint resume_checkpoint resume_start_epoch '
    'metric_protocol output_directory end_time exit_code final_metrics best_metrics best_epoch'
).split()
COLUMNS = ['session', 'epoch', 'global_step', 'learning_rate', 'train_loss',
           'validation_psnr', 'validation_ssim', 'is_best', 'elapsed_seconds']
ALIASES = {'lr': 'learning_rate', 'valpsnr': 'validation_psnr',
           'valssim': 'validation_ssim', 'elapsed': 'elapsed_seconds'}
_DYNAMIC = {'command', 'cwd', 'resume_checkpoint', 'resume_start_epoch',
            'resume_start_step', 'resume_startstep', 'resume_start_global_step',
            'resume_optimizer_restored', 'resume_scheduler_restored', 'status',
            'start_time', 'end_time', 'exit_code', 'final_metrics', 'best_metrics',
            'best_epoch', 'sessions', 'output_directory', 'run_id', 'session',
            'git_dirty', 'git_diff_summary'}
_CURRENT = None


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Configuration must contain JSON values, not {type(value).__name__}')


def _plain(value):
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _atomic_json(path, value):
    # PID/thread-specific temp file; directory ownership prevents competing writers.
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class _Tee:
    def __init__(self, terminal, log, lock):
        self.terminal, self.log, self.lock = terminal, log, lock
        self._progress, self._pending = False, ''
        self._last_progress = time.monotonic()

    def write(self, text):
        with self.lock:
            result = self.terminal.write(text)
            self.terminal.flush()
            if '\r' not in text and not self._progress:
                self.log.write(text)
            else:
                # Keep terminal carriage-return rendering intact; collapse log
                # refreshes to the latest line, sampled at most every 5 seconds.
                self._progress = True
                parts = text.split('\n')
                for index, part in enumerate(parts):
                    self._pending = (self._pending + part).split('\r')[-1]
                    if index < len(parts) - 1:
                        self.log.write(self._pending + '\n')
                        self._pending = ''
                        self._progress = False
                if self._pending and time.monotonic() - self._last_progress >= 5:
                    self.log.write(self._pending + '\n')
                    self._last_progress = time.monotonic()
            self.log.flush()
            return result

    def finish_progress(self):
        with self.lock:
            if self._pending:
                self.log.write(self._pending + '\n')
                self.log.flush()
            self._pending, self._progress = '', False

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class RunLogger:
    """Use one context per process; all independent experiment groups own a directory.

    Entry configuration is validated before capture. Capture covers Python stdout
    and stderr (including warnings/tracebacks); subprocesses must pipe their output
    to these streams rather than inheriting raw OS descriptors.
    """
    def __init__(self, output_root, scheme, scale, seed, config, resume_dir=None, output_directory=None):
        for key in ('RANK', 'LOCAL_RANK', 'SLURM_PROCID'):
            if int(os.environ.get(key, '0')) != 0:
                raise RuntimeError('Paired adaptation logger supports rank 0 only')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise RuntimeError('Paired adaptation requires world_size=1')
        self.scheme, self.scale, self.seed = str(scheme), int(scale), int(seed)
        self.input_config = _plain(config)
        if self.input_config.get('world_size') not in (None, 1):
            raise ValueError('Only independent single-process runs are supported')
        self.root = Path(output_root).resolve()
        self.resume = resume_dir is not None
        if self.resume and output_directory is not None:
            raise ValueError('output_directory and resume_dir are mutually exclusive')
        self.exact_directory = Path(output_directory).resolve() if output_directory is not None else None
        self.directory = Path(resume_dir).resolve() if self.resume else None
        self.last_epoch, self.last_step = -1, -1
        self.config, self.session, self._lock_fd = None, None, None
        self._entered = False
        self.log = self.csv_stream = None

    def _create_directory(self):
        if self.exact_directory is not None:
            self.exact_directory.parent.mkdir(parents=True, exist_ok=True)
            self.exact_directory.mkdir(exist_ok=False)
            self.directory = self.exact_directory
            return
        self.root.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r'[^\w.-]+', '_', self.scheme).strip('._')
        if not safe:
            raise ValueError('Scheme must contain usable name characters')
        stem = f'{safe}_x{self.scale}_seed{self.seed}_{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        for suffix in range(100000):
            candidate = self.root / (stem if suffix == 0 else f'{stem}_{suffix:03d}')
            try:
                candidate.mkdir()
                self.directory = candidate
                return
            except FileExistsError:
                continue
        raise RuntimeError('Cannot allocate unique run directory')

    def _acquire(self):
        self._lock_path = self.directory / '.logger.lock'
        try:
            self._lock_fd = os.open(str(self._lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.write(self._lock_fd, f'{os.getpid()}\n'.encode())
        except FileExistsError as exc:
            raise RuntimeError(f'Run already has a writer (or stale lock): {self._lock_path}') from exc

    def _read_history(self):
        with (self.directory / 'metrics.csv').open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != COLUMNS:
                raise ValueError('Resume CSV schema differs from registered logging schema')
            for row in reader:
                epoch, step = int(row['epoch']), int(row['global_step'])
                if epoch <= self.last_epoch or step < self.last_step:
                    raise ValueError('Existing metrics timeline is not monotonic')
                self.last_epoch, self.last_step = epoch, step

    def _prepare_config(self):
        incoming = dict(self.input_config)
        for key in ('pretrained_checkpoint', 'resume_checkpoint'):
            if incoming.get(key):
                incoming[key] = str(Path(incoming[key]).resolve())
        for key, expected in [('experiment_name', self.scheme), ('scale', self.scale), ('seed', self.seed)]:
            if incoming.get(key) is not None and incoming[key] != expected:
                raise ValueError(f'Config {key} disagrees with constructor')
            incoming[key] = expected
        if self.resume:
            with (self.directory / 'config.json').open(encoding='utf-8') as stream:
                stored = json.load(stream)
            for key, value in incoming.items():
                if key not in _DYNAMIC and (key not in stored or stored[key] != value):
                    raise ValueError(f'Resume changes original configuration field: {key}')
            checkpoint = incoming.get('resume_checkpoint')
            if not checkpoint or not Path(checkpoint).is_file():
                raise ValueError('Resume requires an existing resume_checkpoint')
            self._read_history()
            start_epoch = incoming.get('resume_start_epoch')
            start_step = incoming.get('resume_start_step', incoming.get('resume_startstep',
                         incoming.get('resume_start_global_step')))
            if start_epoch is None or int(start_epoch) != self.last_epoch + 1:
                raise ValueError('Resume must start at the first unrecorded epoch')
            if start_step is None or int(start_step) != max(self.last_step, 0):
                raise ValueError('Resume checkpoint start step must match the completed timeline')
            self.config = stored
            for key in _DYNAMIC:
                if key in incoming:
                    self.config[key] = incoming[key]
        else:
            self.config = {key: None for key in REQUIRED_CONFIG}
            self.config.update(incoming)
            self.config.update(run_id=self.directory.name, start_time=_now(),
                               command=incoming.get('command', sys.argv),
                               cwd=incoming.get('cwd', str(Path.cwd())), world_size=1,
                               sessions=[])
            (self.directory / 'checkpoints').mkdir()
        sessions = self.config.setdefault('sessions', [])
        self.session = len(sessions) + 1
        sessions.append(dict(session=self.session, start_time=_now(), pid=os.getpid(),
                             mode='resume' if self.resume else 'new',
                             resume_checkpoint=incoming.get('resume_checkpoint'),
                             resume_start_epoch=incoming.get('resume_start_epoch'),
                             resume_start_step=incoming.get('resume_start_step',
                                 incoming.get('resume_startstep', incoming.get('resume_start_global_step')))))
        self.config.update(status='running', end_time=None, exit_code=None,
                           output_directory=str(self.directory))
        _atomic_json(self.directory / 'config.json', self.config)

    def __enter__(self):
        global _CURRENT
        if _CURRENT is not None or self._entered:
            raise RuntimeError('Nested or repeated logger contexts are prohibited')
        try:
            if self.resume:
                if not self.directory.is_dir():
                    raise FileNotFoundError(self.directory)
            else:
                self._create_directory()
            self._acquire()
            self._prepare_config()
            self.log = (self.directory / 'train_log.txt').open('a', encoding='utf-8', buffering=1)
            self.csv_stream = (self.directory / 'metrics.csv').open('a', encoding='utf-8', newline='')
            self.writer = csv.DictWriter(self.csv_stream, fieldnames=COLUMNS)
            if not self.resume:
                self.writer.writeheader()
                self.csv_stream.flush()
            self.original_stdout, self.original_stderr = sys.stdout, sys.stderr
            lock = threading.RLock()
            sys.stdout = _Tee(sys.stdout, self.log, lock)
            sys.stderr = _Tee(sys.stderr, self.log, lock)
            self._entered, _CURRENT = True, self
            print(f'=== SESSION {self.session} {"RESUME" if self.resume else "NEW"} {_now()} ===')
            print('Run directory:', self.directory)
            print('Startup configuration:', json.dumps(self.config, ensure_ascii=False))
            return self
        except BaseException:
            self._cleanup()
            raise

    def update(self, **fields):
        if not self._entered:
            raise RuntimeError('Logger context has not started')
        value = dict(self.config)
        value.update(_plain(fields))
        _atomic_json(self.directory / 'config.json', value)
        self.config = value

    def record(self, row):
        if not self._entered:
            raise RuntimeError('Logger context has not started')
        row = _plain(row)
        normalized = {}
        for key, value in row.items():
            key = ALIASES.get(key, key)
            if key in normalized:
                raise ValueError(f'Duplicate canonical metric field: {key}')
            normalized[key] = value
        unknown = normalized.keys() - set(COLUMNS)
        if unknown:
            raise ValueError(f'Unsupported metric fields: {sorted(unknown)}')
        epoch, step = normalized.get('epoch'), normalized.get('global_step')
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not isinstance(step, int) or isinstance(step, bool):
            raise ValueError('epoch and global_step must be integers')
        if epoch < 0 or epoch <= self.last_epoch or step < 0 or step < self.last_step:
            raise ValueError('Duplicate/decreasing epoch or decreasing global_step')
        normalized['session'] = self.session
        for key, value in normalized.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'Nonfinite metric: {key}')
        self.writer.writerow({k: '' if normalized.get(k) is None else normalized[k] for k in COLUMNS})
        self.csv_stream.flush()
        self.last_epoch, self.last_step = epoch, step
        self.config['final_metrics'] = normalized
        if normalized.get('is_best'):
            self.config['best_metrics'], self.config['best_epoch'] = normalized, epoch
        self.update(last_completed_epoch=epoch, last_completed_global_step=step,
                    final_metrics=normalized, best_metrics=self.config.get('best_metrics'),
                    best_epoch=self.config.get('best_epoch'))

    def _cleanup(self):
        global _CURRENT
        if self._entered:
            sys.stdout, sys.stderr = self.original_stdout, self.original_stderr
        for stream in (self.csv_stream, self.log):
            if stream is not None:
                stream.close()
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
            self._lock_path.unlink(missing_ok=True)
        if _CURRENT is self:
            _CURRENT = None
        self._entered = False

    def __exit__(self, exc_type, exc, tb):
        status = 'completed' if exc is None else ('interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed')
        code = 0 if exc is None else (130 if isinstance(exc, KeyboardInterrupt) else 1)
        if isinstance(exc, SystemExit):
            code = exc.code if isinstance(exc.code, int) else 1
            status = 'completed' if code == 0 else 'interrupted'
        try:
            for stream in (sys.stdout, sys.stderr):
                if isinstance(stream, _Tee):
                    stream.finish_progress()
            if exc is not None:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            print(f'=== SESSION {self.session} END status={status} exit_code={code} {_now()} ===')
            try:
                self.config['sessions'][-1].update(end_time=_now(), status=status, exit_code=code)
                self.update(status=status, end_time=_now(), exit_code=code,
                            final_metrics=self.config.get('final_metrics'),
                            best_metrics=self.config.get('best_metrics'),
                            best_epoch=self.config.get('best_epoch'))
            except Exception as logging_error:
                print(f'Config finalization failed; existing logs preserved: {logging_error}', file=sys.stderr)
        finally:
            self._cleanup()
        return False
