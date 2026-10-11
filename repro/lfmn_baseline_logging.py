"""Strict, fail-closed logging shared by baseline checks and training."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import socket
import traceback
import uuid


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, value):
        self.terminal.write(value)
        self.file.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


class RunLog:
    def __init__(self, root, seed=1, resume=None, config=None):
        if resume:
            self.path = Path(resume).resolve()
            self.config = json.loads((self.path / 'config.json').read_text(encoding='utf-8'))
        else:
            stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.path = Path(root).resolve() / ('LFMN_original_x4_seed%d_%s_%s' %
                                              (seed, stamp, uuid.uuid4().hex[:6]))
            self.path.mkdir(parents=True, exist_ok=False)
            self.config = dict(config or {})
            required = ('model','model_source','command','cwd','git_commit','git_dirty',
                        'git_diff_summary','dataset','train_range','validation_range','scale',
                        'degradation','epochs','steps_per_epoch','batch_size','patch_size',
                        'optimizer','learning_rate','scheduler','loss','seed','device','gpu_name',
                        'world_size','ema_enabled','self_ensemble_enabled','pretrained_checkpoint',
                        'resume_checkpoint','resume_start_epoch','metric_protocol')
            for key in required:
                self.config.setdefault(key,None)
            self.config.update(run_id=self.path.name, start_time=dt.datetime.now().isoformat(),
                               output_directory=str(self.path), sessions=[])
        self.lock = self.path / 'run.lock'
        # A stale lock after SIGKILL needs explicit manual inspection/removal.
        with self.lock.open('x',encoding='utf-8') as lock:
            json.dump({'pid':os.getpid(),'host':socket.gethostname()},lock)
        self.file = (self.path / 'train_log.txt').open('a', encoding='utf-8', buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(self.stdout, self.file), Tee(self.stderr, self.file)
        self.config['status'] = 'running'
        self.config.pop('end_time',None)
        self.config.pop('exit_code',None)
        self.config['sessions'].append(dict(start_time=dt.datetime.now().isoformat(), resume=bool(resume)))
        self.save()
        print('SESSION %s run=%s' % ('RESUME' if resume else 'NEW', self.path), flush=True)

    def save(self):
        atomic_json(self.path / 'config.json', self.config)

    def finish(self, error=None):
        if error:
            traceback.print_exception(type(error), error, error.__traceback__)
        self.config.update(status=('interrupted' if isinstance(error, KeyboardInterrupt) else
                                   'failed' if error else 'complete'),
                           end_time=dt.datetime.now().isoformat(), exit_code=1 if error else 0)
        self.config['sessions'][-1].update(end_time=self.config['end_time'],
                                         status=self.config['status'],exit_code=self.config['exit_code'])
        try:
            self.save()
        finally:
            sys.stdout, sys.stderr = self.stdout, self.stderr
            self.file.close()
            self.lock.unlink()
