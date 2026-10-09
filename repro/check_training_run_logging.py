#!/usr/bin/env python3
"""Exercise streaming stdout/stderr, distinct directories, failure and append."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
from run_logging import launch


def main():
    scratch = ROOT / 'experiment'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lfmn_logging_', dir=scratch) as temporary:
        root = Path(temporary)
        live = root / 'live'
        command = [sys.executable, '-u', '-c',
                   "import sys,time; print('STDOUT_FIRST'); print('STDERR_FIRST',file=sys.stderr); time.sleep(1.5); print('DONE')"]
        errors = []
        def run():
            try:
                launch(command, live, {'test': True}, ROOT)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 5
        observed = False
        while thread.is_alive() and time.monotonic() < deadline:
            path = live / 'train_log.txt'
            if path.exists() and 'STDERR_FIRST' in path.read_text(encoding='utf-8'):
                observed = True
                break
            time.sleep(0.02)
        thread.join()
        assert observed and not errors, (observed, errors)
        first = (live / 'train_log.txt').read_bytes()
        launch([sys.executable, '-c', "print('SECOND_RUN')"], root / 'second', {}, ROOT)
        assert (live / 'train_log.txt').read_bytes() == first
        try:
            launch(command, live, {}, ROOT)
            raise AssertionError('directory overwrite accepted')
        except FileExistsError:
            pass
        try:
            launch([sys.executable, '-c', "raise RuntimeError('controlled failure')"], root / 'failure', {}, ROOT)
            raise AssertionError('failure swallowed')
        except subprocess.CalledProcessError:
            pass
        failed = (root / 'failure/train_log.txt').read_text(encoding='utf-8')
        assert 'Traceback' in failed and 'controlled failure' in failed
        assert json.loads((root / 'failure/config.json').read_text())['status'] == 'failed'
        launch([sys.executable, '-c', "print('RESUME_MARKER')"], live,
               {'resume_checkpoint': 'test_checkpoint', 'resume_start_epoch': 1}, ROOT, resume=True)
        payload = json.loads((live / 'config.json').read_text())
        assert len(payload['sessions']) == 2
        assert all(session['status'] == 'completed' for session in payload['sessions'])
        assert (live / 'train_log.txt').read_bytes().startswith(first)
        assert payload['resume_start_epoch'] == 1
        print('LOGGING CHECK PASSED: live streams, non-overwrite, traceback, append/resume')


if __name__ == '__main__':
    main()
