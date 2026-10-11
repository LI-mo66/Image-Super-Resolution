"""No training starts before these infrastructure checks pass."""
import json
from pathlib import Path
import tempfile
from lfmn_baseline_logging import RunLog
import sys


def main():
    output = Path(__file__).resolve().parents[1] / 'experiment' / 'baseline_checks'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='lfmn_baseline_log_', dir=output) as root:
        a = RunLog(root, config={'probe': True})
        print('stdout live marker')
        print('stderr live marker', file=sys.stderr)
        assert 'stderr live marker' in (a.path / 'train_log.txt').read_text(encoding='utf-8')
        a.finish()
        before = (a.path / 'train_log.txt').read_bytes()
        b = RunLog(root)
        assert a.path != b.path
        assert before == (a.path / 'train_log.txt').read_bytes()
        try:
            raise RuntimeError('controlled logging fault')
        except RuntimeError as error:
            b.finish(error)
        assert json.loads((b.path / 'config.json').read_text())['status'] == 'failed'
        assert 'Traceback' in (b.path / 'train_log.txt').read_text()
        c = RunLog(root, resume=a.path)
        print('resumed live marker')
        c.finish()
        assert (a.path / 'train_log.txt').read_bytes().startswith(before)
        assert len(json.loads((a.path / 'config.json').read_text())['sessions']) == 2
    print('PASS: live stdout/stderr, unique directories, exception traceback/status, append resume')


if __name__ == '__main__':
    main()
