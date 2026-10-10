#!/usr/bin/env python3
"""Export only committed F1 server sources to a checksum-verified offline ZIP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ['LFMN', 'repro/run_f1_screen_server.py', 'repro/f1_train_entry.py',
         'repro/check_f1_identity_delta_esa.py', 'repro/check_training_run_logging.py',
         'repro/profile_f1_resources.py', 'repro/summarize_f1_screen.py',
         'repro/check_f1_epoch_summary.py', 'F1_服务器运行说明.md']
FILES += ['repro/diagnose_f1_20e.py', 'repro/check_f1_diagnostic.py', 'F1_20e_服务器只读诊断说明.md']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'experiment/offline_packages')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    data = subprocess.check_output(['git', 'archive', '--format=zip', commit, '--', *FILES], cwd=ROOT)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / ('F1_Set5_server_' + commit[:7] + '.zip')
    if target.exists():
        raise FileExistsError(target)
    prefix = 'Image-Super-Resolution-F1-Set5-offline/'
    hashes = {}
    with zipfile.ZipFile(io.BytesIO(data)) as original, zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as result:
        for info in original.infolist():
            if info.is_dir():
                continue
            if info.filename.endswith(('.so', '.pyd', '.pyc')):
                continue
            content = original.read(info)
            hashes[info.filename] = hashlib.sha256(content).hexdigest()
            result.writestr(prefix + info.filename, content)
        manifest = {'source_commit': commit, 'files_sha256': hashes,
                    'scope': 'committed LFMN sources, official bundled weights, F1 server scripts; no datasets or experiment results'}
        result.writestr(prefix + 'OFFLINE_SOURCE.json', json.dumps(manifest, indent=2, ensure_ascii=False))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise AssertionError('ZIP integrity check failed')
        manifest = json.loads(archive.read(prefix + 'OFFLINE_SOURCE.json'))
        for name, expected in manifest['files_sha256'].items():
            if hashlib.sha256(archive.read(prefix + name)).hexdigest() != expected:
                raise AssertionError('ZIP member checksum mismatch: ' + name)
    print(json.dumps({'file': str(target), 'bytes': target.stat().st_size,
                     'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                     'source_commit': commit, 'files': len(hashes)}, indent=2))


if __name__ == '__main__':
    main()
