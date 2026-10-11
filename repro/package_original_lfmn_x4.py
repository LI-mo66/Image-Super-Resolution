"""Build a small standalone server delivery, excluding all weights/data/results."""
import hashlib
import json
from pathlib import Path
import zipfile
from original_lfmn_x4 import ROOT, git, sha


def main():
    files = ['原始LFMN_x4_服务器复现说明.md',
             'repro/original_lfmn_x4.py','repro/lfmn_baseline_logging.py',
             'repro/check_lfmn_baseline_logging.py','repro/check_original_lfmn_x4.py',
             'repro/preflight_original_lfmn_x4.py','repro/run_original_lfmn_x4_server.sh',
             'repro/requirements-original-lfmn-py39.txt',
             'LFMN/model/lfmn.cpython-39-x86_64-linux-gnu.so','LFMN/model/lfmn.py',
             'LFMN/model/common.py','LFMN/model/__init__.py','LFMN/data/common.py',
             'LFMN/data/__init__.py','LFMN/utility.py']
    output = ROOT/'experiment/deliverables'
    output.mkdir(parents=True,exist_ok=True)
    commit = git('rev-parse','HEAD')
    destination = output/('original_lfmn_x4_server_%s.zip' % commit[:8])
    manifest = dict(repository_commit=commit,files={f:sha(ROOT/f) for f in files},
                    original_binary_sha256=sha(ROOT/files[8]),
                    note='Python source included for local diagnostics only; formal uses original binary.')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for file in files:
            # Normalize text newlines for Linux; binary bytes are untouched.
            data = (ROOT/file).read_bytes()
            if not file.endswith('.so'):
                data = data.replace(b'\r\n',b'\n')
            z.writestr(file,data)
        # Use hashes of the actual delivered bytes, after newline normalization.
        manifest['files'] = {f:hashlib.sha256(z.read(f)).hexdigest() for f in files}
        z.writestr('delivery_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        assert not any(n.endswith('.pt') or 'datasets/' in n or '.git/' in n for n in z.namelist())
        assert all(hashlib.sha256(z.read(f)).hexdigest() == h for f,h in manifest['files'].items())
    print('DELIVERY %s bytes=%d SHA256=%s' % (destination,destination.stat().st_size,sha(destination)))


if __name__ == '__main__':
    main()
