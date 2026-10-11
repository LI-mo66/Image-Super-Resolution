"""Package committed gate diagnostic source, never experiment assets."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    required=['repro/probe_f4_gates.py','repro/shutdown_f4_gate_probe.py',
              'repro/run_f4_gate_probe_nohup.sh','repro/test_f4_gate_probe.py',
              'F4_门控干预预注册.md','F4_门控干预_服务器运行说明.md']
    raw=subprocess.check_output(['git','archive','--format=zip',commit,'--','LFMN','requirements-server.txt',*required],cwd=ROOT)
    out=ROOT/'experiment/offline_packages';out.mkdir(parents=True,exist_ok=True)
    target=out/('F4_gate_server_'+commit[:7]+'.zip');prefix='Image-Super-Resolution-F4-gate/'
    hashes={}
    with zipfile.ZipFile(io.BytesIO(raw)) as source,zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as dest:
        for item in source.infolist():
            if item.is_dir() or item.filename.endswith(('.pt','.pyc','.pyd','.so')):continue
            data=source.read(item);hashes[item.filename]=hashlib.sha256(data).hexdigest()
            dest.writestr(prefix+item.filename,data)
        dest.writestr(prefix+'OFFLINE_SOURCE.json',json.dumps(dict(source_commit=commit,files_sha256=hashes),indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,digest in hashes.items():
            data=archive.read(prefix+name)
            assert hashlib.sha256(data).hexdigest()==digest
            if name.endswith('.sh'):assert b'\r' not in data, 'Linux shell script contains CR: '+name
        assert all(name in hashes for name in required)
    print(json.dumps(dict(file=str(target),commit=commit,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
