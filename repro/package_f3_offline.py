"""Committed sources only, checksum-verified offline F3 server archive."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=['LFMN','requirements-server.txt','repro/run_f1_screen_server.py','repro/check_training_run_logging.py',
        'repro/run_f3_screen_server.py','repro/f3_train_entry.py','repro/check_f3.py',
        'repro/profile_f3_resources.py','repro/summarize_f3_screen.py','F3_服务器运行说明.md']
    data=subprocess.check_output(['git','archive','--format=zip',commit,'--',*files],cwd=ROOT)
    out=ROOT/'experiment/offline_packages';out.mkdir(parents=True,exist_ok=True)
    target=out/('F3_server_'+commit[:7]+'.zip');prefix='Image-Super-Resolution-F3/'
    hashes={}
    with zipfile.ZipFile(io.BytesIO(data)) as original,zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for item in original.infolist():
            if item.is_dir() or item.filename.endswith(('.so','.pyd','.pyc','.pt')):continue
            content=original.read(item);hashes[item.filename]=hashlib.sha256(content).hexdigest()
            archive.writestr(prefix+item.filename,content)
        archive.writestr(prefix+'OFFLINE_SOURCE.json',json.dumps(dict(source_commit=commit,files_sha256=hashes),indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name,value in hashes.items():assert hashlib.sha256(archive.read(prefix+name)).hexdigest()==value
    print(json.dumps(dict(file=str(target),source_commit=commit,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))

if __name__=='__main__':main()
