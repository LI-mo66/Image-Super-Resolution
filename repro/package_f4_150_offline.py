"""Complete checksum-verified F4 150 continuation package from committed files."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    candidate=subprocess.check_output(['git','ls-files','repro/*f4*'],cwd=ROOT,text=True).splitlines()
    files=['LFMN','requirements-server.txt','repro/run_f1_screen_server.py',
           'repro/check_training_run_logging.py','F4_150_服务器运行说明.md']+candidate
    raw=subprocess.check_output(['git','archive','--format=zip',commit,'--',*files],cwd=ROOT)
    out=ROOT/'experiment/offline_packages';out.mkdir(parents=True,exist_ok=True)
    target=out/('F4_150_server_'+commit[:7]+'.zip');prefix='Image-Super-Resolution-F4-150/'
    hashes={}
    with zipfile.ZipFile(io.BytesIO(raw)) as original,zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for item in original.infolist():
            if item.is_dir() or item.filename.endswith(('.so','.pyd','.pt','.pyc')):continue
            data=original.read(item);hashes[item.filename]=hashlib.sha256(data).hexdigest()
            archive.writestr(prefix+item.filename,data)
        archive.writestr(prefix+'OFFLINE_SOURCE.json',json.dumps(dict(source_commit=commit,files_sha256=hashes),indent=2))
    required=['repro/f4_150_train_entry.py','repro/run_f4_150_server.py','repro/run_f4_150_nohup.sh',
              'repro/f4_150_monitor.py','repro/summarize_f4_150.py','repro/shutdown_f4_150.py']
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise ValueError('invalid archive')
        for name,h in hashes.items():
            if hashlib.sha256(archive.read(prefix+name)).hexdigest()!=h:raise ValueError(name)
        for name in required:
            if name not in hashes:raise ValueError('package missing '+name)
    print(json.dumps(dict(file=str(target),commit=commit,sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
