"""Optional official teacher preparation. Resume partial bytes; publish only verified files.

No PyTorch installation or training. Existing invalid final weights are preserved
and rejected rather than overwritten. Interrupted partial downloads are resumable.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request
from v1n23c_protocol import ROOT, TEACHER_COMMIT, TEACHER_SHA, sha256

URL = ('https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
       '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')
SIZE = 59611499


def download_once(url, partial, total=SIZE, opener=urllib.request.urlopen):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError('Partial file is larger than the official asset; preserve and inspect it')
    if offset == total:
        return
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-',
                                                   'User-Agent': 'V1N23C-research-setup'})
    with opener(request, timeout=60) as response:
        status = response.status
        if status == 206:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if (not match or int(match[1]) != offset or int(match[3]) != total
                    or not offset <= int(match[2]) < total):
                raise ValueError('Invalid Content-Range; partial file untouched')
        elif status != 200 or offset:
            raise ValueError('Server did not honor resume; preserve partial and retry later')
        length = response.headers.get('Content-Length')
        if length and int(length) > total-offset:
            raise ValueError('Unexpected response size')
        with partial.open('ab') as stream:
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                if stream.tell()+len(chunk) > total:
                    raise ValueError('Oversized response; do not publish')
                stream.write(chunk)
                stream.flush()
                print(f'Teacher download {stream.tell()}/{total} bytes', flush=True)
            os.fsync(stream.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=ROOT/'repro/swinir_ref')
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')
    args = ap.parse_args()
    repo, weight = args.repo.resolve(), args.checkpoint.resolve()
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        # Failed clones also stay under an ignored path, not as untracked repro/.
        staging = ROOT/'experiment/teacher_staging'
        staging.mkdir(parents=True, exist_ok=True)
        clone = Path(tempfile.mkdtemp(prefix=repo.name+'.clone.', dir=staging))
        # Do not reset/delete any existing user reference repository.
        subprocess.run(['git','-c','http.version=HTTP/1.1','clone','--depth','1',
                        'https://github.com/JingyunLiang/SwinIR.git',str(clone)], check=True)
        subprocess.run(['git','-C',str(clone),'fetch','--depth','1','origin',TEACHER_COMMIT],check=True)
        subprocess.run(['git','-C',str(clone),'checkout','--detach',TEACHER_COMMIT],check=True)
        if repo.exists():
            raise FileExistsError(repo)
        clone.rename(repo)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if commit != TEACHER_COMMIT or dirty:
        raise ValueError('Existing teacher code has wrong commit/modifications; preserved, not reset')
    weight.parent.mkdir(parents=True, exist_ok=True)
    if weight.exists():
        if sha256(weight) != TEACHER_SHA:
            raise ValueError('Existing final weight invalid; preserved. Provide another --checkpoint path')
    else:
        partial = weight.with_name(weight.name+'.partial')
        for attempt in range(1,7):
            try:
                download_once(URL, partial)
                if partial.stat().st_size == SIZE:
                    break
            except (OSError, ValueError) as error:
                print(f'Download attempt {attempt}/6: {error}; partial bytes preserved', flush=True)
            if attempt < 6:
                time.sleep(3)
        if not partial.exists() or partial.stat().st_size != SIZE or sha256(partial) != TEACHER_SHA:
            raise RuntimeError(f'Download incomplete/invalid; rerun to resume {partial}; NO training started')
        # Linux publication without replacing an existing destination.
        os.link(partial, weight)
        partial.unlink()
    print(f'TEACHER READY\nrepo={repo}\ncheckpoint={weight}\nsha256={TEACHER_SHA}',flush=True)


if __name__ == '__main__':
    main()
