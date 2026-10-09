"""Locked F2/B0 150-epoch protocol and artifact helpers."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MODELS={'B0':'LFMN','F2':'LFMNF2'}
BENCHMARKS={'Set5':5,'Set14':14,'B100':100,'Urban100':100,'Manga109':109}
PROTOCOL=dict(scale=4,seed=1,patch_size=256,batch_size=4,optimizer='ADAM',
              betas=[0.9,0.999],epsilon=1e-8,weight_decay=0,lr=2e-4,
              eta_min=1e-6,scheduler='cosine',scheduler_t_max=150,
              planned_epochs=150,stop_epoch=150,test_every=1000,
              data_range='1-800/801-900',loss='1*L1',initialization='scratch',
              pretrained_checkpoint=None,self_ensemble=False,chop=False,
              weight_ema=False,tab_centroid_ema=True,precision='single',rgb_range=255,
              augmentation=True,checkpoint_primary='fixed epoch150',
              checkpoint_secondary='best independent DIV2K mean; first epoch wins exact ties')
SOURCE_PATHS=['LFMN/model/lfmn.py','LFMN/model/lfmnf2.py','LFMN/model/__init__.py',
              'LFMN/trainer.py','LFMN/utility.py','LFMN/option.py',
              'LFMN/data/__init__.py','LFMN/data/common.py','LFMN/data/srdata.py',
              'LFMN/data/div2k.py','LFMN/loss/__init__.py','LFMN/run_logging.py',
              'repro/f2_train_entry.py','repro/f2_150_common.py',
              'repro/f2_150_train_entry.py','repro/run_f2_150_server.py',
              'repro/summarize_f2_150.py','repro/check_f2_150_summary.py',
              'repro/check_f2_budget.py','repro/check_training_run_logging.py','repro/check_f2_150_observation.py',
              'repro/profile_f2_150_resources.py']


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def source_hashes():
    return {p:hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in SOURCE_PATHS}


def sha256_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path,rows):
    if not rows:raise ValueError('Refusing an empty metric table')
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows);stream.flush()


def manifest(group):
    return json.loads((Path(group)/'protocol.json').read_text(encoding='utf-8'))


def run_directory(group,label):
    return Path(group)/manifest(group)['runs'][label]
