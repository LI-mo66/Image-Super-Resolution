"""Three independent V1 adaptations. No baseline training and no implicit protocol override."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from v1_combo_protocol import ROOT, TEACHER_SHA, TEACHER_COMMIT, sha256, command, validate_v1_config


def checked(cmd, cwd=ROOT):
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def data_fingerprint(data):
    hashes = {}
    for split, indices in [('train',range(1,801)),('valid',range(801,901))]:
        for index in indices:
            for suffix in (f'DIV2K/DIV2K_{split}_HR/{index:04d}.png',
                           f'DIV2K/DIV2K_{split}_LR_bicubic/X4/{index:04d}x4.png'):
                hashes[suffix] = sha256(data/suffix)
                if len(hashes) % 200 == 0:
                    print(f'Data audit {len(hashes)}/1800',flush=True)
    return hashes


def source_fingerprint():
    files = checked(['git','ls-files','--','LFMN','repro']).splitlines()
    return {f: sha256(ROOT/f) for f in files if (ROOT/f).is_file()}


def gpu_environment():
    import torch
    import numpy
    query = checked(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,driver_version',
                     '--format=csv,noheader,nounits'])
    devices = query.splitlines()
    if len(devices) != 1:
        raise RuntimeError('Dedicated single-GPU instance required')
    parts = [v.strip() for v in devices[0].split(',')]
    if int(parts[3]) < 16000 or int(parts[4]) > 1000:
        raise RuntimeError('Require >=16GB GPU with <=1GB allocated; inspect nvidia-smi')
    active = checked(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
    if active and 'No running' not in active:
        raise RuntimeError('Existing CUDA process; do not interfere')
    return dict(python=sys.version,torch=str(torch.__version__),cuda_build=torch.version.cuda,
                numpy=numpy.__version__,gpu=parts[1],uuid=parts[2],memory_mb=parts[3],driver=parts[5])


def shutdown_command(path=Path('/usr/bin/shutdown')):
    if not path.is_file():
        raise FileNotFoundError(path)
    prefix = path.read_bytes()[:4]
    if prefix == b'\x7fELF' or prefix.startswith(b'#!'):
        return [str(path)]
    subprocess.run(['/bin/bash','-n',str(path)],check=True)
    return ['/bin/bash',str(path)]


def run_child(cmd, console):
    # Inherit nohup stdout: tail -F launcher log sees live child progress.
    print('START: '+ ' '.join(map(str,cmd)),flush=True)
    with console.open('w',encoding='utf-8') as stream:
        proc = subprocess.Popen(cmd,cwd=ROOT/'LFMN',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                text=True,bufsize=1)
        try:
            for line in proc.stdout:
                stream.write(line)
                stream.flush()
                print(line,end='',flush=True)
            status = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
        if status:
            raise subprocess.CalledProcessError(status,cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',required=True,type=Path)
    ap.add_argument('--teacher-repo',type=Path,default=ROOT/'repro/swinir_ref')
    ap.add_argument('--teacher-checkpoint',type=Path,default=ROOT/'repro/teacher_weights/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth')
    ap.add_argument('--output',required=True,type=Path)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--baseline',type=Path,help='Historical V1 for reference, not automatically approved reuse')
    mode.add_argument('--collect-only',action='store_true',help='No baseline file access or relative result')
    ap.add_argument('--groups',nargs='+',choices=('n21','n22','n23'),default=['n21','n22','n23'])
    ap.add_argument('--run',action='store_true',help='Explicitly authorize 3x20 candidate epochs plus smoke')
    ap.add_argument('--shutdown-on-success',action='store_true')
    ap.add_argument('--dedicated-instance',action='store_true')
    args = ap.parse_args()
    if len(set(args.groups)) != len(args.groups):
        raise ValueError('Duplicate groups')
    args.output = args.output.resolve()
    args.output.relative_to((ROOT/'experiment/all_runs').resolve())
    if args.output == (ROOT/'experiment/all_runs').resolve() or args.output.exists():
        raise FileExistsError('Provide a NEW child output directory')
    if checked(['git','status','--porcelain','--untracked-files=normal']):
        raise RuntimeError('Dirty checkout. Run git status --short; keep launcher log OUTSIDE checkout')
    if args.shutdown_on_success and not args.dedicated_instance:
        raise ValueError('Whole-instance shutdown requires --dedicated-instance')
    poweroff = shutdown_command() if args.shutdown_on_success else None
    args.data_root = args.data_root.resolve()
    args.teacher_repo = args.teacher_repo.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    if sha256(args.teacher_checkpoint) != TEACHER_SHA:
        raise ValueError('Wrong official teacher checkpoint')
    if checked(['git','rev-parse','HEAD'],args.teacher_repo) != TEACHER_COMMIT:
        raise ValueError('Wrong SwinIR source commit')
    if checked(['git','status','--porcelain','--untracked-files=no'],args.teacher_repo):
        raise ValueError('Modified teacher source')
    print('Auditing 1800 data files, source, teacher and environment; no optimizer yet',flush=True)
    environment = gpu_environment()
    data_hashes = data_fingerprint(args.data_root)
    sources = source_fingerprint()
    baseline = None
    if args.baseline:
        args.baseline = args.baseline.resolve()
        blocks = validate_v1_config(args.baseline/'config.txt')
        from summarize_v1_combos import metrics
        metrics(args.baseline)
        baseline = dict(directory=str(args.baseline),config_blocks=len(blocks),
            config_sha=sha256(args.baseline/'config.txt'),
            curves={n:sha256(args.baseline/n) for n in ('psnr_log.pt','ssim_log.pt')},
            model_20_sha=sha256(args.baseline/'model/model_20.pt'),
            per_image_hashes={f'per_image_metrics/epoch_{e:04d}.pt':sha256(args.baseline/'per_image_metrics'/f'epoch_{e:04d}.pt') for e in range(1,21)},
            status='REFERENCE_ONLY_PROVENANCE_PENDING',
            unknown=['historical data hashes','training environment','baseline source/metric hashes','historical update/data-stream audit'])
    if shutil.disk_usage(ROOT).free < 5*1024**3:
        raise RuntimeError('Less than 5GB free disk')
    manifest = dict(commit=checked(['git','rev-parse','HEAD']),branch=checked(['git','branch','--show-current']),
        started=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment=environment,
        data_root=str(args.data_root),data_hashes=data_hashes,source_hashes=sources,baseline=baseline,
        baseline_reuse='NOT_APPROVED_NO_RETRAIN',groups=args.groups,teacher_sha256=TEACHER_SHA,
        protocol=dict(epochs=20,cosine_tmax=150,eta_min=1e-6,seed=1,workers=8,steps_per_epoch=1000,
                      batch=4,hr_patch=256,scale=4,output_kd=.1,teacher_microbatch=1),
        efficiency='Parameters <2% overhead; actual latency/MAC budget unverified')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('data_hashes','source_hashes')},indent=2),flush=True)
    if not args.run:
        print('AUDIT ONLY COMPLETE. Add --run to collect candidates; no V1 retraining.',flush=True)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    try:
        streams = []
        for group in args.groups:
            stream_cmd = command(group,args.data_root,args.teacher_repo,args.teacher_checkpoint,args.output/group)
            stream_cmd[2] = str(ROOT/'repro/check_v1_combo_data_stream.py')
            run_child(stream_cmd,args.output/f'{group}_data_stream_console.log')
            streams.append(json.loads((args.output/'data_stream'/f'{group}.json').read_text()))
        if any(value != streams[0] for value in streams[1:]):
            raise RuntimeError('Production data streams differ before training')
        # All group smoke gates complete BEFORE any 20-epoch candidate starts.
        for group in args.groups:
            run_child(command(group,args.data_root,args.teacher_repo,args.teacher_checkpoint,
                              args.output/'smoke'/group,smoke=True),args.output/f'{group}_smoke_console.log')
            run_child([sys.executable,'-u',str(ROOT/'repro/check_v1_combo_efficiency.py'),
                       '--group',group,'--output',str(args.output/'efficiency'/f'{group}.json')],
                       args.output/f'{group}_efficiency_console.log')
        data_stream = None
        for group in args.groups:
            run_child(command(group,args.data_root,args.teacher_repo,args.teacher_checkpoint,args.output/group),
                      args.output/f'{group}_console.log')
            from summarize_v1_combos import metrics
            metrics(args.output/group)
            records = [json.loads(line) for line in (args.output/group/'mechanism.jsonl').read_text().splitlines()]
            if len(records) != 20 or any(r['steps'] != 1000 for r in records):
                raise RuntimeError('Wrong candidate update budget')
            for epoch, record in enumerate(records,1):
                expected_lr = 1e-6+(2e-4-1e-6)*(1+math.cos(math.pi*(epoch-1)/150))/2
                if record['epoch'] != epoch or abs(record['lr']-expected_lr) > 1e-12:
                    raise RuntimeError('Unexpected cosine time axis')
            current_stream = [r['first_train_batch'] for r in records]
            if current_stream[0] != streams[args.groups.index(group)][0]:
                raise RuntimeError('Actual training stream differs from preflight')
            if data_stream is not None and data_stream != current_stream:
                raise RuntimeError('Candidate first-batch streams differ')
            data_stream = current_stream
        if sources != source_fingerprint() or data_hashes != data_fingerprint(args.data_root):
            raise RuntimeError('Source/data changed during run')
        if sha256(args.teacher_checkpoint) != TEACHER_SHA:
            raise RuntimeError('Teacher weights changed')
        if baseline and (baseline['config_sha'] != sha256(args.baseline/'config.txt') or
            baseline['model_20_sha'] != sha256(args.baseline/'model/model_20.pt') or
            any(h != sha256(args.baseline/n) for n,h in {**baseline['curves'],**baseline['per_image_hashes']}.items())):
            raise RuntimeError('Reference baseline changed during collection')
        from summarize_v1_combos import summarize
        summarize(args.output,args.groups,args.baseline,write=True)
        manifest.update(status='COLLECTION_COMPLETE_NOT_GO',finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status='FAILED_NO_SHUTDOWN',error=repr(error))
        raise
    finally:
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if poweroff:
        print('All requested groups and integrity checks complete; shutting down dedicated instance.',flush=True)
        os.sync()
        subprocess.run(poweroff,check=True)


if __name__ == '__main__':
    main()
