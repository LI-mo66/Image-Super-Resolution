"""Pinned P0/P3/P4 server orchestration. --plan never executes models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
REFS = {'P0': 'origin/codex/p0-paired-adaptation', 'P3': 'origin/codex/p3-mse-tail', 'P4': 'origin/codex/p4-lrsa-position-bias'}
WEIGHT_SHA = '44999471d8cc2d5f7dbf10d354766e08a5d23a84200a1060dfa9a9ed7a7711dd'
_ACTIVE = set()
_LOCK = threading.Lock()
_STOP = threading.Event()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def json_write(path, value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        stream.write(json.dumps(value,indent=2,ensure_ascii=False)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)

def data_root(explicit):
    candidates = [Path(explicit)] if explicit else [ROOT/'datasets', ROOT/'dataset', ROOT.parent/'datasets', ROOT.parent/'dataset', Path('/root/autodl-tmp/datasets'), Path('/root/autodl-tmp/dataset'), Path('/root/autodl-tmp')]
    for path in candidates:
        if (path/'DIV2K/DIV2K_train_HR').is_dir() and (path/'DIV2K/DIV2K_valid_HR').is_dir():
            return path.resolve()
    raise FileNotFoundError('DIV2K not found; use --data-root containing DIV2K/')

def terminate_owned(process):
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=10)
        except ProcessLookupError:
            pass

def stop_owned():
    _STOP.set()
    with _LOCK:
        active=list(_ACTIVE)
    for process in active:
        terminate_owned(process)

def invoke(command, cwd, gpu, log_path):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    with Path(log_path).open('w', encoding='utf-8') as log:
        with _LOCK:
            if _STOP.is_set():
                raise RuntimeError('Launcher stopping; no new child process permitted')
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            _ACTIVE.add(process)
        try:
            for line in process.stdout:
                with _LOCK:
                    print(f'[{Path(cwd).name} GPU {gpu}] {line}', end='', flush=True)
                log.write(line); log.flush()
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            terminate_owned(process)
            with _LOCK:
                _ACTIVE.discard(process)

def validate_check_reports(reports, model_hashes, shared_hashes, data):
    for scheme,report in reports.items():
        expected={'status':'verified','scheme':scheme,'checkpoint_sha256':WEIGHT_SHA,
                  'model_sha256':model_hashes[scheme],
                  'runner_sha256':shared_hashes['repro/train_p_adaptation.py'][scheme],
                  'data_root':str(data)}
        for key,value in expected.items():
            if report.get(key)!=value:
                raise ValueError(f'Engineering gate mismatch: {scheme} {key}')
        if not report.get('gpu') or not report.get('torch'):
            raise ValueError('Missing GPU/PyTorch engineering evidence')
    if len({(r['gpu'],r['torch']) for r in reports.values()})!=1:
        raise ValueError('GPU architecture/PyTorch mismatch; no training started')

def commands(args, worktree, scheme, out, data, checkpoint):
    check = [sys.executable, str(worktree/'repro/check_p_adaptation.py'), '--scheme', scheme, '--checkpoint', str(checkpoint), '--data-root', str(data), '--device', 'cuda', '--output-root', str(out.parent/'checks'/scheme)]
    train = [sys.executable, str(worktree/'repro/train_p_adaptation.py'), '--scheme', scheme,
             '--data-root', str(data), '--output-dir', str(out), '--device', 'cuda',
             '--checkpoint', str(checkpoint), '--checks-report', str(out.parent/'checks'/scheme/'check_report.json'), '--steps', '2000', '--steps-per-epoch', '100',
             '--batch-size', '4', '--patch-lr', '48', '--seed', '1', '--lr', '1e-5']
    if args.benchmark_final:
        train.append('--benchmark-final')
    return check, train

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--checkpoint', default=str(ROOT/'LFMN/model/scale4_model_939.pt'))
    parser.add_argument('--gpus', default='0', help='Comma-separated GPU IDs, each runs at most one process')
    parser.add_argument('--p0-ref', default=REFS['P0']); parser.add_argument('--p3-ref', default=REFS['P3']); parser.add_argument('--p4-ref', default=REFS['P4'])
    parser.add_argument('--benchmark-final', action='store_true')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--no-fetch', action='store_true', help='Use already resolved local refs')
    args = parser.parse_args()
    gpus = [g.strip() for g in args.gpus.split(',')]
    if not gpus or any(not g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU IDs must be distinct nonnegative integers')
    if os.name == 'nt' and not args.plan:
        parser.error('Real model checks/training must run on the Linux server; --plan is allowed locally')
    _STOP.clear()
    if not args.plan and not args.no_fetch:
        subprocess.run(['git', 'fetch', 'origin'], cwd=ROOT, check=True)
    refs = {'P0': args.p0_ref, 'P3': args.p3_ref, 'P4': args.p4_ref}
    commits = {scheme: git('rev-parse', ref+'^{commit}') for scheme,ref in refs.items()}
    model_hashes = {scheme: hashlib.sha256(subprocess.check_output(['git', 'show', commit+':LFMN/model/lfmn.py'], cwd=ROOT)).hexdigest() for scheme,commit in commits.items()}
    if len(set(model_hashes.values())) != 1:
        raise ValueError('Original lfmn.py differs across pinned refs')
    protocol_hashes = {scheme: hashlib.sha256(subprocess.check_output(['git', 'show', commit+':repro/p_adaptation_protocol.json'], cwd=ROOT)).hexdigest() for scheme,commit in commits.items()}
    if len(set(protocol_hashes.values())) != 1:
        raise ValueError('Protocol files differ across pinned refs')
    shared_hashes={}
    for source in ('LFMN/utility.py', 'repro/train_p_adaptation.py', 'repro/p_adapt_logging.py', 'repro/run_p_diagnostics.py'):
        hashes={scheme: hashlib.sha256(subprocess.check_output(['git','show',commit+':'+source],cwd=ROOT)).hexdigest() for scheme,commit in commits.items()}
        if len(set(hashes.values()))!=1:
            raise ValueError('Common source differs across pinned refs: '+source)
        shared_hashes[source]=hashes
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = ROOT/'experiment/all_runs'/('P_adapt_compare_'+stamp)
    workroot = ROOT/'experiment/p_adaptation_worktrees'/stamp
    plan = {'refs': refs, 'commits': commits, 'original_model_sha256': model_hashes, 'protocol_file_sha256': protocol_hashes,
            'common_source_sha256': shared_hashes, 'gpus': gpus, 'steps_per_scheme': 2000, 'total_training_updates': 6000,
            'check_phase': 'all checks before any training', 'benchmark_final': args.benchmark_final,
            'output_directory': str(output), 'worktree_directory': str(workroot),
            'mode': 'plan' if args.plan else 'server', 'status': 'planned'}
    if args.plan:
        print(json.dumps(plan, indent=2)); return
    data = data_root(args.data_root)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file() or digest(checkpoint) != WEIGHT_SHA:
        raise ValueError('Official initial checkpoint missing or SHA256 mismatch')
    for target in (output, workroot):
        if subprocess.run(['git', 'check-ignore', '--quiet', str(target)], cwd=ROOT).returncode:
            raise ValueError('Output and detached-worktree directories must be git-ignored')
    output.mkdir(parents=True, exist_ok=False); workroot.mkdir(parents=True, exist_ok=False)
    plan.update(status='preparing', data_root=str(data), checkpoint=str(checkpoint), initial_checkpoint_sha256=WEIGHT_SHA, jobs={})
    manifest = output/'launcher_manifest.json'; json_write(manifest,plan)
    try:
        for index, scheme in enumerate(REFS):
            worktree = workroot/scheme
            subprocess.run(['git', 'worktree', 'add', '--detach', str(worktree), commits[scheme]], cwd=ROOT, check=True)
            scheme_out=output/(scheme+'_x4_seed1_'+stamp)
            check,train = commands(args,worktree,scheme,scheme_out,data,checkpoint)
            plan['jobs'][scheme] = {'commit': commits[scheme], 'gpu': gpus[index%len(gpus)], 'worktree': str(worktree), 'output_directory': str(scheme_out), 'check_command': check, 'train_command': train, 'status': 'pending_check'}
        plan['status']='checking'; json_write(manifest,plan)
        # Checks are sequential. All must succeed before any group receives updates.
        for scheme,job in plan['jobs'].items():
            invoke(job['check_command'],job['worktree'],job['gpu'],output/(scheme+'_check_launcher.log'))
            job['status']='checked'; json_write(manifest,plan)
        reports={scheme: json.loads((output/'checks'/scheme/'check_report.json').read_text(encoding='utf-8')) for scheme in REFS}
        validate_check_reports(reports,model_hashes,shared_hashes,data)
        p4_source=hashlib.sha256(subprocess.check_output(['git','show',commits['P4']+':LFMN/model/lfmnp4.py'],cwd=ROOT)).hexdigest()
        if reports['P4'].get('candidate_source_sha256')!=p4_source:
            raise ValueError('P4 engineering report differs from pinned candidate source')
        plan['engineering_environment']={'gpu':reports['P0']['gpu'],'torch':reports['P0']['torch']}
        plan['status']='training'; json_write(manifest,plan)
        lanes = [[] for _ in gpus]
        for index,scheme in enumerate(REFS): lanes[index%len(gpus)].append(scheme)
        def lane(schemes):
            try:
                for scheme in schemes:
                    if _STOP.is_set():
                        return
                    job=plan['jobs'][scheme]
                    with _LOCK:
                        job['status']='training'; json_write(manifest,plan)
                    invoke(job['train_command'],job['worktree'],job['gpu'],output/(scheme+'_train_launcher.log'))
                    with _LOCK:
                        job['status']='completed'; json_write(manifest,plan)
            except BaseException:
                stop_owned()
                raise
        pool=ThreadPoolExecutor(max_workers=len(gpus))
        futures=[]
        try:
            futures=[pool.submit(lane,schemes) for schemes in lanes if schemes]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop_owned()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
        subprocess.run([sys.executable, str(ROOT/'repro/summarize_p_adaptation.py'), '--runs', *(plan['jobs'][s]['output_directory'] for s in REFS), '--output', str(output/'summary.json')], cwd=ROOT, check=True)
        plan['status']='completed'
    except BaseException as error:
        plan.update(status='failed', error=repr(error))
        stop_owned()
        raise
    finally:
        plan['end_time']=datetime.now().isoformat(); json_write(manifest,plan)
    print(output)

if __name__ == '__main__':
    main()
