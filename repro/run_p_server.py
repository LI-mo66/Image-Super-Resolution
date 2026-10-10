"""Server-only P0/P1/P2 dispatcher with isolated refs and one process per GPU.

Fetch the three branches before invocation. --plan performs read-only planning.
Default execution evaluates preregistered diagnostic crops, never trains.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
REFS = {'P0': 'origin/codex/p0-lfmn-diagnostics',
        'P1': 'origin/codex/p1-sfml-diagnostics',
        'P2': 'origin/codex/p2-iasa-diagnostics'}
PLUGINS = {'P1': 'repro/p1_sfml_probe.py', 'P2': 'repro/p2_iasa_probe.py'}
WEIGHTS = {2: 'scale2_model_996.pt', 3: 'scale3_model_969.pt', 4: 'scale4_model_939.pt'}


def default_data_root():
    candidates = [Path('/root/autodl-tmp/dataset'), Path('/root/autodl-tmp/datasets'),
                  Path('/root/autodl-tmp')]
    for path in candidates:
        if (path / 'DIV2K').is_dir() and (path / 'benchmark').is_dir():
            return path
    return Path('/root/autodl-tmp/datasets')


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT, text=True).strip()


def json_save(path, data):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def build_plan(args, run_root):
    refs = {scheme: getattr(args, scheme.lower() + '_ref') for scheme in args.schemes}
    commits = {scheme: git('rev-parse', ref + '^{commit}') for scheme, ref in refs.items()}
    blobs = {scheme: git('rev-parse', commit + ':LFMN/model/lfmn.py') for scheme, commit in commits.items()}
    if len(set(blobs.values())) != 1:
        raise ValueError('P0/P1/P2 original model blobs differ; refuse comparison')
    paths = {scheme: ROOT / 'experiment/p_server_worktrees' / f'{scheme.lower()}_{commit[:12]}'
             for scheme, commit in commits.items()}
    jobs = []
    for scale in args.scales:
        checkpoint = ROOT / 'LFMN/model' / WEIGHTS[scale]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for scheme in args.schemes:
            worktree = paths[scheme]
            command = [args.python, '-u', str(worktree / 'repro/run_p_diagnostics.py'),
                       '--scheme', scheme, '--project-root', str(ROOT), '--data-root', str(args.data_root),
                       '--output-root', str(run_root / scheme / f'x{scale}'),
                       '--checkpoint', str(checkpoint), '--scale', str(scale), '--device', 'cuda',
                       '--crop-lr', '64' if args.mode == 'diagnose' else '0',
                       '--benchmarks', 'smoke' if args.mode == 'diagnose' else 'full', '--div2k-ids']
            if args.mode == 'diagnose' and scale == 4:
                command.extend(str(i) for i in range(801, 809))
            if scheme in PLUGINS:
                command.extend(['--plugin', str(worktree / PLUGINS[scheme])])
            jobs.append({'scheme': scheme, 'scale': scale, 'commit': commits[scheme], 'ref': refs[scheme],
                         'worktree': str(worktree), 'command': command,
                         'gpu': args.gpus[len(jobs) % len(args.gpus)], 'status': 'pending'})
    return {'mode': args.mode, 'training': False, 'root': str(ROOT), 'output': str(run_root),
            'data_root': str(args.data_root), 'refs': refs, 'commits': commits, 'model_blobs': blobs,
            'jobs': jobs, 'gpu_policy': 'one running child per listed GPU; lanes execute sequentially',
            'scope': 'DIV2K and benchmark smoke crops' if args.mode == 'diagnose' else 'full five benchmarks'}


def prepare(plan):
    seen = set()
    for job in plan['jobs']:
        path = Path(job['worktree'])
        if str(path) in seen:
            continue
        seen.add(str(path))
        if path.exists():
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
            dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()
            if head != job['commit'] or dirty:
                raise RuntimeError(f'Existing checkout has different commit or local changes: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'worktree', 'add', '--detach', str(path), job['commit']], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--data-root', type=Path, default=default_data_root())
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--scales', nargs='+', type=int, choices=(2, 3, 4), default=[4])
    parser.add_argument('--schemes', nargs='+', choices=('P0', 'P1', 'P2'), default=['P0', 'P1', 'P2'])
    parser.add_argument('--mode', choices=('diagnose', 'benchmark'), default='diagnose')
    parser.add_argument('--plan', action='store_true')
    for scheme, ref in REFS.items():
        parser.add_argument('--' + scheme.lower() + '-ref', default=ref)
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.schemes)) != len(args.schemes) or len(set(args.scales)) != len(args.scales):
        parser.error('GPU IDs, schemes and scales must be unique')
    if args.mode == 'benchmark' and args.schemes != ['P0']:
        parser.error('First full benchmark pass is P0 only; P1/P2 are preregistered crop diagnostics')
    if not args.plan and os.name == 'nt':
        parser.error('Execution is server-only. Windows local use supports --plan only.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_root = ROOT / 'experiment/all_runs' / f'P_compare_{args.mode}_{stamp}'
    plan = build_plan(args, run_root)
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    prepare(plan)
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = run_root / 'launcher_manifest.json'
    plan['status'] = 'running'
    json_save(manifest, plan)
    print(f'Results: {run_root}', flush=True)
    lock = threading.Lock()
    children = []
    stop = threading.Event()

    def lane(gpu):
        for job in (j for j in plan['jobs'] if j['gpu'] == gpu):
            if stop.is_set():
                break
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env['PYTHONUNBUFFERED'] = '1'
            with lock:
                job['status'] = 'running'
                job['start_time'] = datetime.now().isoformat()
                json_save(manifest, plan)
            log_path = run_root / f'{job["scheme"]}_x{job["scale"]}_launcher.log'
            try:
                with log_path.open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(job['command'], cwd=job['worktree'], env=env,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, encoding='utf-8', errors='replace', bufsize=1)
                    with lock:
                        children.append(child)
                        job['pid'] = child.pid
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        with lock:
                            print(f'[{job["scheme"]} x{job["scale"]} GPU{gpu}] {line}', end='', flush=True)
                    code = child.wait()
                with lock:
                    job['exit_code'] = code
                    job['status'] = 'completed' if code == 0 else 'failed'
                    job['end_time'] = datetime.now().isoformat()
                    json_save(manifest, plan)
                if code:
                    break
            except BaseException as exc:
                with lock:
                    job['status'] = 'failed'
                    job['error'] = repr(exc)
                    json_save(manifest, plan)
                raise

    try:
        pool = ThreadPoolExecutor(max_workers=len(args.gpus))
        try:
            futures = [pool.submit(lane, gpu) for gpu in args.gpus]
            for future in futures:
                future.result()
        except BaseException:
            stop.set()
            for child in children:
                if child.poll() is None:
                    child.terminate()
            raise
        finally:
            pool.shutdown(wait=True)
        plan['status'] = 'completed' if all(j['status'] == 'completed' for j in plan['jobs']) else 'failed'
        json_save(manifest, plan)
        if plan['status'] != 'completed':
            raise RuntimeError('At least one diagnostic failed or remained pending; retain logs, do not report gains')
        for scale in args.scales:
            runs = []
            for scheme in args.schemes:
                candidates = list((run_root / scheme / f'x{scale}').glob('*/config.json'))
                if len(candidates) != 1:
                    raise RuntimeError('Expected exactly one child run per scheme and scale')
                runs.append(str(candidates[0].parent))
            summary_command = [args.python, str(ROOT / 'repro/summarize_p_diagnostics.py'), '--runs', *runs,
                               '--output', str(run_root / f'summary_x{scale}.json')]
            subprocess.run(summary_command, cwd=ROOT, check=True)
        print(f'Completed. Return the folder: {run_root}', flush=True)
    except BaseException:
        for child in children:
            if child.poll() is None:
                child.terminate()
        plan['status'] = 'failed_or_interrupted'
        json_save(manifest, plan)
        raise


if __name__ == '__main__':
    main()
