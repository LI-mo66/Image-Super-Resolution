"""Explicit paired screening or N23-only collection with baseline pending.

Default is audit-only. No implicit B0 training, resume, overwrite or shutdown.
"""
import argparse
import ast
import datetime
import hashlib
import importlib
import json
import os
import math
import re
from pathlib import Path
import shutil
import subprocess
import sys

os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMON = 'b198de7'
OLD_ENTRY = 'd43cdb8f5483884e22badb1ed7f1febee4d57286'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args, binary=False):
    result = subprocess.check_output(['git', *args], cwd=ROOT)
    return result if binary else result.decode().strip()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_hashes():
    paths = sorted(list((ROOT/'LFMN').rglob('*.py')) + list((ROOT/'repro').glob('*n23*.py')) +
                   [ROOT/'repro/check_overlap_protocol.py', ROOT/'repro/check_overlap_fast.py'])
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if '__pycache__' not in p.parts}


def data_hashes(root):
    records = {}
    for split, ids in [('train', range(1, 801)), ('valid', range(801, 901))]:
        for index in ids:
            for rel in [f'DIV2K/DIV2K_{split}_HR/{index:04d}.png',
                        f'DIV2K/DIV2K_{split}_LR_bicubic/X4/{index:04d}x4.png']:
                path = root/rel
                if not path.is_file():
                    raise FileNotFoundError(f'Missing data: {path}')
                records[rel] = sha(path)
    return records


def environment():
    raw = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'], text=True).strip()
    name, memory, driver = [s.strip() for s in raw.split(',')]
    modules = ('numpy', 'einops', 'PIL', 'imageio', 'skimage', 'cv2', 'matplotlib', 'tqdm')
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=name, driver=driver,
                total_mib=int(memory), python=sys.version, executable=sys.executable,
                libraries={n: str(getattr(importlib.import_module(n), '__version__', 'unknown')) for n in modules})


def gpu_jobs():
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    if any(s.strip() and not s.strip().isdigit() for s in lines):
        raise RuntimeError('Cannot establish GPU job ownership')
    return [int(s.strip()) for s in lines if s.strip() and int(s.strip()) != os.getpid()]


def config(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if ': ' in line:
            key, value = line.split(': ', 1)
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            if key in result:
                raise ValueError(f'Duplicate config field: {key}')
            result[key] = value
    return result


def expected_config(threads):
    return dict(data_train=['DIV2K'], data_test=['DIV2K'], data_range='1-800/801-900',
                scale=[4], patch_size=256, batch_size=4, epochs=40, test_every=1000, seed=1,
                ext='img', rgb_range=255, n_colors=3, n_threads=threads, lr=0.0002,
                scheduler='cosine', scheduler_t_max=150, eta_min=1e-6, optimizer='ADAM',
                betas=(0.9, 0.999), weight_decay=0.0, epsilon=1e-8, loss='1*L1',
                precision='single', pre_train='', load='', resume=0, no_augment=False,
                self_ensemble=False, chop=False, cpu=False, n_GPUs=1, gclip=0,
                max_train_batches=0, rgcrd_mode='off')


def audit_schedule(directory):
    scheduler = torch.load(directory/'scheduler.pt', map_location='cpu', weights_only=True)
    assert scheduler['T_max'] == 150 and scheduler['eta_min'] == 1e-6
    assert scheduler['last_epoch'] == 40 and scheduler['base_lrs'] == [2e-4]
    expected_lr = 1e-6 + (2e-4 - 1e-6) * (1 + math.cos(math.pi * 40 / 150)) / 2
    assert abs(scheduler['_last_lr'][0] - expected_lr) < 1e-12
    optimizer = torch.load(directory/'optimizer.pt', map_location='cpu', weights_only=True)
    assert len(optimizer['param_groups']) == 1
    group = optimizer['param_groups'][0]
    assert abs(group['lr'] - expected_lr) < 1e-12
    assert tuple(group['betas']) == (0.9, 0.999) and group['eps'] == 1e-8 and group['weight_decay'] == 0
    assert all(int(row['step']) == 40000 for row in optimizer['state'].values())
    rates = re.findall(r'\[Epoch (\d+)\]\s+Learning rate:\s*([\d.eE+-]+)',
                       (directory/'log.txt').read_text(encoding='utf-8'))
    assert len(rates) == 40
    for epoch, (logged_epoch, rate) in enumerate(rates, 1):
        theory = 1e-6 + (2e-4 - 1e-6) * (1 + math.cos(math.pi * (epoch - 1) / 150)) / 2
        assert int(logged_epoch) == epoch and float(rate) == float(f'{theory:.2e}')
    return dict(status='PASS', scheduler_sha256=sha(directory/'scheduler.pt'),
                optimizer_sha256=sha(directory/'optimizer.pt'), logged_epochs=40)


def audit_baseline(directory, data, env):
    old = json.loads((directory.parent/'manifest.json').read_text(encoding='utf-8'))
    if old.get('status') not in ('COMPLETE', 'COMPLETE_SHUTDOWN_REQUESTED', 'COMPLETE_SHUTDOWN_FAILED'):
        raise ValueError('Old queue did not complete successfully')
    if old.get('group_exit_codes', {}).get('b0') != 0:
        raise ValueError('Missing successful B0 exit status')
    p = old['protocol']
    assert p['smoke'] is False and p['epochs'] == 40 and p['updates_per_epoch'] == 1000
    assert p['training_ids'] == list(range(1, 801)) and p['validation_ids'] == list(range(801, 901))
    assert p['baseline_source'] == COMMON and p['baseline_reuse'] is False
    threads = int(p['threads'])
    assert threads > 0
    expected = expected_config(threads)
    actual = config(directory/'config.txt')
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f'B0 config mismatch: {key}={actual.get(key)!r}, expected {value!r}')
    assert actual['model'].lower() == 'lfmn'
    schedule = audit_schedule(directory)
    if old['data_hashes'] != data:
        raise ValueError('Old/current dataset hashes differ')
    for key in ('torch', 'cuda', 'gpu', 'driver', 'python', 'libraries'):
        if old['environment'].get(key) != env[key]:
            raise ValueError(f'Old/current environment differs: {key}; do not silently cross protocols')
    # Audit actual old transitive baseline source against the agreed common
    # commit. New patch_reverse repair is registered separately and does not
    # change the fixed LR64 training graph (check_overlap_protocol.py).
    core = ['LFMN/main.py', 'LFMN/trainer.py', 'LFMN/utility.py', 'LFMN/option.py',
            'LFMN/template.py', 'LFMN/model/__init__.py', 'LFMN/model/lfmn.py']
    core += [s for s in git('ls-tree', '-r', '--name-only', COMMON, 'LFMN/data', 'LFMN/loss').splitlines()
             if s.endswith('.py')]
    for name in core:
        registered = git('show', f'{COMMON}:{name}', binary=True)
        previous = git('show', f'{old["commit"]}:{name}', binary=True)
        if previous != registered or old['source_hashes'].get(name) != hashlib.sha256(previous).hexdigest():
            raise ValueError(f'Old B0 baseline source proof mismatch: {name}')
        if name != 'LFMN/model/lfmn.py':
            current = (ROOT/name).read_bytes().replace(b'\r\n', b'\n')
            if current != registered.replace(b'\r\n', b'\n'):
                raise ValueError(f'Current common training source changed: {name}')
    registered_repair = git('show', '2ec1170:LFMN/model/lfmn.py', binary=True).replace(b'\r\n', b'\n')
    assert (ROOT/'LFMN/model/lfmn.py').read_bytes().replace(b'\r\n', b'\n') == registered_repair
    entry = 'repro/screen_train_entry.py'
    trusted = git('show', f'{OLD_ENTRY}:{entry}', binary=True)
    previous = git('show', f'{old["commit"]}:{entry}', binary=True)
    assert previous == trusted
    assert old['source_hashes'][entry] == hashlib.sha256(previous).hexdigest()
    updates = [json.loads(s) for s in (directory/'mechanism.jsonl').read_text().splitlines()]
    assert len(updates) == 40
    for epoch, row in enumerate(updates, 1):
        assert row['epoch'] == epoch and row['updates'] == 1000 and row['total_updates'] == epoch * 1000
    batches = [json.loads(s) for s in (directory/'batch_fingerprints.jsonl').read_text().splitlines()]
    assert [s['epoch'] for s in batches] == list(range(1, 41))
    assert all(len(s['lr_sha256']) == 64 for s in batches)
    initial = json.loads((directory/'initial_state_proof.json').read_text())
    assert initial['parameters'] == 759627 and initial['tf32'] is False and initial['seed'] == 1
    # Checkpoint/log existence and shape, never use these old curves as scores.
    hashes = {}
    sys.path.insert(0, str(ROOT/'LFMN'))
    from model.lfmn import Net as Baseline
    with torch.random.fork_rng(devices=[]):
        checker = Baseline(scale=4)
    for epoch in range(1, 41):
        path = directory/'model'/f'model_{epoch}.pt'
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state and all(torch.isfinite(v).all() for v in state.values())
        checker.load_state_dict(state, strict=True)
        initialized = [v for k, v in state.items() if k.endswith('.initted')]
        assert len(initialized) == 8 and all(bool(v.item()) for v in initialized)
        hashes[str(epoch)] = sha(path)
    for name in ('psnr_log.pt', 'ssim_log.pt'):
        values = torch.load(directory/name, map_location='cpu', weights_only=True)
        assert values.shape == (40, 1, 1) and torch.isfinite(values).all()
    return dict(status='PASS', directory=str(directory), old_manifest_sha256=sha(directory.parent/'manifest.json'),
                old_commit=old['commit'], checkpoint_sha256=hashes, threads=threads,
                initial_state_sha256=initial['common_initial_state_sha256'],
                schedule_audit=schedule,
                batch_fingerprints=batches, baseline_reuse='YES_WEIGHTS_ONLY_REEVALUATE_ALL_EPOCHS',
                training_equivalence='Registered LR64 exact_coverage_v1 graph; preflight regression required',
                expected_config=expected)


def run(cmd, logfile, cwd=ROOT, extra_env=None):
    print('RUN', ' '.join(map(str, cmd)), flush=True)
    child_env = dict(os.environ)
    for key in ('N23_BASELINE_MANIFEST', 'N23_PAIR_MANIFEST', 'N23_PROTOCOL_PROBE_ONLY'):
        child_env.pop(key, None)
    child_env.update({'PYTHONUNBUFFERED': '1', **(extra_env or {})})
    with logfile.open('w', encoding='utf-8') as stream:
        child = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=child_env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        rc = child.wait()
        os.fsync(stream.fileno())
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def training_command(args, threads, group='n23'):
    model = {'n23': 'lfmn_n23', 'b0': 'lfmn_exact_overlap'}[group]
    return [sys.executable, ROOT/'repro/n23_train_entry.py', '--model', model,
            '--dir_data', args.data_root, '--data_train', 'DIV2K', '--data_test', 'DIV2K',
            '--data_range', '1-800/801-900', '--scale', '4', '--patch_size', '256',
            '--batch_size', '4', '--n_threads', str(threads), '--ext', 'img', '--epochs', '40',
            '--test_every', '1000', '--lr', '2e-4', '--scheduler', 'cosine',
            '--scheduler_t_max', '150', '--eta_min', '1e-6', '--optimizer', 'ADAM',
            '--epsilon', '1e-8', '--weight_decay', '0', '--loss', '1*L1', '--seed', '1',
            '--precision', 'single', '--print_every', '100', '--save_models',
            '--save_per_image_metrics', '--save', args.output/group]


def shutdown_command():
    helper = Path('/usr/bin/shutdown')
    if sys.platform != 'linux' or not helper.is_file():
        raise RuntimeError('AutoDL helper missing; shutdown unavailable')
    prefix = helper.read_bytes()[:128]
    if prefix.startswith((b'\x7fELF', b'#!')):
        if not os.access(helper, os.X_OK):
            raise RuntimeError('Shutdown helper not executable')
        return [str(helper)]
    # AutoDL may supply shell text without a shebang. Check syntax without
    # executing it, then use bash explicitly to avoid Exec format error.
    subprocess.run(['/bin/bash', '-n', str(helper)], check=True)
    return ['/bin/bash', str(helper)]


def complete_run(path, manifest, poweroff, single=False):
    manifest['status'] = 'COMPLETE_N23_ONLY_BASELINE_PENDING' if single else 'COMPLETE'
    manifest['completed_utc'] = now()
    write(path, manifest)
    if poweroff:
        if gpu_jobs():
            raise RuntimeError('Other GPU jobs prevent shutdown; results retained')
        manifest['status'] = 'COMPLETE_SHUTDOWN_REQUESTED'
        write(path, manifest)
        os.sync()
        subprocess.run(poweroff, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    baseline_choice = ap.add_mutually_exclusive_group(required=True)
    baseline_choice.add_argument('--baseline', type=Path, help='Old n21_n22.../b0 directory with parent manifest')
    baseline_choice.add_argument('--train-new-baseline', action='store_true',
                                 help='Explicit separate choice: train a new 40e B0; never automatic fallback')
    baseline_choice.add_argument('--n23-only', action='store_true',
                                 help='Only N23 40e; no old B0 access or B0 training; comparison pending')
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--run', action='store_true', help='Launch selected mode: N23-only 40e, reused pair, or new pair 2x40e')
    ap.add_argument('--shutdown-on-success', action='store_true')
    ap.add_argument('--dedicated-instance', action='store_true')
    args = ap.parse_args()
    args.data_root, args.output = args.data_root.resolve(), args.output.resolve()
    if args.baseline:
        args.baseline = args.baseline.resolve()
    args.output.relative_to((ROOT/'experiment/all_runs').resolve())
    if args.output == (ROOT/'experiment/all_runs').resolve() or args.output.exists():
        raise FileExistsError('Output exists: refuse overwrite/resume; choose a new run name')
    for source in (s for s in (args.data_root, args.baseline) if s is not None):
        if source == args.output or source in args.output.parents or args.output in source.parents:
            ap.error('Results must be disjoint from data and old baseline')
    if args.shutdown_on_success and (not args.run or not args.dedicated_instance):
        ap.error('Shutdown requires --run and --dedicated-instance')
    poweroff = shutdown_command() if args.shutdown_on_success else None
    if git('status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('Dirty checkout; do not start against uncommitted source')
    if os.environ.get('CUDA_VISIBLE_DEVICES', '0') != '0' or not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA_VISIBLE_DEVICES=0 and CUDA')
    env = environment()
    if args.run and (gpu_jobs() or env['total_mib'] < 16 * 1024):
        raise RuntimeError('Use idle dedicated GPU with >=16GiB VRAM; recommend 24GiB')
    print('Auditing all 1800 data files; no optimizer started', flush=True)
    data = data_hashes(args.data_root)
    audit = (dict(status='BASELINE_PENDING', baseline_reuse='NOT_REQUESTED_N23_ONLY',
                  threads=4, expected_config=expected_config(4)) if args.n23_only else
             audit_baseline(args.baseline, data, env) if args.baseline else
             dict(status='NEW_BASELINE_EXPLICITLY_REQUESTED', baseline_reuse='NO_EXPLICIT_NEW_PAIR',
                  threads=4, expected_config=expected_config(4)))
    protocol = dict(mode='scratch', epochs=40, updates_per_epoch=1000, threads=audit['threads'],
                    overlap='exact_coverage_v1', common_source=COMMON,
                    expected_config=audit['expected_config'],
                    metric='full quantized RGB PSNR border10; quantized Y SSIM border4; no chop/x8')
    print(json.dumps({'baseline_reuse': audit['baseline_reuse'], 'protocol': protocol}, indent=2), flush=True)
    if not args.run:
        print('AUDIT_PASS: no output created, no re-evaluation/training/shutdown. Add --run only to launch.', flush=True)
        return
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise RuntimeError('Need >=5GiB free result storage')
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output/'manifest.json'
    hashes = source_hashes()
    manifest = dict(status='PREPARING', commit=git('rev-parse', 'HEAD'), source_hashes=hashes,
                    baseline_audit=audit, data_hashes=data, environment=env, protocol=protocol,
                    data_root=str(args.data_root), output=str(args.output), started_utc=now(),
                    comparison_audit={'status': 'PENDING'}, shutdown_requested=bool(poweroff))
    manifest['run_mode'] = 'N23_ONLY' if args.n23_only else 'PAIRED_SCREEN'
    write(path, manifest)
    try:
        # Target environment gates run in children: controller never owns CUDA.
        run([sys.executable, ROOT/'repro/check_n23.py', '--data', args.data_root/'DIV2K'], args.output/'engineering.log')
        run([sys.executable, ROOT/'repro/check_overlap_protocol.py', '--data', args.data_root/'DIV2K'], args.output/'overlap_protocol.log')
        run([sys.executable, ROOT/'repro/check_overlap_fast.py'], args.output/'overlap_fast.log')
        smoke = [sys.executable, ROOT/'repro/check_n23_smoke.py', '--data-root', args.data_root,
                 '--output', args.output/'smoke']
        if args.n23_only:
            smoke.append('--n23-only')
        run(smoke, args.output/'smoke.log')
        if args.baseline:
            smoke_proof = json.loads((args.output/'smoke/n23/initial_state_proof.json').read_text())
            assert smoke_proof['legacy_common_initial_state_sha256'] == audit['initial_state_sha256'], 'Old/new initial state mismatch; do not screen'
            probe = training_command(args, audit['threads'])
            probe[probe.index('--save')+1] = args.output/'protocol_probe'
            run(probe, args.output/'protocol_probe.log', ROOT/'LFMN', extra_env={
                'N23_BASELINE_MANIFEST': str(path), 'N23_PROTOCOL_PROBE_ONLY': '1'})
        efficiency_checkpoint = (args.output/'smoke/n23/model/model_1.pt' if args.n23_only else
                                args.baseline/'model/model_40.pt' if args.baseline else args.output/'smoke/b0/model/model_1.pt')
        efficiency_hash = audit['checkpoint_sha256']['40'] if args.baseline else sha(efficiency_checkpoint)
        run([sys.executable, ROOT/'repro/check_n23_efficiency.py',
             '--n23-only-checkpoint' if args.n23_only else '--baseline-checkpoint',
             efficiency_checkpoint, '--expected-sha256', efficiency_hash,
             '--cache-miss', '--output', args.output/'efficiency/summary.json'], args.output/'efficiency.log')
        if gpu_jobs():
            raise RuntimeError('Another GPU job appeared during preparation')
        manifest['status'] = 'VERIFIED'
        write(path, manifest)
        if args.baseline:
            run([sys.executable, ROOT/'repro/evaluate_n23_b0.py', '--manifest', path], args.output/'b0_reeval.log')
        manifest['status'] = 'RUNNING_N23_ONLY_BASELINE_PENDING' if args.n23_only else 'SCREENING'
        write(path, manifest)
        run(training_command(args, audit['threads']), args.output/'n23_console.log', ROOT/'LFMN',
            extra_env={'N23_BASELINE_MANIFEST': str(path)} if args.baseline else None)
        if args.n23_only:
            audit_schedule(args.output/'n23')
            assert source_hashes() == hashes and data_hashes(args.data_root) == data, 'Source/data changed during run'
            manifest['comparison_audit'] = {'status': 'BASELINE_PENDING', 'n23_integrity': 'PASS'}
            write(path, manifest)
            run([sys.executable, ROOT/'repro/summarize_n23_only.py', args.output], args.output/'summary.log')
            complete_run(path, manifest, poweroff, single=True)
            return
        if not args.baseline:
            candidate_proof = json.loads((args.output/'n23/initial_state_proof.json').read_text())
            manifest['pair_reference'] = dict(kind='CURRENT_PAIR_DATAFLOW_NOT_OLD_BASELINE_REUSE',
                initial_state_sha256=candidate_proof['legacy_common_initial_state_sha256'],
                batch_fingerprints=[json.loads(s) for s in (args.output/'n23/batch_fingerprints.jsonl').read_text().splitlines()])
            assert len(manifest['pair_reference']['batch_fingerprints']) == 40
            write(path, manifest)
            run(training_command(args, audit['threads'], 'b0'), args.output/'b0_console.log', ROOT/'LFMN',
                extra_env={'N23_PAIR_MANIFEST': str(path)})
            write(args.output/'b0/evaluation_protocol.json', {'protocol': protocol,
                  'note': 'Explicit new B0, same current training and evaluation protocol'})
        proof = json.loads((args.output/'n23/initial_state_proof.json').read_text())
        if args.baseline:
            assert proof['legacy_common_initial_state_sha256'] == audit['initial_state_sha256'], 'Old/new initial B0 state mismatch'
        else:
            baseline_proof = json.loads((args.output/'b0/initial_state_proof.json').read_text())
            for key in ('common_initial_state_sha256', 'retained_state_sha256', 'cpu_rng_sha256'):
                assert proof[key] == baseline_proof[key], key
        batches = [json.loads(s) for s in (args.output/'n23/batch_fingerprints.jsonl').read_text().splitlines()]
        assert len(batches) == 40
        baseline_batches = (audit['batch_fingerprints'] if args.baseline else
                            [json.loads(s) for s in (args.output/'b0/batch_fingerprints.jsonl').read_text().splitlines()])
        assert [(r['epoch'], r['lr_sha256']) for r in batches] == [(r['epoch'], r['lr_sha256']) for r in baseline_batches]
        if not args.baseline:
            assert [r['hr_sha256'] for r in batches] == [r['hr_sha256'] for r in baseline_batches]
            audit_schedule(args.output/'b0')
        audit_schedule(args.output/'n23')
        assert source_hashes() == hashes and data_hashes(args.data_root) == data, 'Source/data changed during queue'
        if args.baseline:
            assert sha(args.baseline.parent/'manifest.json') == audit['old_manifest_sha256']
            for epoch, value in audit['checkpoint_sha256'].items():
                assert sha(args.baseline/'model'/f'model_{epoch}.pt') == value
        manifest['comparison_audit'] = {'status': 'PASS', 'init_and_first_batch_all_epochs': 'MATCH',
                                        'old_weights_reassessed_on_same_current_device': bool(args.baseline)}
        write(path, manifest)
        run([sys.executable, ROOT/'repro/summarize_n23_screen.py', args.output], args.output/'summary.log')
        complete_run(path, manifest, poweroff)
    except BaseException as error:
        manifest.update(status='FAILED_NO_SHUTDOWN' if manifest['status'] != 'COMPLETE_SHUTDOWN_REQUESTED'
                        else 'COMPLETE_SHUTDOWN_FAILED', error=repr(error), ended_utc=now())
        write(path, manifest)
        raise


if __name__ == '__main__':
    main()
