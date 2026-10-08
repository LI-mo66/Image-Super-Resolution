"""Fixed historical V1 prefix protocol; no baseline optimizer command exists."""
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = {g: 'lfmn_v1'+g for g in ('n21', 'n22', 'n23')}
TEACHER_SHA = '129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a'
TEACHER_COMMIT = '6545850fbf8df298df73d81f3e8cba638787c8bd'
EXPECTED = dict(model='LFMNSRPRV2', n_threads='8', data_train="['DIV2K']", data_test="['DIV2K']",
    data_range='1-800/801-900', scale='[4]', patch_size='256', batch_size='4', seed='1',
    ext='img', rgb_range='255', precision='single', no_augment='False', n_GPUs='1', cpu='False',
    scheduler='cosine', scheduler_t_max='150', eta_min='1e-06', lr='0.0002', optimizer='ADAM',
    betas='(0.9, 0.999)', epsilon='1e-08', weight_decay='0', gclip='0', loss='1*L1',
    test_every='1000', max_train_batches='0', chop='False', self_ensemble='False',
    rgcrd_mode='output', rgcrd_lambda_output='0.1', rgcrd_teacher_amp='False',
    rgcrd_teacher_microbatch='1', rgcrd_grad_diag_every='1', pre_train='', test_only='False',
    save_per_image_metrics='True')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def config_blocks(path):
    blocks, block = [], {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        if key in block:
            blocks.append(block)
            block = {}
        block[key] = value
    if block:
        blocks.append(block)
    if not blocks:
        raise ValueError('Empty config')
    return blocks


def validate_v1_config(path):
    blocks = config_blocks(path)
    for index, block in enumerate(blocks):
        for key, value in EXPECTED.items():
            if block.get(key) != value:
                raise ValueError(f'V1 block {index}: {key}={block.get(key)!r}; expected {value!r}')
        if block.get('epochs') not in ('20', '40', '150'):
            raise ValueError('Unknown historical V1 stage')
        resume = int(block.get('resume', '-1'))
        # Historical initial scratch config predates the replay option. Missing
        # is inert ONLY when resume=0; continuation must record replay explicitly.
        replay = int(block.get('resume_data_epochs', '0' if resume == 0 else '-1'))
        if resume != replay or (resume == 0 and block.get('load')) or (resume and not block.get('load')):
            raise ValueError('Invalid historical resume/data replay')
        if resume not in (0, 20, 40):
            raise ValueError('Unknown historical resume')
    return blocks


def command(group, data, teacher_repo, teacher_weight, output, smoke=False):
    if group not in MODELS:
        raise ValueError(group)
    return [sys.executable, '-u', str(ROOT/'repro/v1_combo_train_entry.py'),
        '--model', MODELS[group], '--dir_data', str(data), '--data_train', 'DIV2K', '--data_test', 'DIV2K',
        '--data_range', '1-2/859-859' if smoke else '1-800/801-900', '--scale', '4',
        '--patch_size', '256', '--batch_size', '4', '--seed', '1', '--n_threads', '0' if smoke else '8',
        '--ext', 'img', '--rgb_range', '255', '--precision', 'single', '--n_GPUs', '1',
        '--epochs', '1' if smoke else '20', '--test_every', '1000', '--max_train_batches', '2' if smoke else '0',
        '--lr', '0.0002', '--optimizer', 'ADAM', '--epsilon', '1e-8',
        '--weight_decay', '0', '--gclip', '0', '--loss', '1*L1', '--scheduler', 'cosine',
        '--scheduler_t_max', '150', '--eta_min', '1e-6', '--resume', '0', '--resume_data_epochs', '0',
        '--rgcrd_mode', 'output', '--rgcrd_lambda_output', '0.1', '--rgcrd_teacher_microbatch', '1',
        '--rgcrd_grad_diag_every', '1', '--rgcrd_teacher_repo', str(teacher_repo),
        '--rgcrd_teacher_checkpoint', str(teacher_weight), '--save_per_image_metrics',
        '--experiment_root', str(output.parent), '--save', output.name, '--print_every', '100']
