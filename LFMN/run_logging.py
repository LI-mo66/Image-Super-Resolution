"""Durable console capture and epoch records for the common training entry."""
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def launch(command, directory, config, cwd, resume=False):
    directory = Path(directory).resolve()
    if resume:
        if not (directory / 'config.json').is_file():
            raise ValueError('resume requires an existing config.json')
        previous = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
        sessions = previous.get('sessions', [])
        config = dict(previous, **config)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        sessions = []
    session = {'start_time': datetime.datetime.now().astimezone().isoformat(),
               'resume': resume, 'command': command}
    sessions.append(session)
    config.update(status='running', sessions=sessions, output_directory=str(directory))
    write_json(directory / 'config.json', config)
    metrics = directory / 'metrics.csv'
    if not metrics.exists():
        metrics.write_text('epoch,learning_rate,train_loss,validation_psnr,validation_ssim,elapsed_seconds,set5_psnr,set5_ssim\n', encoding='utf-8')
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
               TQDM_MININTERVAL='5', LFMN_MANAGED_RUN=str(directory))
    process = None
    started = time.monotonic()
    with (directory / 'train_log.txt').open('ab', buffering=0) as log:
        header = ('\nSESSION ' + json.dumps(config, ensure_ascii=False) + '\n').encode('utf-8')
        log.write(header)
        print(header.decode('utf-8'), flush=True)
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            while True:
                chunk = process.stdout.read1(8192)
                if not chunk:
                    break
                log.write(chunk)
                if hasattr(sys.stdout, 'buffer'):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                else:
                    sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                    sys.stdout.flush()
            code = process.wait()
            status = 'completed' if code == 0 else 'failed'
        except KeyboardInterrupt:
            status, code = 'interrupted', 130
            if process is not None:
                process.terminate()
                remaining, _ = process.communicate()
                if remaining:
                    log.write(remaining)
        except BaseException:
            import traceback
            log.write(traceback.format_exc().encode('utf-8'))
            status, code = 'failed', 1
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
            raise
        finally:
            on_disk = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
            config.update(on_disk)
            config['sessions'] = sessions
            session.update(end_time=datetime.datetime.now().astimezone().isoformat(),
                           status=status, exit_code=code,
                           elapsed_seconds=time.monotonic() - started)
            config.update(status=status, exit_code=code, end_time=session['end_time'])
            with metrics.open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            valid = [row for row in rows if row.get('validation_psnr')]
            if valid:
                best = max(valid, key=lambda row: float(row['validation_psnr']))
                config.update(final_metrics=valid[-1], best_metrics=best,
                              best_epoch=int(best['epoch']))
            write_json(directory / 'config.json', config)
            footer = '\nEND {} code={} final={} best_epoch={}\n'.format(
                status, code, config.get('final_metrics'), config.get('best_epoch'))
            log.write(footer.encode('utf-8'))
            print(footer, flush=True)
    if code:
        raise subprocess.CalledProcessError(code, command)


def restore_training_state(trainer, loader, directory, epoch):
    import random
    import numpy as np
    import torch
    path = Path(directory) / 'resume_state.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['epoch'] != epoch:
        raise ValueError('resume_state epoch does not match requested checkpoint')
    if trainer.optimizer.get_last_epoch() != epoch:
        raise ValueError('scheduler/checkpoint boundary mismatch; preserve the interrupted run')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    loader.loader_train.generator.set_state(state['loader'])
    trainer.error_last = state['error_last']
    print('Restored RNG/DataLoader state at epoch {}'.format(epoch), flush=True)


def record_epoch(trainer, loader, directory, learning_rate, elapsed):
    import random
    import numpy as np
    import torch
    directory = Path(directory)
    epoch = trainer.optimizer.get_last_epoch()
    loss = float(trainer.loss.log[-1, -1])
    psnr = float(trainer.ckp.log[-1, 0, 0])
    ssim = float(trainer.ckp.log_ssim[-1, 0, 0])
    if not all(np.isfinite(value) for value in (loss, psnr, ssim)):
        raise FloatingPointError('non-finite epoch loss or validation metric')
    path = directory / 'metrics.csv'
    with path.open(encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = list(reader)
    if rows and int(rows[-1]['epoch']) >= epoch:
        raise ValueError('metrics epoch would duplicate or overwrite history')
    set5_psnr, set5_ssim = '', ''
    if 'Set5' in trainer.args.data_test:
        image_metrics = torch.load(
            directory / 'per_image_metrics' / ('epoch_{:04d}.pt'.format(epoch)),
            map_location='cpu', weights_only=True,
        )
        set5_rows = [row for row in image_metrics if row['dataset'] == 'Set5' and row['scale'] == 4]
        if len(set5_rows) != 5 or len({row['filename'] for row in set5_rows}) != 5:
            raise ValueError('Set5 epoch report requires five distinct x4 images')
        set5_psnr = sum(row['psnr'] for row in set5_rows) / 5
        set5_ssim = sum(row['ssim'] for row in set5_rows) / 5
        if not np.isfinite(set5_psnr) or not np.isfinite(set5_ssim):
            raise FloatingPointError('non-finite Set5 metrics')
    state = dict(epoch=epoch, python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 loader=loader.loader_train.generator.get_state(), error_last=trainer.error_last)
    temporary = directory / 'resume_state.pt.tmp'
    torch.save(state, temporary)
    temporary.replace(directory / 'resume_state.pt')
    with path.open('a', newline='', encoding='utf-8') as stream:
        row = dict(epoch=epoch, learning_rate=learning_rate, train_loss=loss,
                   validation_psnr=psnr, validation_ssim=ssim, elapsed_seconds=elapsed,
                   set5_psnr=set5_psnr, set5_ssim=set5_ssim)
        csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore').writerow(row)
        stream.flush()
    print('EPOCH_RECORD epoch={} loss={} validation_PSNR={} SSIM={} checkpoint={}'.format(
        epoch, loss, psnr, ssim, directory / 'model' / ('model_{}.pt'.format(epoch))), flush=True)
    if set5_psnr != '':
        print('SET5_EPOCH epoch={} PSNR={:.9f} SSIM={:.9f} (monitor only)'.format(
            epoch, set5_psnr, set5_ssim), flush=True)
