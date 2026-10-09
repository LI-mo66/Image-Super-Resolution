#!/usr/bin/env python3
"""Managed F2 entry: original trainer, epoch checkpoint-linked Set5 records."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
import numpy as np
import torch
import data
import loss
import model
import utility
from option import args
from trainer import Trainer
from run_logging import record_epoch, restore_training_state, write_json


def append_csv(path, rows):
    path = Path(path)
    existing = []
    if path.exists():
        with path.open(encoding='utf-8', newline='') as stream:
            existing = list(csv.DictReader(stream))
    if existing and int(existing[-1]['epoch']) >= int(rows[0]['epoch']):
        raise ValueError('checkpoint metric timeline would duplicate or go backwards')
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not existing:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()


def export_epoch(directory, epoch):
    checkpoint = directory / 'model' / ('model_{}.pt'.format(epoch))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = torch.load(directory / 'per_image_metrics' / ('epoch_{:04d}.pt'.format(epoch)),
                      map_location='cpu', weights_only=True)
    set5 = [row for row in rows if row['dataset'] == 'Set5']
    if len(set5) != 5 or len({row['filename'] for row in set5}) != 5:
        raise ValueError('Set5 must contain five unique image records for every checkpoint')
    for row in rows:
        if not np.isfinite(row['psnr']) or not np.isfinite(row['ssim']):
            raise FloatingPointError('nonfinite per-image metric')
    detail = [dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                   filename=row['filename'], psnr=row['psnr'], ssim=row['ssim'],
                   scale=4, self_ensemble=False) for row in set5]
    mean = dict(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                set5_psnr=sum(row['psnr'] for row in set5) / 5,
                set5_ssim=sum(row['ssim'] for row in set5) / 5,
                image_count=5, scale=4, self_ensemble=False)
    append_csv(directory / 'set5_per_image.csv', detail)
    append_csv(directory / 'set5_per_checkpoint.csv', [mean])
    print('SET5_CHECKPOINT epoch={} PSNR={:.10f} SSIM={:.10f} sha256={}'.format(
          epoch, mean['set5_psnr'], mean['set5_ssim'], digest), flush=True)
    return rows


def main():
    managed = os.environ.get('LFMN_MANAGED_RUN')
    if not managed:
        raise RuntimeError('Use repro/run_f2_screen_server.py for automatic durable logs')
    if (args.self_ensemble or args.chop or args.n_GPUs != 1 or
            args.rgcrd_mode != 'off' or args.precision != 'single' or args.scale != [4]):
        raise ValueError('F2 requires x4 FP32, one GPU/process, ensemble/chop/auxiliary loss OFF')
    if args.model.lower() not in ('lfmnf2', 'lfmn'):
        raise ValueError('Only F2 and original LFMN are supported')
    if not args.test_only and args.pre_train:
        raise ValueError('Registered F2 training uses scratch initialization')
    if args.epochs > 20 and not args.test_only:
        raise ValueError('This delivery pauses at epoch20; continuation needs separate approval')
    directory = Path(managed)
    if args.load:
        state = torch.load(directory / 'resume_state.pt', map_location='cpu', weights_only=False)
        if state['epoch'] != args.resume:
            raise ValueError('resume requires the last fully recorded checkpoint boundary')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config_path = directory / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['resolved_arguments'] = vars(args)
    write_json(config_path, config)
    print('Resolved arguments: {}'.format(vars(args)), flush=True)
    checkpoint = utility.checkpoint(args)
    if args.load and checkpoint.resume_epoch != args.resume:
        raise ValueError('validation history differs from resume boundary')
    loader = data.Data(args)
    net = model.Model(args, checkpoint)
    source = None
    if args.load:
        source = directory / 'model' / ('model_{}.pt'.format(args.resume))
    elif args.pre_train:
        source = Path(args.pre_train)
    if source:
        state = torch.load(source, map_location='cpu', weights_only=True)
        net.model.load_state_dict(state, strict=True)
        config['input_checkpoint_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        shared = hashlib.sha256()
        for key, tensor in sorted(net.model.state_dict().items()):
            if any(key.startswith('blocks.{}.1.layer.1.fn.'.format(i)) for i in range(8)):
                continue
            shared.update(key.encode())
            shared.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        config['initial_unchanged_state_sha256'] = shared.hexdigest()
    config['parameter_count'] = sum(p.numel() for p in net.parameters())
    write_json(config_path, config)
    print('Parameters: {}'.format(config['parameter_count']), flush=True)
    criterion = None if args.test_only else loss.Loss(args, checkpoint)
    trainer = Trainer(args, loader, net, criterion, checkpoint)
    if args.load and not args.test_only:
        restore_training_state(trainer, loader, directory, checkpoint.resume_epoch)
    try:
        while not trainer.terminate():
            next_epoch = trainer.optimizer.get_last_epoch() + 1
            next_path = directory / 'model' / ('model_{}.pt'.format(next_epoch))
            if next_path.exists():
                raise FileExistsError('Refusing to overwrite checkpoint: {}'.format(next_path))
            started = time.monotonic()
            learning_rate = trainer.optimizer.get_lr()
            trainer.train()
            trainer.test()  # DIV2K first (selection), Set5 second (observation only)
            epoch = trainer.optimizer.get_last_epoch()
            export_epoch(directory, epoch)
            record_epoch(trainer, loader, directory, learning_rate, time.monotonic() - started)
    finally:
        checkpoint.done()


if __name__ == '__main__':
    main()
