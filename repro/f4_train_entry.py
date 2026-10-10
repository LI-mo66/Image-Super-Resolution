#!/usr/bin/env python3
"""Managed F4 entry using the repository's unchanged training operations."""
import os
import hashlib
import json
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


def main():
    directory = os.environ.get('LFMN_MANAGED_RUN')
    if not directory:
        raise RuntimeError('Run through repro/run_f4_screen_server.py for durable logs')
    if args.self_ensemble or args.n_GPUs != 1 or args.rgcrd_mode != 'off':
        raise ValueError('F4 protocol requires one GPU/process, ensemble OFF, no auxiliary loss')
    if args.load:
        state_path = Path(directory) / 'resume_state.pt'
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['epoch'] != args.resume:
            raise ValueError('resume epoch differs from last complete RNG/checkpoint boundary')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    config_path = Path(directory) / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    config['resolved_arguments'] = vars(args)
    write_json(config_path, config)
    print('Resolved training arguments: {}'.format(vars(args)), flush=True)
    checkpoint = utility.checkpoint(args)
    if args.load and checkpoint.resume_epoch != args.resume:
        raise ValueError('validation history differs from requested resume epoch')
    loader = data.Data(args)
    reference_digest = None
    if not args.load and not args.pre_train:
        from model.lfmn import Net as BaselineNet
        with torch.random.fork_rng(devices=[]):
            reference = BaselineNet(scale=4)
            reference_digest = hashlib.sha256()
            for key, tensor in sorted(reference.state_dict().items()):
                reference_digest.update(key.encode('utf-8'))
                reference_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            reference_digest = reference_digest.hexdigest()
            del reference
    net = model.Model(args, checkpoint)
    if not args.load and not args.pre_train:
        digest = hashlib.sha256()
        for key, tensor in sorted(net.model.state_dict().items()):
            if 'residual_gates.' in key:
                continue
            digest.update(key.encode('utf-8'))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        config['initial_shared_state_sha256'] = digest.hexdigest()
        config['same_environment_B0_initial_shared_state_sha256'] = reference_digest
        config['matches_historical_B0_initialization'] = (digest.hexdigest() == config.get('historical_B0_initial_shared_state_sha256'))
        expected = reference_digest
        if expected and expected != digest.hexdigest():
            raise ValueError('B0/F4 shared scratch initialization differs')
        write_json(config_path, config)
    source = None
    if args.load:
        source = Path(checkpoint.dir) / 'model' / ('model_{}.pt'.format(args.resume))
    elif args.pre_train:
        source = Path(args.pre_train)
    if source:
        state = torch.load(source, map_location='cpu', weights_only=True)
        net.model.load_state_dict(state, strict=True)
        config['input_checkpoint_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        write_json(config_path, config)
    print('Parameters: {}'.format(sum(p.numel() for p in net.parameters())), flush=True)
    if net.self_ensemble or net.chop:
        raise ValueError('Actual model forward must use ensemble/chop OFF')
    def forbidden_x8(*inputs, **kwargs):
        raise AssertionError('Self-ensemble was entered during F4 OFF evaluation')
    net.forward_x8 = forbidden_x8
    criterion = None if args.test_only else loss.Loss(args, checkpoint)
    trainer = Trainer(args, loader, net, criterion, checkpoint)
    if args.load and not args.test_only:
        restore_training_state(trainer, loader, directory, checkpoint.resume_epoch)
    try:
        while not trainer.terminate():
            started = time.monotonic()
            learning_rate = trainer.optimizer.get_lr()
            trainer.train()
            trainer.test()
            record_epoch(trainer, loader, directory, learning_rate, time.monotonic() - started)
    finally:
        checkpoint.done()


if __name__ == '__main__':
    main()
