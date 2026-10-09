#!/usr/bin/env python3
"""Authorized independent B0/F2 full cosine150 entry; original trainer operations."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'LFMN'))
import numpy as np
import torch
import data
import loss
import model
import utility
from option import args
from trainer import Trainer
from run_logging import record_epoch,restore_training_state,write_json
from f2_train_entry import export_epoch,append_csv
from f2_150_common import read_csv


class FirstBatchAudit:
    """Observe, without changing, the CPU tensors delivered by the original loader."""
    def __init__(self,loader,directory):
        self.loader=loader;self.directory=directory;self.epoch=0
    def __len__(self):return len(self.loader)
    def __getattr__(self,key):return getattr(self.loader,key)
    @staticmethod
    def tensor_hash(tensor):
        t=tensor.detach().cpu().contiguous()
        result=hashlib.sha256()
        result.update(str(t.dtype).encode());result.update(str(tuple(t.shape)).encode())
        result.update(t.numpy().tobytes())
        return result.hexdigest()
    def __iter__(self):
        for index,batch in enumerate(self.loader):
            if index==0:
                lr,hr,_=batch
                row=dict(epoch=self.epoch,lr_sha256=self.tensor_hash(lr),hr_sha256=self.tensor_hash(hr),
                         batch_size=int(lr.shape[0]),lr_shape=str(tuple(lr.shape)),hr_shape=str(tuple(hr.shape)))
                path=self.directory/'first_batch_hashes.csv'
                prior=read_csv(path) if path.exists() else []
                found=[r for r in prior if int(r['epoch'])==self.epoch]
                if found:
                    if len(found)!=1 or any(found[0][k]!=str(v) for k,v in row.items()):
                        raise ValueError('Restarted first batch differs from preserved epoch record')
                else:append_csv(path,[row])
                print('FIRST_BATCH epoch={} LR={} HR={}'.format(self.epoch,row['lr_sha256'],row['hr_sha256']),flush=True)
            yield batch


def export_validation(directory,epoch,raw,smoke):
    rows=[r for r in raw if r['dataset']=='DIV2K']
    expected={'0801'} if smoke else {str(i).zfill(4) for i in range(801,901)}
    if len(rows)!=len(expected) or {r['filename'] for r in rows}!=expected:
        raise ValueError('Independent validation image identities differ from registered range')
    checkpoint=directory/'model'/('model_{}.pt'.format(epoch))
    row=dict(epoch=epoch,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             validation_psnr=sum(r['psnr'] for r in rows)/len(rows),
             validation_ssim=sum(r['ssim'] for r in rows)/len(rows),image_count=len(rows),scale=4,self_ensemble=False)
    append_csv(directory/'validation_per_checkpoint.csv',[row])


def main():
    managed=os.environ.get('LFMN_MANAGED_RUN')
    if not managed:raise RuntimeError('Use repro/run_f2_150_server.py for durable logs')
    directory=Path(managed);config_path=directory/'config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('authorized_experiment')!='F2_B0_INDEPENDENT_COS150':
        raise ValueError('Missing registered 150e experiment authorization')
    if args.self_ensemble or args.chop or args.n_GPUs!=1 or args.rgcrd_mode!='off' or args.precision!='single' or args.scale!=[4]:
        raise ValueError('Registered run requires x4 FP32, one process/GPU, ensemble/chop/aux OFF')
    if args.model.lower() not in ('lfmn','lfmnf2'):raise ValueError('Only original B0 and F2 supported')
    spec=importlib.util.find_spec('model.lfmn')
    if spec is None or not spec.origin.endswith('.py'):raise RuntimeError('Registered Python lfmn.py baseline required')
    if args.epochs>150:raise ValueError('Maximum authorized epoch is150')
    smoke=bool(config.get('smoke'))
    if not args.test_only:
        if args.pre_train:raise ValueError('New150e runs must start scratch, not old20e checkpoints')
        if args.scheduler!='cosine' or args.scheduler_t_max!=150 or args.batch_size!=4 or args.patch_size!=256 or args.seed!=1:
            raise ValueError('Training configuration differs from registered protocol')
        expected_args={'optimizer':'ADAM','lr':2e-4,'eta_min':1e-6,'betas':(0.9,0.999),
                       'epsilon':1e-8,'weight_decay':0,'loss':'1*L1','rgb_range':255,
                       'no_augment':False,'gclip':0,'split_batch':1,'n_threads':4}
        if any(getattr(args,key)!=value for key,value in expected_args.items()):
            raise ValueError('Actual optimizer, loss or augmentation settings differ from registered protocol')
        if not smoke and args.epochs!=150:
            raise ValueError('Formal independent run must target all150 epochs')
        if args.data_test!=['DIV2K','Set5'] or args.data_train!=['DIV2K']:
            raise ValueError('DIV2K is independent validation; Set5 is observation only')
        if args.data_range!=('1-1/801-801' if smoke else '1-800/801-900'):
            raise ValueError('Data range differs from registered protocol')
        if args.max_train_batches!=(1 if smoke else 0) or args.test_every!=(1 if smoke else 1000):
            raise ValueError('Batch budget differs from registered protocol')
    if args.load:
        saved=torch.load(directory/'resume_state.pt',map_location='cpu',weights_only=False)
        if saved['epoch']!=args.resume:raise ValueError('Resume must use last complete RNG/checkpoint boundary')
    torch.manual_seed(1);random.seed(1);np.random.seed(1)
    config['resolved_arguments']=vars(args)
    write_json(config_path,config)
    print('Resolved arguments: {}'.format(vars(args)),flush=True)
    checkpoint=utility.checkpoint(args)
    if args.load and checkpoint.resume_epoch!=args.resume:raise ValueError('Validation/checkpoint boundary mismatch')
    loader=data.Data(args)
    if loader.loader_train is not None:
        loader.loader_train=FirstBatchAudit(loader.loader_train,directory)
        if len(loader.loader_train)!=(1 if smoke else 1000):raise ValueError('Unexpected actual batches per epoch')
    net=model.Model(args,checkpoint)
    source=directory/'model'/('model_{}.pt'.format(args.resume)) if args.load else Path(args.pre_train) if args.pre_train else None
    if source:
        net.model.load_state_dict(torch.load(source,map_location='cpu',weights_only=True),strict=True)
        config['input_checkpoint_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    else:
        digest=hashlib.sha256()
        for key,tensor in sorted(net.model.state_dict().items()):
            if any(key.startswith('blocks.{}.1.layer.1.fn.'.format(i)) for i in range(8)):continue
            digest.update(key.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        config['initial_unchanged_state_sha256']=digest.hexdigest()
        expected=config.get('expected_initial_unchanged_state_sha256')
        if expected and expected!=digest.hexdigest():raise ValueError('Paired unchanged scratch initialization differs')
    config['parameter_count']=sum(p.numel() for p in net.parameters())
    if config['parameter_count']!=759627:raise ValueError('Unexpected parameter budget')
    write_json(config_path,config)
    print('Parameters: {}'.format(config['parameter_count']),flush=True)
    criterion=None if args.test_only else loss.Loss(args,checkpoint)
    trainer=Trainer(args,loader,net,criterion,checkpoint)
    if args.load and not args.test_only:restore_training_state(trainer,loader,directory,checkpoint.resume_epoch)
    try:
        while not trainer.terminate():
            epoch=trainer.optimizer.get_last_epoch()+1
            target=directory/'model'/('model_{}.pt'.format(epoch))
            if target.exists():raise FileExistsError('Refusing to overwrite checkpoint: '+str(target))
            loader.loader_train.epoch=epoch
            started=time.monotonic();learning_rate=trainer.optimizer.get_lr()
            trainer.train();trainer.test()
            raw=export_epoch(directory,epoch)
            export_validation(directory,epoch,raw,smoke)
            record_epoch(trainer,loader,directory,learning_rate,time.monotonic()-started)
    finally:checkpoint.done()


if __name__=='__main__':main()
