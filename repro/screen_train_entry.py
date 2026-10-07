"""Private, common training entry: same safety/precision checks for all groups."""
import hashlib
import json
from pathlib import Path
import random
import runpy
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.set_num_threads(4)
seed = int(sys.argv[sys.argv.index('--seed') + 1])
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
import model
import utility
import data
from model.lfmn import Net as Baseline
from trainer import Trainer


def state_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode())
        digest.update(state[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


original_build = model.Model._build_model
original_data_init = data.Data.__init__
def checked_data(self,args):
    original_data_init(self,args)
    for subset,text in [(self.loader_train.dataset.datasets[0],args.data_range.split('/')[0]),
                        (self.loader_test[0].dataset,args.data_range.split('/')[1])]:
        first,last = map(int,text.split('-'))
        actual = [Path(path).stem for path in subset.images_hr]
        assert actual == [f'{i:04d}' for i in range(first,last+1)], 'Dataset scan/range identity mismatch'
data.Data.__init__ = checked_data

def checked_build(self, module, args):
    with torch.random.fork_rng(devices=[]):
        expected = Baseline(scale=args.scale[0])
        expected_rng = torch.get_rng_state()
    net = original_build(self, module, args)
    assert torch.equal(expected_rng, torch.get_rng_state()), 'Model initialization alters data RNG'
    expected_state = expected.state_dict()
    actual = net.state_dict()
    for key, value in expected_state.items():
        assert torch.equal(value, actual[key]), f'Initial baseline weight mismatch: {key}'
    initial_hash = state_hash(expected_state)
    parameters = sum(p.numel() for p in net.parameters())
    assert parameters <= 759627 * 1.02, 'Parameter overhead exceeds 2%'
    proof = {'common_initial_state_sha256': initial_hash, 'parameters':parameters,
             'model':args.model, 'tf32':False, 'seed':seed}
    Path(args.save, 'initial_state_proof.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    def finite_forward(_module, _input, output):
        if not torch.isfinite(output).all():
            raise FloatingPointError('Nonfinite model output')
        if _module.training:
            calls = getattr(_module,'screen_training_calls',0)
            every = args.max_train_batches or 1000
            if calls % every == 0:
                batch_hash = hashlib.sha256(_input[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                with Path(args.save,'batch_fingerprints.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'epoch':calls//every+1,'lr_sha256':batch_hash})+'\n')
            _module.screen_training_calls = calls + 1
    net.register_forward_hook(finite_forward)
    return net
model.Model._build_model = checked_build

original_optimizer = utility.make_optimizer
def checked_optimizer(args, target):
    optimizer = original_optimizer(args,target)
    original_step = optimizer.step
    optimizer.checked_steps = 0
    def step(*a, **kw):
        gradients = [p.grad.detach() for p in target.parameters() if p.grad is not None]
        if not gradients or not torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all():
            raise FloatingPointError('Nonfinite/missing gradients')
        result = original_step(*a, **kw)
        optimizer.checked_steps += 1
        return result
    step._wrapped_by_lr_sched = getattr(original_step,'_wrapped_by_lr_sched',False)
    optimizer.step = step
    return optimizer
utility.make_optimizer = checked_optimizer

original_train = Trainer.train
def logged_train(self):
    before = self.optimizer.checked_steps
    original_train(self)
    count = self.optimizer.checked_steps - before
    if not torch.isfinite(self.loss.log[-1]).all():
        raise FloatingPointError('Nonfinite training loss')
    expected_steps = self.args.max_train_batches or len(self.loader_train)
    assert count == expected_steps, (count,expected_steps)
    if not torch.isfinite(torch.stack(torch._foreach_norm(list(self.model.parameters())))).all():
        raise FloatingPointError('Nonfinite parameters')
    mechanism = {name:float(p.detach()) for name,p in self.model.named_parameters()
                 if name.endswith(('inheritance_gain','prior_gain'))}
    record = {'epoch':self.optimizer.get_last_epoch(), 'updates':count,
              'total_updates':self.optimizer.checked_steps, 'raw_gains':mechanism,
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
    with Path(self.ckp.dir,'mechanism.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
Trainer.train = logged_train

runpy.run_path(str(ROOT / 'LFMN/main.py'),run_name='__main__')
