"""Private V1N23C entry: unchanged framework, finite/update/save audits."""
import hashlib
import json
from pathlib import Path
import sys
import types
from v1n23c_startup import audit_updates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'LFMN'))


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    import torch
    import data
    import model
    import loss
    import utility
    from option import args
    from trainer import Trainer
    from model.lfmnsrprv2 import Net as V1
    if args.model.lower() not in ('lfmn_v1n23c',):
        raise ValueError('This entry never trains V1 or B0')
    if args.load or args.pre_train or args.resume or args.resume_data_epochs:
        raise ValueError('Scratch candidates only; no implicit resume/warm start')
    destination = (Path(args.experiment_root) / args.save).resolve()
    destination.relative_to((ROOT / 'experiment/all_runs').resolve())
    if destination.exists():
        raise FileExistsError(destination)
    torch.manual_seed(args.seed)
    # Do not force strict CUDA determinism/TF32 changes: preserve old V1 defaults.
    with torch.random.fork_rng(devices=[]):
        reference = V1(scale=4)
        reference_rng = torch.get_rng_state().clone()
    checkpoint = utility.checkpoint(args)
    loader = data.Data(args)
    student = model.Model(args, checkpoint)
    net = student.model
    assert torch.equal(torch.get_rng_state(), reference_rng), 'Init RNG drift'
    for name, value in reference.state_dict().items():
        assert torch.equal(value, net.state_dict()[name].cpu()), name
    old_names = set(dict(reference.named_parameters()))
    new = {n: p for n, p in net.named_parameters() if n not in old_names}
    del reference
    assert new and sum(p.numel() for p in net.parameters()) <= 841563 * 1.02
    before = {n: p.detach().clone() for n, p in new.items()}
    objective = loss.Loss(args, checkpoint)
    trainer = Trainer(args, loader, student, objective, checkpoint)
    optimizer = trainer.optimizer
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len(set(map(id, params))) == len(list(net.parameters()))
    assert set(map(id, params)) == set(map(id, net.parameters()))
    stage_calls = []
    handles = [m.register_forward_hook(
        lambda m, x, y, index=index: stage_calls.append(index))
        for index, m in enumerate(net.proximal)]
    native_forward = net.forward
    audit = {'steps': 0, 'active_new_first_two': [], 'first_train_batch': None}
    mechanism = {}
    first_eval = {}

    def audited_forward(x, *a, **kw):
        stage_calls.clear()
        capture = (net.training and audit['steps'] == 0) or (not net.training and not first_eval)
        net.set_collect_statistics(capture)
        result = native_forward(x, *a, **kw)
        if capture:
            mechanism['train_first' if net.training else 'eval_first'] = net.mechanism_statistics()
        net.set_collect_statistics(False)
        if stage_calls != list(range(8)):
            raise RuntimeError('Missing/reordered SRPR writeback')
        if tuple(result.shape) != (x.shape[0], 3, x.shape[2]*4, x.shape[3]*4):
            raise RuntimeError('Output shape mismatch')
        if not torch.isfinite(result).all():
            raise RuntimeError('Nonfinite student output')
        if not net.training and not first_eval:
            first_eval.update(lr=x.detach().clone(), raw_hash=tensor_hash(result))
        return result

    net.forward = audited_forward
    native_prepare = trainer.prepare

    def audited_prepare(*values):
        if net.training and audit['first_train_batch'] is None and len(values) == 2:
            audit['first_train_batch'] = [tensor_hash(x) for x in values]
        return native_prepare(*values)

    trainer.prepare = audited_prepare
    native_step = optimizer.step

    def audited_step(this, *a, **kw):
        gradients = [p.grad for p in net.parameters() if p.grad is not None]
        # Single GPU->CPU synchronization, not one sync per parameter.
        if not gradients or not torch.isfinite(torch.stack(torch._foreach_norm(gradients))).all():
            raise RuntimeError('Nonfinite gradient')
        if audit['steps'] < 2:
            active = [n for n, p in new.items() if p.grad is not None and bool(p.grad.abs().sum() > 0)]
            if not active:
                raise RuntimeError('No active new gradient')
            if audit['steps'] == 1 and set(active) != set(new):
                raise RuntimeError('New second-step gradient inactive: '+','.join(set(new)-set(active)))
            audit['active_new_first_two'].append(len(active))
        result = native_step(*a, **kw)
        audit['steps'] += 1
        return result

    optimizer.step = types.MethodType(audited_step, optimizer)
    optimizer.step.__func__._wrapped_by_lr_sched = True
    try:
        while not trainer.terminate():
            audit.update(steps=0, active_new_first_two=[], first_train_batch=None)
            mechanism.clear()
            first_eval.clear()
            torch.cuda.reset_peak_memory_stats() if not args.cpu else None
            learning_rate = optimizer.get_lr()
            trainer.train()
            expected_steps = min(len(loader.loader_train), args.max_train_batches) if args.max_train_batches else len(loader.loader_train)
            if audit['steps'] != expected_steps or audit['steps'] < 2:
                raise RuntimeError('Unexpected update count')
            changed,rounding = audit_updates(net,new,before,optimizer)
            trainer.test()
            epoch = optimizer.get_last_epoch()
            saved = torch.load(destination / 'model' / f'model_{epoch}.pt', map_location=student.device, weights_only=True)
            net.load_state_dict(saved, strict=True)
            with torch.no_grad():
                if tensor_hash(net(first_eval['lr'])) != first_eval['raw_hash']:
                    raise RuntimeError('Saved checkpoint does not reproduce same-device eval')
            record = dict(epoch=epoch, lr=learning_rate, **audit, calibrated_writeback=dict(mechanism),
                new_tensors_changed=len(changed),sub_ulp_norm_updates=rounding,
                new_tensors=len(new), gates={n: float(p.detach()) for n, p in new.items() if 'gain' in n and p.numel()==1},
                checkpoint_sha256=hashlib.sha256((destination / 'model' / f'model_{epoch}.pt').read_bytes()).hexdigest(),
                strict_reload=True, peak_allocated=torch.cuda.max_memory_allocated() if not args.cpu else None)
            with (destination / 'mechanism.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            print('FRAMEWORK_AUDIT ' + json.dumps(record), flush=True)
            # Do not retain a full validation image across training epochs.
            first_eval.clear()
    finally:
        for handle in handles:
            handle.remove()
        checkpoint.done()


if __name__ == '__main__':
    main()
