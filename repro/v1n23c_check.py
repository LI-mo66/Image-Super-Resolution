"""V1N23C engineering only: two local Adam steps, not a training result."""
import argparse
import importlib
import io
import json
import hashlib
from types import SimpleNamespace
from pathlib import Path
import sys

import imageio.v2 as imageio
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'LFMN'))
from model.lfmnsrprv2 import Net as V1


def main(module_name):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--teacher-repo', type=Path)
    ap.add_argument('--teacher-checkpoint', type=Path)
    ap.add_argument('--output', type=Path, help='New JSON file inside ignored experiment/all_runs')
    args = ap.parse_args()
    if args.output:
        args.output = args.output.resolve()
        args.output.relative_to((ROOT/'experiment/all_runs').resolve())
        assert not args.output.exists(), 'Do not overwrite evidence'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    factory = importlib.import_module('model.'+module_name).Net
    torch.manual_seed(17)
    reference = V1(scale=4)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(17)
    candidate = factory(scale=4)
    assert torch.equal(expected_rng, torch.get_rng_state()), 'Initialization/data RNG drift'
    original_keys = set(reference.state_dict())
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    old_params = set(dict(reference.named_parameters()))
    new = {k: p for k, p in candidate.named_parameters() if k not in old_params}
    assert new
    count = sum(p.numel() for p in candidate.parameters())
    assert count <= 841563*1.02, count
    device = torch.device(args.device)
    reference, candidate = reference.to(device).eval(), candidate.to(device).eval()
    errors = []
    with torch.no_grad():
        for shape in ((1,3,32,32), (2,3,33,47), (1,3,64,64)):
            image = torch.rand(*shape, device=device)*255
            target = reference(image)
            result = candidate(image)
            assert result.shape == (shape[0],3,shape[2]*4,shape[3]*4)
            assert torch.isfinite(result).all() and torch.equal(target,result), ('zero-gain mismatch',shape,float((target-result).abs().max()))
            candidate.set_enabled(False)
            assert torch.equal(target,candidate(image)), 'Disabled path does not restore V1'
            candidate.set_enabled(True)
            errors.append(float((target-result).abs().max()))
    # Two real images, shared top-left LR64 / HR256 crops. Teacher placeholder
    # tests the output-KD loss path only; no official teacher/quality claim.
    pairs = []
    for index in (1,2):
        lr = imageio.imread(args.data_root/f'DIV2K/DIV2K_train_LR_bicubic/X4/{index:04d}x4.png')
        hr = imageio.imread(args.data_root/f'DIV2K/DIV2K_train_HR/{index:04d}.png')
        pairs.append((torch.from_numpy(lr.copy()).permute(2,0,1).float()[:,:64,:64],
                      torch.from_numpy(hr.copy()).permute(2,0,1).float()[:,:256,:256]))
    lr = torch.stack([p[0] for p in pairs]).to(device)
    hr = torch.stack([p[1] for p in pairs]).to(device)
    del reference
    if bool(args.teacher_repo) != bool(args.teacher_checkpoint):
        raise ValueError('Provide both official teacher paths or neither')
    teacher_label = 'HR placeholder ONLY; official teacher not tested'
    teacher_hash = None
    teacher_sr = hr.detach()
    if args.teacher_repo:
        from rgcrd.teacher import SwinIRTeacher
        teacher_hash = hashlib.sha256(args.teacher_checkpoint.read_bytes()).hexdigest()
        assert teacher_hash == '129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a'
        with torch.random.fork_rng(devices=[]):
            teacher = SwinIRTeacher(SimpleNamespace(rgcrd_teacher_repo=str(args.teacher_repo),
                rgcrd_teacher_checkpoint=str(args.teacher_checkpoint), rgcrd_teacher_amp=False),device)
        assert all(not p.requires_grad for p in teacher.parameters())
        with torch.no_grad():
            teacher_sr, _ = teacher(lr)
        assert teacher_sr.shape == hr.shape and not teacher_sr.requires_grad
        assert torch.isfinite(teacher_sr).all()
        del teacher
        teacher_label = 'Frozen official SwinIR-M x4, FP32, output KD0.1'
    from check_n16_srpr_c1 import criterion_args
    from rgcrd.criterion import RGCRDCriterion
    criterion = RGCRDCriterion(criterion_args()).to(device)
    candidate.train()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=2e-4)
    grouped = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(grouped) == len(set(grouped)) == len(list(candidate.parameters()))
    before = {name:p.detach().clone() for name,p in new.items()}
    activity = []
    stage_calls = []
    handles = [module.register_forward_hook(
        lambda module, inputs, output, index=index: stage_calls.append(index))
        for index,module in enumerate(candidate.proximal)]
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        stage_calls.clear()
        output = candidate(lr)
        assert stage_calls == list(range(8)), 'SRPR stage writeback path was lost or reordered'
        kd, _ = criterion(output,None,teacher_sr,None,hr)
        loss = F.l1_loss(output,hr)+kd
        assert torch.isfinite(loss) and torch.isfinite(output).all()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new.values())
        active = [name for name,p in new.items() if bool(p.grad.abs().sum()>0)]
        assert active, 'No new learning path'
        activity.append({'step':step+1,'active_new_tensors':len(active),'total_new_tensors':len(new)})
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in candidate.parameters())
    for handle in handles:
        handle.remove()
    changed = [name for name,p in new.items() if not torch.equal(before[name],p.detach())]
    assert len(changed) == len(new), ('inactive new tensors after startup',set(new)-set(changed))
    candidate.eval()
    with torch.no_grad():
        image = lr[:1]
        on = candidate(image)
        reference = V1(scale=4).to(device).eval()
        reference.load_state_dict({k:v for k,v in candidate.state_dict().items() if k in original_keys}, strict=True)
        candidate.set_enabled(False)
        assert torch.equal(candidate(image),reference(image)), 'Post-update disabled V1 drift'
        candidate.set_enabled(True)
        assert torch.equal(on,candidate(image)), 'Transient state leaked across images'
        blob = io.BytesIO()
        torch.save(candidate.state_dict(),blob)
        blob.seek(0)
        reloaded = factory(scale=4).to(device).eval()
        reloaded.load_state_dict(torch.load(blob,map_location=device,weights_only=True),strict=True)
        assert torch.equal(on,reloaded(image)), 'Strict reload drift'
        effect = float((on-reference(image)).abs().max())
        assert effect > 0, 'Learned branch has no measurable output effect'
    report = dict(purpose='ENGINEERING_NOT_PSNR', model=module_name, v1_params=841563,
                  torch=str(torch.__version__), device=str(device),
                  data='DIV2K train0001/0002 top-left LR64/HR256; two engineering Adam steps',
                  precision='FP32, TF32 off', seed=17,
                  srpr_stages_called_in_order=list(range(1,9)),
                  params=count, parameter_delta=count-841563, shared_state_keys=len(original_keys),
                  initial_and_disabled_output_max=errors, optimizer_steps=2, batch=2,
                  teacher=teacher_label, teacher_sha256=teacher_hash,
                  gradient_startup=activity, new_tensors_changed=len(changed),
                  learned_output_effect_max=effect, strict_reload_max=0,
                  status='IMPLEMENTED_ENGINEERING_PRECHECK_PASS',
                  unverified=['full training-framework Gate2','full validation PSNR','AMP',
                              'server efficiency/FLOPs','existing V1 protocol reuse'])
    print(json.dumps(report,indent=2),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
