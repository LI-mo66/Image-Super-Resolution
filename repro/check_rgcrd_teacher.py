#!/usr/bin/env python3
"""Load the official teacher and verify its output/feature contract."""
import argparse
import hashlib
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'LFMN'))
from rgcrd.teacher import SwinIRTeacher
from rgcrd.criterion import RGCRDCriterion
from model.lfmnrgcrd import Net as Student

EXPECTED_SHA256 = '129dc773ba2d4c07f3eb0bb116fbe692011b7cc072d9ca12797cd3748198610a'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--repo', required=True)
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--teacher-only', action='store_true')
args = parser.parse_args()

actual_hash = sha256(args.checkpoint)
if actual_hash != EXPECTED_SHA256:
    raise RuntimeError('teacher SHA256 mismatch: {}'.format(actual_hash))
device = torch.device('cpu' if args.cpu else 'cuda')
teacher_args = SimpleNamespace(
    rgcrd_teacher_repo=args.repo,
    rgcrd_teacher_checkpoint=args.checkpoint,
    rgcrd_teacher_amp=args.amp,
)
teacher = SwinIRTeacher(teacher_args, device)
image = torch.rand(1, 3, 16, 16, device=device) * 255
output, features = teacher(image)
assert output.shape == (1, 3, 64, 64)
assert features['stage4'].shape[-2:] == (16, 16)
assert features['stage8'].shape[-2:] == (16, 16)
assert not any(parameter.requires_grad for parameter in teacher.parameters())
print('PASS official SwinIR teacher contract; SHA256={}'.format(actual_hash))

if not args.teacher_only:
    criterion_args = SimpleNamespace(
        rgcrd_mode='full', scale=[4], rgcrd_lambda_output=0.1,
        rgcrd_lambda_rel=0.1, rgcrd_lambda_evo=0.05,
        rgcrd_local_windows='4+8', rgcrd_global_grid=8,
        rgcrd_reliability_pixel=0.5, rgcrd_reliability_low=0.25,
        rgcrd_reliability_grad=0.25,
        rgcrd_reliability_temperature=0.25,
        rgcrd_reliability_margin=0.0,
        rgcrd_reliability_smooth=3,
    )
    student = Student(scale=4).to(device).train()
    criterion = RGCRDCriterion(criterion_args).to(device)
    axis = torch.linspace(0, 255, 32, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    lr = torch.stack((xx, yy, 0.5 * (xx + yy)), dim=0).unsqueeze(0)
    hr = torch.nn.functional.interpolate(
        lr, scale_factor=4, mode='bicubic', align_corners=False
    ).clamp(0, 255)
    student_sr, student_features = student(lr)
    teacher_sr, teacher_features = teacher(lr)
    tensors = {
        'student_sr': student_sr, 'teacher_sr': teacher_sr,
        'student_stage4': student_features['stage4'],
        'student_stage8': student_features['stage8'],
        'teacher_stage4': teacher_features['stage4'],
        'teacher_stage8': teacher_features['stage8'],
    }
    nonfinite_tensors = [
        name for name, value in tensors.items()
        if not torch.isfinite(value).all()
    ]
    if nonfinite_tensors:
        raise RuntimeError('non-finite forward tensors: {}'.format(nonfinite_tensors))
    auxiliary, stats = criterion(
        student_sr, student_features, teacher_sr, teacher_features, hr
    )
    (torch.nn.functional.l1_loss(student_sr, hr) + auxiliary).backward()
    named_gradients = [
        (name, parameter.grad) for name, parameter in student.named_parameters()
        if parameter.grad is not None
    ]
    gradients = [item for _, item in named_gradients]
    nonfinite = [
        name for name, item in named_gradients
        if not torch.isfinite(item).all()
    ]
    if nonfinite:
        raise RuntimeError(
            'non-finite student gradients (first 10): {}; auxiliary={}; stats={}'.format(
                nonfinite[:10], float(auxiliary.detach()),
                {key: float(value) for key, value in stats.items()},
            )
        )
    assert gradients
    assert sum(item.abs().sum() for item in gradients) > 0
    assert all(torch.isfinite(value) for value in stats.values())
    print(
        'PASS end-to-end RGCRD step: loss={:.6f}, gate={:.4f}'.format(
            auxiliary.detach(), stats['gate_mean']
        )
    )
