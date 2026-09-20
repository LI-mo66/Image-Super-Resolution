"""Frozen official SwinIR-M x4 teacher with normalized-depth feature taps."""
import importlib
import os
import sys

import torch
import torch.nn as nn


class SwinIRTeacher(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        repo = os.path.abspath(args.rgcrd_teacher_repo)
        checkpoint = os.path.abspath(args.rgcrd_teacher_checkpoint)
        network_file = os.path.join(repo, 'models', 'network_swinir.py')
        if not os.path.isfile(network_file):
            raise FileNotFoundError('Official SwinIR code not found: {}'.format(network_file))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError('SwinIR teacher checkpoint not found: {}'.format(checkpoint))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        network = importlib.import_module('models.network_swinir')
        self.teacher = network.SwinIR(
            upscale=4, in_chans=3, img_size=48, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv',
        )
        payload = torch.load(checkpoint, map_location='cpu')
        state = payload.get('params_ema', payload.get('params', payload))
        self.teacher.load_state_dict(state, strict=True)
        self.teacher.to(device).eval().requires_grad_(False)
        self.use_amp = bool(args.rgcrd_teacher_amp and device.type == 'cuda')

    def train(self, mode=True):
        super().train(False)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def forward(self, image):
        with torch.autocast(
            device_type=image.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            teacher = self.teacher
            original_h, original_w = image.shape[-2:]
            x = image / 255.0
            mean = teacher.mean.to(device=x.device, dtype=x.dtype)
            x = (x - mean) * teacher.img_range
            x = teacher.check_image_size(x)
            x_size = (x.shape[2], x.shape[3])
            x_first = teacher.conv_first(x)
            tokens = teacher.patch_embed(x_first)
            if teacher.ape:
                tokens = tokens + teacher.absolute_pos_embed
            tokens = teacher.pos_drop(tokens)
            stage4 = None
            for index, layer in enumerate(teacher.layers):
                tokens = layer(tokens, x_size)
                if index == 2:
                    stage4 = teacher.patch_unembed(teacher.norm(tokens), x_size)
            stage8 = teacher.patch_unembed(teacher.norm(tokens), x_size)
            body = teacher.conv_after_body(stage8) + x_first
            out = teacher.conv_before_upsample(body)
            out = teacher.conv_last(teacher.upsample(out))
            out = out / teacher.img_range + mean
            out = out[..., :original_h * 4, :original_w * 4] * 255.0
            stage4 = stage4[..., :original_h, :original_w]
            stage8 = stage8[..., :original_h, :original_w]
        return out.float(), {'stage4': stage4.float(), 'stage8': stage8.float()}
