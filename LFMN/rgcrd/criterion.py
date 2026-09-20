"""Parameter-free reliability-gated relational distillation losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCRDCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.rgcrd_mode
        self.scale = int(args.scale[0])
        self.lambda_output = float(args.rgcrd_lambda_output)
        self.lambda_rel = float(args.rgcrd_lambda_rel)
        self.lambda_evo = float(args.rgcrd_lambda_evo)
        self.local_windows = tuple(
            int(item) for item in args.rgcrd_local_windows.split('+') if item
        )
        self.global_grid = int(args.rgcrd_global_grid)
        self.reliability_weights = (
            float(args.rgcrd_reliability_pixel),
            float(args.rgcrd_reliability_low),
            float(args.rgcrd_reliability_grad),
        )
        self.temperature = float(args.rgcrd_reliability_temperature)
        self.margin = float(args.rgcrd_reliability_margin)
        self.smooth = int(args.rgcrd_reliability_smooth)
        if not self.local_windows or min(self.local_windows) < 2:
            raise ValueError('RGCRD local windows must be integers >= 2')
        if self.global_grid < 2:
            raise ValueError('RGCRD global grid must be >= 2')
        if self.temperature <= 0:
            raise ValueError('RGCRD reliability temperature must be positive')
        if self.smooth < 1 or self.smooth % 2 == 0:
            raise ValueError('RGCRD reliability smoothing must be a positive odd integer')
        if any(weight < 0 for weight in self.reliability_weights):
            raise ValueError('RGCRD reliability weights must be non-negative')
        total = sum(self.reliability_weights)
        if total <= 0 or self.reliability_weights[0] <= 0:
            raise ValueError('RGCRD reliability needs a nonzero pixel term')
        self.reliability_weights = tuple(weight / total for weight in self.reliability_weights)

        gaussian_1d = torch.tensor([1., 4., 6., 4., 1.]) / 16.
        gaussian = gaussian_1d[:, None] * gaussian_1d[None, :]
        self.register_buffer('gaussian', gaussian.view(1, 1, 5, 5))
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3) / 8.,
        )
        self.register_buffer('rgb_to_y', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def _to_lr(self, error, lr_size):
        return F.adaptive_avg_pool2d(error.unsqueeze(1), lr_size).squeeze(1)

    def _gaussian_blur(self, image):
        channels = image.shape[1]
        kernel = self.gaussian.to(dtype=image.dtype).expand(channels, 1, 5, 5)
        return F.conv2d(image, kernel, padding=2, groups=channels)

    def _gradient_direction_error(self, image, target):
        weights = self.rgb_to_y.to(dtype=image.dtype)
        image_y = (image * weights).sum(dim=1, keepdim=True)
        target_y = (target * weights).sum(dim=1, keepdim=True)
        sx = self.sobel_x.to(dtype=image.dtype)
        sy = sx.transpose(-1, -2)
        image_grad = torch.cat((F.conv2d(image_y, sx, padding=1), F.conv2d(image_y, sy, padding=1)), 1)
        target_grad = torch.cat((F.conv2d(target_y, sx, padding=1), F.conv2d(target_y, sy, padding=1)), 1)
        dot = (image_grad * target_grad).sum(dim=1)
        denom = image_grad.square().sum(dim=1).sqrt() * target_grad.square().sum(dim=1).sqrt()
        return 1.0 - dot / denom.clamp_min(1e-3)

    def reliability(self, student_sr, teacher_sr, hr, lr_size):
        student_sr = student_sr.float()
        teacher_sr = teacher_sr.float()
        hr = hr.float()
        pixel_s = (student_sr - hr).abs().mean(dim=1)
        pixel_t = (teacher_sr - hr).abs().mean(dim=1)
        low_hr = self._gaussian_blur(hr)
        low_s = (self._gaussian_blur(student_sr) - low_hr).abs().mean(dim=1)
        low_t = (self._gaussian_blur(teacher_sr) - low_hr).abs().mean(dim=1)
        grad_s = self._gradient_direction_error(student_sr, hr)
        grad_t = self._gradient_direction_error(teacher_sr, hr)

        components_s = (pixel_s, low_s, grad_s)
        components_t = (pixel_t, low_t, grad_t)
        error_s = 0.0
        error_t = 0.0
        for weight, item_s, item_t in zip(self.reliability_weights, components_s, components_t):
            pair_scale = torch.cat((item_s, item_t), dim=0).mean(
                dim=(-2, -1), keepdim=True
            ).detach().clamp_min(1e-6)
            pair_scale_s, pair_scale_t = pair_scale.chunk(2, dim=0)
            # Use the same batch-wise statistic for the student/teacher pair.
            scale = 0.5 * (pair_scale_s + pair_scale_t)
            error_s = error_s + weight * item_s / scale
            error_t = error_t + weight * item_t / scale
        error_s = self._to_lr(error_s, lr_size)
        error_t = self._to_lr(error_t, lr_size)
        gate = torch.sigmoid((error_s - error_t - self.margin) / self.temperature)
        if self.smooth > 1:
            gate = F.avg_pool2d(
                gate.unsqueeze(1), self.smooth, stride=1,
                padding=self.smooth // 2,
            ).squeeze(1)
        return gate.detach(), error_s.detach(), error_t.detach()

    @staticmethod
    def _nodes(feature):
        return F.normalize(feature.float(), dim=1, eps=1e-6)

    def _local_relation(self, feature, gate, window):
        feature = self._nodes(feature)
        batch, channels, height, width = feature.shape
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        if pad_h or pad_w:
            feature = F.pad(feature, (0, pad_w, 0, pad_h), mode='replicate')
            gate = F.pad(gate.unsqueeze(1), (0, pad_w, 0, pad_h), mode='replicate').squeeze(1)
        patches = F.unfold(feature, kernel_size=window, stride=window)
        patches = patches.view(batch, channels, window * window, -1).permute(0, 3, 2, 1)
        patches = F.normalize(patches, dim=-1, eps=1e-6)
        relation = torch.matmul(patches, patches.transpose(-1, -2))
        gate_nodes = F.unfold(
            gate.unsqueeze(1), kernel_size=window, stride=window
        ).transpose(1, 2)
        pair_weight = torch.sqrt(
            gate_nodes.unsqueeze(-1) * gate_nodes.unsqueeze(-2)
        )
        return relation, pair_weight

    def _global_relation(self, feature, gate):
        feature = F.adaptive_avg_pool2d(feature.float(), self.global_grid)
        nodes = F.normalize(feature.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        relation = torch.matmul(nodes, nodes.transpose(-1, -2))
        gate_nodes = F.adaptive_avg_pool2d(
            gate.unsqueeze(1), self.global_grid
        ).flatten(2).transpose(1, 2)
        pair_weight = torch.sqrt(gate_nodes * gate_nodes.transpose(-1, -2))
        return relation, pair_weight

    @staticmethod
    def _weighted_l1(left, right, weight):
        numerator = ((left - right).abs() * weight).sum()
        return numerator / weight.sum().clamp_min(1.0)

    def _relation_terms(self, student_features, teacher_features, gate):
        stage_keys = ('stage4', 'stage8')
        rel_losses = []
        evo_losses = []
        for window in self.local_windows:
            student_rel = []
            teacher_rel = []
            pair_weights = []
            for key in stage_keys:
                sr, weight = self._local_relation(student_features[key], gate, window)
                tr, _ = self._local_relation(teacher_features[key], gate, window)
                student_rel.append(sr)
                teacher_rel.append(tr)
                pair_weights.append(weight)
                rel_losses.append(self._weighted_l1(sr, tr, weight))
            evo_weight = torch.minimum(pair_weights[0], pair_weights[1])
            evo_losses.append(self._weighted_l1(
                student_rel[1] - student_rel[0],
                teacher_rel[1] - teacher_rel[0],
                evo_weight,
            ))

        student_rel = []
        teacher_rel = []
        pair_weights = []
        for key in stage_keys:
            sr, weight = self._global_relation(student_features[key], gate)
            tr, _ = self._global_relation(teacher_features[key], gate)
            student_rel.append(sr)
            teacher_rel.append(tr)
            pair_weights.append(weight)
            rel_losses.append(self._weighted_l1(sr, tr, weight))
        evo_losses.append(self._weighted_l1(
            student_rel[1] - student_rel[0],
            teacher_rel[1] - teacher_rel[0],
            torch.minimum(pair_weights[0], pair_weights[1]),
        ))
        return torch.stack(rel_losses).mean(), torch.stack(evo_losses).mean()

    def forward(self, student_sr, student_features, teacher_sr, teacher_features, hr):
        if self.mode == 'output':
            raw_output = F.l1_loss(student_sr, teacher_sr.detach())
            total = self.lambda_output * raw_output
            return total, {
                'output': raw_output.detach(), 'rel': total.new_zeros(()),
                'evo': total.new_zeros(()), 'gate_mean': total.new_ones(()),
                'teacher_better': total.new_zeros(()),
            }
        if not student_features or not teacher_features:
            raise ValueError('relation/full RGCRD requires stage4 and stage8 features')
        lr_size = student_features['stage4'].shape[-2:]
        if self.mode == 'full':
            gate, error_s, error_t = self.reliability(
                student_sr, teacher_sr, hr, lr_size
            )
        else:
            gate = student_sr.new_ones((student_sr.shape[0], *lr_size))
            error_s = error_t = gate
        raw_rel, raw_evo = self._relation_terms(
            student_features, teacher_features, gate
        )
        total = self.lambda_rel * raw_rel + self.lambda_evo * raw_evo
        return total, {
            'output': total.new_zeros(()), 'rel': raw_rel.detach(),
            'evo': raw_evo.detach(), 'gate_mean': gate.mean().detach(),
            'teacher_better': (error_t < error_s).float().mean().detach(),
        }
