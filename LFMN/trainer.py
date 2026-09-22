import os
import math
import json
from decimal import Decimal


import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

from rgcrd import RGCRDCriterion, SwinIRTeacher

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        self.rgcrd_mode = getattr(args, 'rgcrd_mode', 'off')
        self.rgcrd = None
        self.rgcrd_teacher = None
        self.rgcrd_log_columns = (
            'epoch', 'aux', 'output', 'rel', 'evo', 'gate_mean',
            'teacher_better', 'base_grad_norm', 'aux_grad_norm',
            'grad_ratio', 'grad_cosine',
        )
        self.rgcrd_log = []
        if self.rgcrd_mode != 'off':
            if args.n_GPUs != 1:
                raise ValueError('RGCRD currently requires --n_GPUs 1')
            if args.precision != 'single':
                raise ValueError('RGCRD student training requires --precision single')
            if self.rgcrd_mode in ('relation', 'full') and args.model.lower() != 'lfmnrgcrd':
                raise ValueError(
                    'relation/full RGCRD requires --model LFMNRGCRD'
                )
            device = torch.device('cpu' if args.cpu else 'cuda')
            # The DataLoader sampler and worker seeds are materialized when
            # iteration begins, after Trainer construction.  Building SwinIR
            # initializes many random parameters before loading its checkpoint;
            # preserve RNG state so B0/M0/M1 see the same shuffle/crop stream.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all()
                if device.type == 'cuda' else None
            )
            try:
                self.rgcrd = RGCRDCriterion(args).to(device)
                self.rgcrd_teacher = SwinIRTeacher(args, device)
            finally:
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)
            self.rgcrd_teacher_microbatch = max(
                0, int(args.rgcrd_teacher_microbatch)
            )
            self.ckp.write_log(
                'RGCRD enabled: mode={} (training only; teacher frozen)'.format(
                    self.rgcrd_mode
                )
            )
            rg_log_path = self.ckp.get_path('rgcrd_log.pt')
            if args.load and os.path.isfile(rg_log_path):
                payload = torch.load(rg_log_path, map_location='cpu')
                if tuple(payload.get('columns', ())) != self.rgcrd_log_columns:
                    raise ValueError('incompatible rgcrd_log.pt columns')
                self.rgcrd_log = payload.get('rows', [])

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=ckp.resume_epoch)

        self.error_last = 1e8
        self.n12_diagnostics = []
        self.n12_diagnostic_columns = (
            'epoch', 'phase', 'batch', 'stage', 'residual_l2',
            'backprojection_l2', 'observation_l2', 'state_l2',
            'state_delta_l2', 'feature_delta_l2', 'q_delta_l2',
            'gate_mean', 'gate_std', 'gate_saturation',
            'data_consistency_ratio',
        )
        if args.load:
            diag_path = self.ckp.get_path('mechanism_diagnostics.pt')
            if os.path.isfile(diag_path):
                payload = torch.load(diag_path, map_location='cpu')
                if tuple(payload.get('columns', ())) != self.n12_diagnostic_columns:
                    raise ValueError('incompatible mechanism_diagnostics.pt columns')
                self.n12_diagnostics = payload.get('rows', [])

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        rg_sums = {
            key: 0.0 for key in
            ('aux', 'output', 'rel', 'evo', 'gate_mean', 'teacher_better')
        }
        grad_diag = None

        timer_data, timer_model = utility.timer(), utility.timer()
        processed_batches = 0
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            model_output = self.model(lr, idx_scale)
            self._capture_n12_diagnostics(epoch, 'train', batch)
            if self.rgcrd_mode == 'off':
                loss = self.loss(model_output, hr)
            else:
                if isinstance(model_output, (tuple, list)):
                    sr, student_features = model_output
                else:
                    sr, student_features = model_output, None
                base_loss = self.loss(sr, hr)
                teacher_sr, teacher_features = self._teacher_forward(lr)
                auxiliary_loss, rg_stats = self.rgcrd(
                    sr, student_features, teacher_sr, teacher_features, hr
                )
                diag_every = max(0, int(self.args.rgcrd_grad_diag_every))
                if batch == 0 and diag_every and epoch % diag_every == 0:
                    grad_diag = self._gradient_diagnostic(
                        base_loss, auxiliary_loss
                    )
                loss = base_loss + auxiliary_loss
                rg_sums['aux'] += float(auxiliary_loss.detach())
                for key, value in rg_stats.items():
                    rg_sums[key] += float(value)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            processed_batches = batch + 1

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            if (
                self.args.max_train_batches > 0
                and processed_batches >= self.args.max_train_batches
            ):
                self.ckp.write_log(
                    'Stopped epoch early after {} batches (--max_train_batches).'.format(
                        processed_batches
                    )
                )
                break

        self.loss.end_log(processed_batches)
        if self.rgcrd_mode != 'off' and processed_batches:
            means = {
                key: value / processed_batches for key, value in rg_sums.items()
            }
            self.ckp.write_log(
                'RGCRD epoch {}: aux={aux:.6f}, output={output:.6f}, '
                'rel={rel:.6f}, evo={evo:.6f}, gate={gate_mean:.4f}, '
                'teacher_better={teacher_better:.4f}'.format(epoch, **means)
            )
            if grad_diag is not None:
                self.ckp.write_log(
                    'RGCRD gradient diagnostic: base_norm={base_norm:.3e}, '
                    'aux_norm={aux_norm:.3e}, ratio={ratio:.4f}, '
                    'cosine={cosine:.4f}'.format(**grad_diag)
                )
            diag_values = grad_diag or {
                'base_norm': float('nan'), 'aux_norm': float('nan'),
                'ratio': float('nan'), 'cosine': float('nan'),
            }
            self.rgcrd_log.append([
                float(epoch), means['aux'], means['output'], means['rel'],
                means['evo'], means['gate_mean'], means['teacher_better'],
                diag_values['base_norm'], diag_values['aux_norm'],
                diag_values['ratio'], diag_values['cosine'],
            ])
            torch.save(
                {
                    'columns': self.rgcrd_log_columns,
                    'rows': self.rgcrd_log,
                },
                self.ckp.get_path('rgcrd_log.pt'),
            )
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def _gradient_diagnostic(self, base_loss, auxiliary_loss):
        parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        base_grads = torch.autograd.grad(
            base_loss, parameters, retain_graph=True, allow_unused=True
        )
        aux_grads = torch.autograd.grad(
            auxiliary_loss, parameters, retain_graph=True, allow_unused=True
        )
        dot = base_loss.new_zeros(())
        base_sq = base_loss.new_zeros(())
        aux_sq = base_loss.new_zeros(())
        for base_grad, aux_grad in zip(base_grads, aux_grads):
            if base_grad is not None:
                base_sq = base_sq + base_grad.detach().float().square().sum()
            if aux_grad is not None:
                aux_sq = aux_sq + aux_grad.detach().float().square().sum()
            if base_grad is not None and aux_grad is not None:
                dot = dot + (
                    base_grad.detach().float() * aux_grad.detach().float()
                ).sum()
        base_norm = base_sq.sqrt()
        aux_norm = aux_sq.sqrt()
        denominator = (base_norm * aux_norm).clamp_min(1e-12)
        return {
            'base_norm': float(base_norm),
            'aux_norm': float(aux_norm),
            'ratio': float(aux_norm / base_norm.clamp_min(1e-12)),
            'cosine': float(dot / denominator),
        }

    def _teacher_forward(self, lr):
        microbatch = self.rgcrd_teacher_microbatch
        if microbatch <= 0 or lr.shape[0] <= microbatch:
            return self.rgcrd_teacher(lr)
        outputs = []
        features = {'stage4': [], 'stage8': []}
        for chunk in lr.split(microbatch, dim=0):
            output, chunk_features = self.rgcrd_teacher(chunk)
            outputs.append(output)
            for key in features:
                features[key].append(chunk_features[key])
        return torch.cat(outputs, dim=0), {
            key: torch.cat(value, dim=0) for key, value in features.items()
        }

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.ckp.add_ssim_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        per_image_metrics = (
            [] if getattr(self.args, 'save_per_image_metrics', False) else None
        )
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                #for lr, hr, filename, _ in tqdm(d, ncols=80):
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    self._capture_n12_diagnostics(epoch, 'eval', idx_data)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    image_psnr = utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    image_ssim = utility.calc_ssim(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    self.ckp.log[-1, idx_data, idx_scale] += image_psnr
                    self.ckp.log_ssim[-1, idx_data, idx_scale] += image_ssim
                    if per_image_metrics is not None:
                        per_image_metrics.append({
                            'dataset': d.dataset.name,
                            'scale': int(scale),
                            'filename': filename[0],
                            'psnr': float(image_psnr),
                            'ssim': float(image_ssim),
                        })
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.log_ssim[-1, idx_data, idx_scale] /= len(d)
                best_psnr = self.ckp.log.max(0)
                best_ssim = self.ckp.log_ssim.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})\tSSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best_psnr[0][idx_data, idx_scale],
                        best_psnr[1][idx_data, idx_scale] + 1,
                        self.ckp.log_ssim[-1, idx_data, idx_scale],
                        best_ssim[0][idx_data, idx_scale],
                        best_ssim[1][idx_data, idx_scale] + 1
                    )
                )

        if per_image_metrics is not None:
            metrics_dir = self.ckp.get_path('per_image_metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            torch.save(
                per_image_metrics,
                os.path.join(metrics_dir, 'epoch_{:04d}.pt'.format(epoch)),
            )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch)
            self._save_n12_diagnostics()
        else:
            # Test-only runs still need machine-readable, full-precision metrics.
            # Without these files downstream comparisons can only recover the
            # rounded values printed in log.txt.
            torch.save(self.ckp.log, self.ckp.get_path('psnr_log.pt'))
            torch.save(
                self.ckp.log_ssim,
                self.ckp.get_path('ssim_log.pt'),
            )

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def _capture_n12_diagnostics(self, epoch, phase, batch):
        source = getattr(self.model, 'diagnostics_snapshot', None)
        if source is None:
            source = getattr(getattr(self.model, 'model', None), 'diagnostics_snapshot', None)
        if source is None:
            return
        diagnostics = source()
        if not diagnostics:
            return
        stages = len(next(iter(diagnostics.values())))
        for stage in range(stages):
            row = [float(epoch), phase, int(batch), int(stage)]
            row.extend(float(diagnostics[key][stage]) for key in self.n12_diagnostic_columns[4:])
            self.n12_diagnostics.append(row)

    def _save_n12_diagnostics(self):
        if not self.n12_diagnostics:
            return
        payload = {
            'columns': self.n12_diagnostic_columns,
            'rows': self.n12_diagnostics,
        }
        torch.save(payload, self.ckp.get_path('mechanism_diagnostics.pt'))
        with open(self.ckp.get_path('mechanism_diagnostics.jsonl'), 'w', encoding='utf-8') as handle:
            for row in self.n12_diagnostics:
                handle.write(json.dumps(dict(zip(self.n12_diagnostic_columns, row))) + '\n')

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # MultiStepLR performs its initial step during construction, so
            # last_epoch is 0 before the first training epoch in current PyTorch.
            # Training is complete only after schedule() reaches args.epochs.
            return self.optimizer.get_last_epoch() >= self.args.epochs
