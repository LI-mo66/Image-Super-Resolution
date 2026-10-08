"""V1/N22: static LR priors plus existing SRPR observation context in SFML.

This adaptation preserves the V1 forward and does not add HR decoding passes.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .lfmn import SFML
from .lfmnsrprv2 import Net as V1Net


def safe_pad(x, right, bottom):
    return F.pad(x, (0, right, 0, bottom), mode='replicate')


class DetailPriors(nn.Module):
    def __init__(self):
        super().__init__()
        for size in (16, 8, 4):
            positions = sorted(((u, v) for u in range(size) for v in range(size)),
                               key=lambda uv: (sum(uv), -uv[0] if sum(uv) % 2 == 0 else uv[0]))[:8]
            coords = torch.arange(size, dtype=torch.float32) + 0.5
            filters = []
            for u, v in positions:
                au = math.sqrt((1 if u == 0 else 2) / size)
                av = math.sqrt((1 if v == 0 else 2) / size)
                filters.append(au * av * torch.cos(math.pi * u * coords[:, None] / size)
                               * torch.cos(math.pi * v * coords[None, :] / size))
            self.register_buffer(f'dct_{size}', torch.stack(filters)[:, None], persistent=False)
        sobel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('sobel', torch.stack((sobel, sobel.T))[:, None], persistent=False)
        coords = torch.arange(-2, 3, dtype=torch.float32)
        g = torch.exp(-coords.square() / (2 * 1.5**2))
        gaussian = g[:, None] * g[None, :]
        self.register_buffer('gaussian', (gaussian / gaussian.sum())[None, None], persistent=False)

    def forward(self, image):
        # Fixed rgb_range=255, no image-max normalization or HR information.
        with torch.autocast(device_type=image.device.type, enabled=False):
            image = image.float() / 255.0
            gray = (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
            h, w = gray.shape[-2:]
            maps = []
            for size in (16, 8, 4):
                padded = safe_pad(gray, (-w) % size, (-h) % size)
                coeff = F.conv2d(padded, getattr(self, f'dct_{size}').float(), stride=size)
                energy = coeff.square()
                response = (energy / (energy + 0.01)).mean(1, keepdim=True)
                maps.append(F.interpolate(response, scale_factor=size, mode='nearest')[:, :, :h, :w])
            gradients = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), self.sobel.float())
            gx, gy = gradients[:, :1], gradients[:, 1:]
            tensor = torch.cat((gx.square(), gy.square(), gx * gy), 1)
            tensor = F.conv2d(F.pad(tensor, (2, 2, 2, 2), mode='replicate'),
                              self.gaussian.float().expand(3, 1, 5, 5), groups=3)
            xx, yy, xy = tensor[:, :1], tensor[:, 1:2], tensor[:, 2:]
            gap = torch.sqrt(((xx-yy).square() + 4*xy.square()).clamp_min(1e-12))
            uncertainty = (1-gap/(xx+yy+1e-6)).clamp(0, 1)
            average = sum(maps) / 3
            return torch.cat((*maps, average * uncertainty), 1)



class PriorSFML(SFML):
    def __init__(self, source, context):
        nn.Module.__init__(self)
        for name in ('reduce', 'dw', 'expand_a', 'expand_b'):
            setattr(self, name, getattr(source, name))
        self.prior_projection = nn.Conv2d(6, self.reduce.out_channels, 1)
        self.prior_gain = nn.Parameter(torch.zeros(()))
        self.context, self.enabled = context, True

    def forward(self, fs):
        if not self.enabled:
            return super().forward(fs)
        priors = self.context['priors']
        observation = self.context['observation']
        assert priors is not None and priors.shape[-2:] == fs.shape[-2:]
        if observation is None:
            drive = torch.zeros_like(priors[:, :1])
            relative_drive = drive
        else:
            assert observation.shape == (fs.shape[0], 3, fs.shape[2], fs.shape[3])
            # Most recent pre-writeback adjoint observation, NOT HR error or
            # post-correction residual. Local pooled energy is a context cue.
            energy = F.avg_pool2d(observation.float().square().mean(1, keepdim=True),
                                 3, stride=1, padding=1, count_include_pad=False)
            magnitude = (torch.sqrt(energy + 1e-12)-1e-6).clamp_min(0) / 255.
            drive = magnitude / (1 + magnitude)
            # Physical amplitude (~1e-3) is much smaller than static priors
            # (~1e-1). Keep absolute magnitude AND spatial-relative context;
            # the relative cue is not reconstruction reliability.
            reference = magnitude.detach().mean((-2,-1),keepdim=True).clamp_min(1e-6)
            relative_drive = magnitude / (magnitude + reference)
        conditioned = torch.cat((priors, drive, relative_drive), 1).to(fs.dtype)
        u = F.leaky_relu(self.dw(F.leaky_relu(self.reduce(fs), .1)), .1)
        correction = torch.tanh(self.prior_projection(conditioned))
        u = u * (1 + .25*torch.tanh(self.prior_gain)*correction)
        return torch.sigmoid(self.expand_a(u)), self.expand_b(u)


class Net(V1Net):
    def __init__(self, scale=4, **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.context = {'priors': None, 'observation': None}
        self.enabled = True
        with torch.random.fork_rng(devices=[]):
            self.sfmls = nn.ModuleList([PriorSFML(module, self.context) for module in self.sfmls])
        self.detail_priors = DetailPriors()
        for module in self.proximal:
            module.register_forward_pre_hook(self._capture_observation)

    def _capture_observation(self, module, inputs):
        if self.enabled:
            self.context['observation'] = inputs[2]

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        for module in self.sfmls:
            module.enabled = self.enabled

    def forward(self, x):
        self.context['observation'] = None
        self.context['priors'] = self.detail_priors(x) if self.enabled else None
        try:
            return super().forward(x)
        finally:
            self.context['priors'] = self.context['observation'] = None


def make_model(args):
    if getattr(args, 'rgb_range', 255) != 255:
        raise ValueError('V1N22 requires rgb_range=255')
    return Net(scale=args.scale[0])
