"""CAMD-inspired deterministic priors inside SFML, without a new backbone."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .lfmn import Net as BaselineNet, SFML


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
            gap = torch.sqrt(((xx-yy).square() + 4*xy.square()).clamp_min(0))
            uncertainty = (1-gap/(xx+yy+1e-6)).clamp(0, 1)
            average = sum(maps) / 3
            return torch.cat((*maps, average * uncertainty), 1)


class PriorSFML(SFML):
    def __init__(self, source):
        nn.Module.__init__(self)
        for name in ('reduce', 'dw', 'expand_a', 'expand_b'):
            setattr(self, name, getattr(source, name))
        self.prior_projection = nn.Conv2d(4, self.reduce.out_channels, 1)
        self.prior_gain = nn.Parameter(torch.zeros(()))
        self.enabled = True

    def forward(self, fs, priors=None):
        if not self.enabled:
            return super().forward(fs)
        if priors is None or priors.shape[-2:] != fs.shape[-2:]:
            raise ValueError('Missing or misaligned CAMD priors')
        u = F.leaky_relu(self.dw(F.leaky_relu(self.reduce(fs), 0.1)), 0.1)
        factor = 1 + 0.25 * torch.tanh(self.prior_gain) * torch.tanh(self.prior_projection(priors.to(u.dtype)))
        u = u * factor
        return torch.sigmoid(self.expand_a(u)), self.expand_b(u)


class Net(BaselineNet):
    def __init__(self, scale=4, **kwargs):
        super().__init__(scale=scale, **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.sfmls = nn.ModuleList([PriorSFML(module) for module in self.sfmls])
        self.detail_priors = DetailPriors()
        self.enabled = True

    def set_enabled(self, value):
        self.enabled = bool(value)
        for module in self.sfmls:
            module.enabled = self.enabled

    def forward(self, x):
        if not self.enabled:
            return super().forward(x)
        priors = self.detail_priors(x)
        x0, fs = self.first_conv(x), self.fea(x)
        feat = x0
        for i, (g_attn, l_attn) in enumerate(self.blocks):
            prev = feat
            beta, gamma = self.sfmls[i](fs, priors)
            s = l_attn(g_attn(beta * prev + gamma), self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        return self.last_conv(u) + F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


def make_model(args):
    if getattr(args, 'rgb_range', 255) != 255:
        raise ValueError('N22 requires rgb_range=255')
    return Net(scale=args.scale[0])
