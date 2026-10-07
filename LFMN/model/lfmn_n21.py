"""CCE-inspired, fixed-budget center inheritance. Not a full CCE reproduction."""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .lfmn import Net as BaselineNet, TAB, center_iter, ema_inplace


def grouped_current(x, labels, count, fallback):
    """Per-image aggregation; empty parents retain an anchor, not NaNs."""
    assignment = F.one_hot(labels, count).to(x.dtype)
    mass = assignment.sum(1)
    values = torch.bmm(assignment.transpose(1, 2), F.normalize(x, dim=-1))
    values = F.normalize(values / mass.clamp_min(1).unsqueeze(-1), dim=-1)
    return torch.where((mass > 0).unsqueeze(-1), values, fallback)


def batched_iasa(module, x, indices, keys, values):
    """Same two SDPA pathways as baseline; global KV now has a batch axis."""
    b, n, _ = x.shape
    q, k, v = module.to_q(x), module.to_k(x), module.to_v(x)
    q = torch.gather(q, 1, indices.expand(q.shape))
    k = torch.gather(k, 1, indices.expand(k.shape))
    v = torch.gather(v, 1, indices.expand(v.shape))
    gs = min(n, module.group_size)
    ng = (n + gs - 1) // gs
    pad = ng * gs - n
    q = torch.cat((q, torch.flip(q[:, n-pad:n], [1])), 1)
    q = rearrange(q, 'b (ng gs) (h d) -> b ng h gs d', ng=ng, h=module.heads)
    k = torch.cat((k, torch.flip(k[:, n-pad-gs:n], [1])), 1).unfold(1, 2*gs, gs)
    v = torch.cat((v, torch.flip(v[:, n-pad-gs:n], [1])), 1).unfold(1, 2*gs, gs)
    k = rearrange(k, 'b ng (h d) gs -> b ng h gs d', h=module.heads)
    v = rearrange(v, 'b ng (h d) gs -> b ng h gs d', h=module.heads)
    kg = rearrange(keys, 'b k (h d) -> b h k d', h=module.heads)[:, None].expand(-1, ng, -1, -1, -1)
    vg = rearrange(values, 'b k (h d) -> b h k d', h=module.heads)[:, None].expand(-1, ng, -1, -1, -1)
    y = F.scaled_dot_product_attention(q, k, v)
    y = y + F.scaled_dot_product_attention(q, kg, vg)
    y = rearrange(y, 'b ng h gs d -> b (ng gs) (h d)')[:, :n]
    y = y.scatter(1, indices.expand(y.shape), y)
    return module.proj(y)


class InheritanceTAB(TAB):
    def __init__(self, source, inherit):
        nn.Module.__init__(self)
        for name in ('norm', 'mlp', 'irca_attn', 'iasa_attn', 'conv1x1'):
            setattr(self, name, getattr(source, name))
        for name in ('n_iter', 'ema_decay', 'num_tokens', 'eval_refine_iters'):
            setattr(self, name, getattr(source, name))
        self.register_buffer('means', source.means.clone())
        self.register_buffer('initted', source.initted.clone())
        self.inherit = inherit
        self.enabled = True
        if inherit:
            dim = self.means.shape[-1]
            self.parent_map = nn.Sequential(nn.Linear(dim, 8), nn.GELU(), nn.Linear(8, dim))
            self.child_code = nn.Parameter(torch.randn(2, dim) * 0.02)
            self.inheritance_gain = nn.Parameter(torch.zeros(()))

    def forward(self, image, state=None, return_state=False):
        if not self.enabled:
            return super().forward(image)
        b, _, h, w = image.shape
        x = rearrange(image, 'b c h w -> b (h w) c')
        residual = x
        x = self.norm(x)
        n = x.shape[1]
        if not self.initted:
            pad = self.num_tokens - n % self.num_tokens
            padded = torch.cat((x, torch.flip(x[:, n-pad:n], [1])), 1)
            anchors = rearrange(padded, 'b (k n) d -> k (b n) d', k=self.num_tokens).mean(1).detach()
        else:
            anchors = self.means.detach()
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    anchors = center_iter(F.normalize(x, dim=-1), F.normalize(anchors, dim=-1))
        # Retain exactly the original train-time anchor updates.
        anchor_keys, anchor_values, anchors = self.irca_attn(x, anchors)
        centers = anchors.unsqueeze(0).expand(b, -1, -1)
        delta = torch.zeros_like(centers)
        if self.inherit:
            if state is None:
                raise ValueError('Missing parent assignment')
            labels, parent_count = state
            if parent_count * 2 != self.num_tokens or labels.shape != (b, n):
                raise ValueError('Parent/child geometry mismatch')
            fallback = F.normalize(centers[:, ::2], dim=-1)
            parents = grouped_current(x, labels, parent_count, fallback)
            offsets = self.parent_map(parents)[:, :, None] + self.child_code[None, None]
            children = parents[:, :, None] + 0.1 * torch.tanh(offsets)
            children = children.reshape(b, self.num_tokens, -1)
            delta = 0.25 * torch.tanh(self.inheritance_gain) * (children - centers)
            centers = centers + delta
        # Routing is discrete; values remain differentiable through global KV.
        with torch.no_grad():
            # Anchor-plus-correction preserves baseline rounding at zero gain
            # without a per-stage GPU-to-host scalar synchronization.
            nx = F.normalize(x, dim=-1)
            scores = torch.einsum('b i c,j c->b i j', nx, F.normalize(anchors, dim=-1))
            if self.inherit:
                shift = F.normalize(centers, dim=-1) - F.normalize(anchors[None].expand_as(centers), dim=-1)
                scores = scores + torch.bmm(nx, shift.transpose(1, 2))
            labels = scores.argmax(-1)
            indices = labels.argsort(-1).unsqueeze(-1)
        # Linear, bias-free projections preserve baseline arithmetic at zero gain
        # while retaining the center delta's learning gradient.
        keys = rearrange(anchor_keys, 'h k d -> k (h d)')[None] + self.irca_attn.to_k(delta)
        values = rearrange(anchor_values, 'h k d -> k (h d)')[None] + self.irca_attn.to_v(delta)
        y = batched_iasa(self.iasa_attn, x, indices, keys, values)
        y = self.conv1x1(rearrange(y, 'b (h w) c -> b c h w', h=h).contiguous())
        x = residual + rearrange(y, 'b c h w -> b (h w) c')
        x = x + self.mlp(x, x_size=(h, w))
        if self.training:
            with torch.no_grad():
                if not self.initted:
                    self.means.copy_(anchors)
                    self.initted.fill_(True)
                else:
                    ema_inplace(self.means, anchors, self.ema_decay)
        result = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return (result, (labels.detach(), self.num_tokens)) if return_state else result


class Net(BaselineNet):
    def __init__(self, scale=4, **kwargs):
        super().__init__(scale=scale, **kwargs)
        # Candidate initialization must not alter the downstream data RNG stream.
        with torch.random.fork_rng(devices=[]):
            for i, block in enumerate(self.blocks):
                block[0] = InheritanceTAB(block[0], inherit=(i % 4 != 0))
        self.enabled = True

    def set_enabled(self, value):
        self.enabled = bool(value)
        for block in self.blocks:
            block[0].enabled = self.enabled

    def forward(self, x):
        if not self.enabled:
            return super().forward(x)
        x0, fs = self.first_conv(x), self.fea(x)
        feat, state = x0, None
        for i, (g_attn, l_attn) in enumerate(self.blocks):
            if i % 4 == 0:
                state = None
            prev = feat
            beta, gamma = self.sfmls[i](fs)
            t, state = g_attn(beta * prev + gamma, state, return_state=True)
            s = l_attn(t, self.patch_size[i])
            feat = self.esas[i](prev + self.mid_convs[i](s))
        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        return self.last_conv(u) + F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


def make_model(args):
    return Net(scale=args.scale[0])
