"""V1/N21 adaptation: current-feature parent refresh, global-response residual.

Retains the original local SDPA, routing, EMA and complete SRPR forward.
This is NOT the previous N21 implementation or a guarantee of PSNR gains.
"""
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .lfmn import TAB, center_iter, ema_inplace
from .lfmnsrprv2 import Net as V1Net


class Bridge:
    def __init__(self):
        self.centers = None


def current_parents(tokens, centers):
    # New membership in current post-writeback feature space, not old labels.
    with torch.no_grad():
        labels = torch.bmm(F.normalize(tokens, dim=-1), F.normalize(centers, dim=-1).transpose(1, 2)).argmax(-1)
    assignment = F.one_hot(labels, centers.shape[1]).to(tokens.dtype)
    mass = assignment.sum(1)
    means = torch.bmm(assignment.transpose(1, 2), F.normalize(tokens, dim=-1)) / mass.clamp_min(1).unsqueeze(-1)
    return torch.where((mass > 0).unsqueeze(-1), F.normalize(means, dim=-1), centers)


def global_response(module, tokens, indices, keys, values, batched=False):
    b, n, _ = tokens.shape
    q = torch.gather(module.to_q(tokens), 1, indices.expand(b, n, module.to_q.out_features))
    gs = min(n, module.group_size)
    ng, pad = (n+gs-1)//gs, ((n+gs-1)//gs)*gs-n
    q = torch.cat((q, torch.flip(q[:, n-pad:n], [1])), 1)
    q = rearrange(q, 'b (g s) (h d) -> b g h s d', g=ng, h=module.heads)
    if not batched:
        kg = keys[None, None].expand(b, ng, -1, -1, -1)
        vg = values[None, None].expand(b, ng, -1, -1, -1)
    else:
        kg = rearrange(keys, 'b k (h d) -> b h k d', h=module.heads)[:, None].expand(-1, ng, -1, -1, -1)
        vg = rearrange(values, 'b k (h d) -> b h k d', h=module.heads)[:, None].expand(-1, ng, -1, -1, -1)
    result = rearrange(F.scaled_dot_product_attention(q, kg, vg), 'b g h s d -> b (g s) (h d)')[:, :n]
    result = result.scatter(1, indices.expand_as(result), result)
    return module.proj(result)


class RefreshTAB(TAB):
    def __init__(self, source, bridge, inherit):
        nn.Module.__init__(self)
        for name in ('norm', 'mlp', 'irca_attn', 'iasa_attn', 'conv1x1'):
            setattr(self, name, getattr(source, name))
        for name in ('n_iter', 'ema_decay', 'num_tokens', 'eval_refine_iters'):
            setattr(self, name, getattr(source, name))
        self.register_buffer('means', source.means.clone())
        self.register_buffer('initted', source.initted.clone())
        self.bridge, self.inherit, self.enabled = bridge, inherit, True
        if inherit:
            self.parent_map = nn.Sequential(nn.Linear(48, 8), nn.GELU(), nn.Linear(8, 48))
            self.child_code = nn.Parameter(torch.randn(2, 48)*.02)
            self.inheritance_gain = nn.Parameter(torch.zeros(()))

    def forward(self, image):
        if not self.enabled:
            return super().forward(image)
        assert self.eval_refine_iters == 0, 'This adaptation preserves V1 default EMA evaluation only'
        b, _, h, w = image.shape
        residual = rearrange(image, 'b c h w -> b (h w) c')
        x = self.norm(residual)
        n = x.shape[1]
        if not self.initted:
            pad = self.num_tokens-n%self.num_tokens
            padded = torch.cat((x, torch.flip(x[:, n-pad:n], [1])), 1)
            anchors = rearrange(padded, 'b (k n) d -> k (b n) d', k=self.num_tokens).mean(1).detach()
        else:
            anchors = self.means.detach()
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter-1):
                    anchors = center_iter(F.normalize(x, dim=-1), F.normalize(anchors, dim=-1))
        keys, values, anchors = self.irca_attn(x, anchors)
        with torch.no_grad():
            labels = torch.einsum('b i c,j c->b i j', F.normalize(x, dim=-1), F.normalize(anchors, dim=-1)).argmax(-1)
            indices = labels.argsort(-1).unsqueeze(-1)
        # Original local AND global response is computed unchanged.
        y = self.iasa_attn(x, indices, keys, values)
        centers = anchors[None].expand(b, -1, -1)
        if self.inherit:
            parent = self.bridge.centers
            assert parent is not None and parent.shape[:2] == (b, self.num_tokens//2)
            parents = current_parents(x, parent)
            offsets = self.parent_map(parents)[:, :, None]+self.child_code[None, None]
            children = (parents[:, :, None]+.1*torch.tanh(offsets)).reshape(b, self.num_tokens, 48)
            new_response = global_response(self.iasa_attn, x, indices,
                self.irca_attn.to_k(children), self.irca_attn.to_v(children), batched=True)
            old_response = global_response(self.iasa_attn, x, indices, keys, values)
            gain = .25*torch.tanh(self.inheritance_gain)
            y = y + gain*(new_response-old_response)
            centers = centers + gain*(children-centers)
        self.bridge.centers = centers.detach()
        y = self.conv1x1(rearrange(y, 'b (h w) c -> b c h w', h=h).contiguous())
        x = residual+rearrange(y, 'b c h w -> b (h w) c')
        x = x+self.mlp(x, x_size=(h, w))
        if self.training:
            with torch.no_grad():
                if not self.initted:
                    self.means.copy_(anchors)
                    self.initted.fill_(True)
                else:
                    ema_inplace(self.means, anchors, self.ema_decay)
        return rearrange(x, 'b (h w) c -> b c h w', h=h)


class Net(V1Net):
    def __init__(self, scale=4, **kwargs):
        super().__init__(scale=scale, **kwargs)
        self.bridges = [Bridge(), Bridge()]
        with torch.random.fork_rng(devices=[]):
            for i, block in enumerate(self.blocks):
                block[0] = RefreshTAB(block[0], self.bridges[i//4], i%4 != 0)

    def set_enabled(self, enabled):
        for block in self.blocks:
            block[0].enabled = bool(enabled)

    def forward(self, x):
        for bridge in self.bridges:
            bridge.centers = None
        try:
            return super().forward(x)
        finally:
            for bridge in self.bridges:
                bridge.centers = None


def make_model(args):
    return Net(scale=args.scale[0])
