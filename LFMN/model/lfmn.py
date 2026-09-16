"""LFMN reproduction (Windows/Python 3.10), macro per paper Eqs.1-7,
blocks re-implemented faithfully from CATANet (CVPR2025) TAB/LRSA source.
Attribute names/shapes are exact matches to model/scale*_*.pt (strict-load clean).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def default_conv(in_c, out_c, kernel_size, bias=True):
    return nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size // 2, bias=bias)


# ---------------------------------------------------------------- fea_Net
class fea_Net(nn.Module):
    """Shared shallow feature prior FS (3xConv+ReLU -> 32ch)."""

    def __init__(self):
        super(fea_Net, self).__init__()
        self.net = nn.Sequential(
            default_conv(3, 8, 3, bias=False), nn.ReLU(True),
            default_conv(8, 16, 3, bias=False), nn.ReLU(True),
            default_conv(16, 32, 3, bias=False), nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- ESA (RFANet)
class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, 1)
        self.conv_f = conv(f, f, 1)
        self.conv2 = nn.Conv2d(f, f, 3, stride=2, padding=0)
        self.conv3 = conv(f, f, 3)
        self.conv4 = conv(f, n_feats, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# ---------------------------------------------------------------- SFML (paper eq4-6)
class SFML(nn.Module):
    """U = phi(DWC(phi(C1x1(FS)))); beta = sigmoid(C1x1(U)); gamma = C1x1(U)."""

    def __init__(self, side_c=32, mid=12, n_feats=48):
        super(SFML, self).__init__()
        self.reduce = nn.Conv2d(side_c, mid, 1)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.expand_a = nn.Conv2d(mid, n_feats, 1)
        self.expand_b = nn.Conv2d(mid, n_feats, 1)

    def forward(self, fs):
        # eq4: U = phi(DWC(phi(C1x1(FS))))
        u = F.leaky_relu(self.dw(F.leaky_relu(self.reduce(fs), 0.1)), 0.1)
        beta = torch.sigmoid(self.expand_a(u))          # eq5: beta = sigma(C1x1(U))
        gamma = self.expand_b(u)                        # eq5: gamma = C1x1(U)
        return beta, gamma


# ========================================================= CATANet internals
def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def default(x, d):
    return d if not exists(x) else d


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=1 - decay)


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]
    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)
    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0
    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    return means.squeeze(0)


class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))
        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N
        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, 'b (ng gs) (h d) -> b ng h gs d', ng=ng, h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        paded_k = rearrange(paded_k, 'b ng (h d) gs -> b ng h gs d', h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        paded_v = rearrange(paded_v, 'b ng (h d) gs -> b ng h gs d', h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)
        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)
        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
        out = out1 + out2
        out = rearrange(out, 'b ng h gs d -> b (ng gs) (h d)')[:, :N, :]
        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        return self.proj(out)


class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            x_global = x_means
        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)
        return k, v, x_global.detach()


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1, groups=hidden_features),
            nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class TAB(nn.Module):
    """Token Aggregation Block: cluster pixels to num_tokens centroids,
    IASA (intra-token window self-attention) + IRCA (inter-token global),
    EMA-updated centroid buffers (eval keeps loaded means)."""

    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3, num_tokens=8,
                 group_size=128, ema_decay=0.999):
        super().__init__()
        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.irca_attn = IRCA(dim, qk_dim, heads)
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape
        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c -> cnt (b n) c',
                                           cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        k_global, v_global, x_means = self.irca_attn(x, x_means)
        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j',
                                    F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)
            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)
        y = self.iasa_attn(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w -> b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x
        if self.training:
            with torch.no_grad():
                if not self.initted:
                    self.means.data.copy_(x_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, x_means, self.ema_decay)
        return rearrange(x, 'b (h w) c -> b c h w', h=h)


# ------------------------------------------------ LRSA patch utilities
def patch_divide(x, step, ps):
    b, c, h, w = x.size()
    if h == ps and w == ps:
        step = ps
    crop_x = []
    nh = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        nh += 1
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            crop_x.append(x[:, :, top:down, left:right])
    nw = len(crop_x) // nh
    crop_x = torch.stack(crop_x, dim=0)
    crop_x = crop_x.permute(1, 0, 2, 3, 4).contiguous()
    return crop_x, nh, nw


def patch_reverse(crop_x, x, step, ps, normalize_overlap=False):
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    if normalize_overlap:
        coverage = torch.zeros_like(x)
        for i in range(0, h + step - ps, step):
            top = i
            down = i + ps
            if down > h:
                top = h - ps
                down = h
            for j in range(0, w + step - ps, step):
                left = j
                right = j + ps
                if right > w:
                    left = w - ps
                    right = w
                coverage[:, :, top:down, left:right] += 1
        return output / coverage.clamp_min(1)

    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


class Attention(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA(nn.Module):
    """Local spatial relation aggregation: patch self-attention."""

    def __init__(self, dim, qk_dim, mlp_dim, heads=1, normalize_overlap=False):
        super().__init__()
        self.normalize_overlap = bool(normalize_overlap)
        self.layer = nn.ModuleList([
            PreNorm(dim, Attention(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')
        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c -> b n c h w', n=n, w=pw)
        x = patch_reverse(
            crop_x, x, step, ps, normalize_overlap=self.normalize_overlap
        )
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        return rearrange(x, 'b (h w) c -> b c h w', h=h)


# ---------------------------------------------------------------- Net
class Net(nn.Module):
    def __init__(
            self, scale=2, n_feats=48, side_c=32, n_stage=8,
            normalize_overlap=False):
        super(Net, self).__init__()
        self.scale = int(scale)
        dim = n_feats
        qk_dim = 36
        mlp_dim = 96
        heads = 4
        patch_size = [16, 20, 24, 28, 16, 20, 24, 28]
        num_tokens = [16, 32, 64, 128, 16, 32, 64, 128]
        group_size = [256, 128, 64, 32, 256, 128, 64, 32]

        self.first_conv = nn.Conv2d(3, dim, 3, 1, 1)
        self.fea = fea_Net()
        self.sfmls = nn.ModuleList([SFML(side_c, 12, dim) for _ in range(n_stage)])
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.esas = nn.ModuleList([ESA(12, dim, default_conv) for _ in range(n_stage)])
        for i in range(n_stage):
            self.blocks.append(nn.ModuleList([
                TAB(dim, qk_dim, mlp_dim, heads, n_iter=3,
                    num_tokens=num_tokens[i], group_size=group_size[i]),
                LRSA(
                    dim, qk_dim, mlp_dim, heads,
                    normalize_overlap=normalize_overlap,
                )]))
            self.mid_convs.append(nn.Conv2d(dim, dim, 3, 1, 1))
        self.patch_size = patch_size
        if self.scale == 4:
            # x4: two-stage upsampling (upconv1 -> PixelShuffle(2) -> upconv2 -> PixelShuffle(2))
            self.upconv1 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(dim, dim * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        else:
            self.upconv = nn.Conv2d(dim, dim * self.scale ** 2, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(self.scale)
        self.last_conv = nn.Conv2d(dim, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x0 = self.first_conv(x)
        fs = self.fea(x)
        feat = x0
        for i in range(len(self.blocks)):
            prev = feat
            beta, gamma = self.sfmls[i](fs)               # FS shared prior
            fm = beta * prev + gamma                      # eq6 FM
            g_attn, l_attn = self.blocks[i]
            t = g_attn(fm)                                # HTAB
            s = l_attn(t, self.patch_size[i])             # HLRSA
            feat = self.esas[i](prev + self.mid_convs[i](s))  # eq7 Xi
        if self.scale == 4:
            u = self.lrelu(self.pixel_shuffle(self.upconv1(x0 + feat)))
            u = self.lrelu(self.pixel_shuffle(self.upconv2(u)))
        else:
            u = self.lrelu(self.pixel_shuffle(self.upconv(x0 + feat)))
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                             align_corners=False)          # HBL(ILR)
        return self.last_conv(u) + base


def make_model(args):
    scale = args.scale[0] if hasattr(args, 'scale') and args.scale else 2
    return Net(scale=scale)
