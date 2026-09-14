import math
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def show_feature_map(feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.detach().numpy().squeeze()  # Squeeze to torch.Size([64, 55, 55])
    feature_map_num = feature_map.shape[0]  # Return the number of channels

    for index in range(feature_map_num):  # Iterate over the 64 channels and extract each tensor
        feature = feature_map[index]
        feature = np.asarray(feature * 255, dtype=np.uint8)
        feature = cv2.resize(feature, (224, 224), interpolation=cv2.INTER_NEAREST)  # Resize the feature map
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_JET)  # Convert to a pseudo-color image
        dirname = "/home/wangziqiang/feature_mapps"
        os.makedirs(dirname, exist_ok=True)
        cv2.imwrite("/home/wangziqiang/feature_mapps/channel_{}.png".format(str(index)), feature)

# Define the shared feature generation network
class fea_Net(nn.Module):
    def __init__(self):
        super(fea_Net, self).__init__()
        self.net = nn.Sequential(
            default_conv(3, 8, 3, bias=False), nn.ReLU(True),
            default_conv(8, 16, 3, bias=False), nn.ReLU(True),
            default_conv(16, 32, 3, bias=False), nn.ReLU(True)
        )
    def forward(self, x):
        x = self.net(x)
        return x 


# # Define multi-scale channel attention
# class MSCA(nn.Module):
#     def __init__(self, in_channels, ratio=4):
#         super(MSCA, self).__init__()
#         self.local_att = nn.Sequential(nn.Conv2d(in_channels,in_channels//ratio, 1),
#                                       nn.ReLU(True), nn.Conv2d(in_channels//ratio, in_channels, 1))
#         self.global_att = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(in_channels,in_channels//ratio, 1),
#                                         nn.ReLU(True), nn.Conv2d(in_channels//ratio, in_channels, 1))
#         self.act = nn.Sigmoid()
#
#     def forward(self, x):
#         local_att = self.local_att(x)
#         global_att = self.global_att(x)
#         att = self.act(local_att + global_att)
#
#         return att


# Define the feature modulation layer
class FML(nn.Module):
    def __init__(self, in_c, ratio=4, down_scale=None):
        super(FML, self).__init__()
        mid = in_c // down_scale if down_scale else in_c // ratio
        mid = max(1, mid)

        self.FML_scale_conv0 = nn.Conv2d(in_c, mid, 1)
        self.FML_scale_conv1 = nn.Conv2d(mid, in_c, 1)
        self.FML_shift_conv0 = nn.Conv2d(in_c, mid, 1)
        self.FML_shift_conv1 = nn.Conv2d(mid, in_c, 1)
        # self.msca = MSCA(in_c)

    def forward(self, x):
        # x[0]: fea; x[1]: input
        # low_fea = x[1]
        # att = self.msca(x[0])
        # low_fea = low_fea * att
        scale = self.FML_scale_conv1(F.leaky_relu(self.FML_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.FML_shift_conv1(F.leaky_relu(self.FML_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


# Define a convolution operation that keeps H and W unchanged
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

# coord attention
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         # self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         # y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h
#
#         return out

# ESA
class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


# Residual block with an FML layer
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        self.res_scale = res_scale

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.act = act
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)

        # Key: map Fs (32 channels) to n_feats (64 channels) to match FML(in_c=64)
        self.cond_proj = nn.Conv2d(32, n_feats, 1, 1, 0)

        # FML uses in_c=n_feats
        self.fml = FML(n_feats, ratio=4, down_scale=16)  # down_scale=16 is more stable

        self.esa = ESA(esa_channels=10, n_feats=n_feats, conv=conv)
    def forward(self, x):
        Fea, Fs = x  # Fea: n_feats, Fs: 32ch from fea_Net
        Fs64 = self.cond_proj(Fs)

        res = self.act(self.conv1(Fea))
        res = self.fml((res, Fs64))
        res = self.conv2(res)
        res = self.esa(res)

        return (Fea + res * self.res_scale, Fs)




class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
