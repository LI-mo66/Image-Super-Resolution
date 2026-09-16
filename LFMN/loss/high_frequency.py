"""Cheap high-frequency consistency loss for efficient SR screening."""
import torch.nn as nn
import torch.nn.functional as F


class HighFrequencyL1Loss(nn.Module):
    """L1 distance between local high-pass residuals of SR and HR images."""

    @staticmethod
    def high_pass(x):
        low = F.avg_pool2d(
            x, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        return x - low

    def forward(self, sr, hr):
        return F.l1_loss(self.high_pass(sr), self.high_pass(hr))
