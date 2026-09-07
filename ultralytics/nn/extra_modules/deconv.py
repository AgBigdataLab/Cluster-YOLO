"""Detail-enhanced convolution blocks used by DSRHead.

This minimal implementation preserves the parameter layout of the training
code so that the published InfloClusterNet checkpoint can be loaded directly.
"""

import math

import torch
import torch.nn as nn

from ..modules import Conv


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        flat = weight.reshape(*weight.shape[:2], -1)
        result = flat.clone()
        result[:, :, 4] = flat[:, :, 4] - flat.sum(2)
        return result.reshape_as(weight), self.conv.bias


class Conv2d_ad(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.theta = theta

    def get_weight(self):
        weight = self.conv.weight
        flat = weight.reshape(*weight.shape[:2], -1)
        result = flat - self.theta * flat[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]
        return result.reshape_as(weight), self.conv.bias


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.theta = theta

    def forward(self, x):
        if math.fabs(self.theta) < 1e-8:
            return self.conv(x)
        weight = self.conv.weight
        flat = weight.reshape(*weight.shape[:2], -1)
        result = weight.new_zeros(weight.shape[0], weight.shape[1], 25)
        result[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = flat[:, :, 1:]
        result[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -flat[:, :, 1:] * self.theta
        result[:, :, 12] = flat[:, :, 0] * (1 - self.theta)
        return nn.functional.conv2d(x, result.reshape(weight.shape[0], weight.shape[1], 5, 5),
                                    self.conv.bias, self.conv.stride, self.conv.padding,
                                    groups=self.conv.groups)


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        result = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
        result[:, :, [0, 3, 6]] = weight
        result[:, :, [2, 5, 8]] = -weight
        return result.reshape(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

    def get_weight(self):
        weight = self.conv.weight
        result = weight.new_zeros(weight.shape[0], weight.shape[1], 9)
        result[:, :, [0, 1, 2]] = weight
        result[:, :, [6, 7, 8]] = -weight
        return result.reshape(weight.shape[0], weight.shape[1], 3, 3), self.conv.bias


class DEConv(nn.Module):
    """Aggregate central, horizontal, vertical, angular and standard kernels."""

    def __init__(self, dim):
        super().__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_4 = Conv2d_ad(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(dim)
        self.act = Conv.default_act

    def forward(self, x):
        if hasattr(self, "conv1_1"):
            w1, b1 = self.conv1_1.get_weight()
            w2, b2 = self.conv1_2.get_weight()
            w3, b3 = self.conv1_3.get_weight()
            w4, b4 = self.conv1_4.get_weight()
            w5, b5 = self.conv1_5.weight, self.conv1_5.bias
            x = nn.functional.conv2d(x, w1 + w2 + w3 + w4 + w5,
                                     b1 + b2 + b3 + b4 + b5, padding=1)
        else:
            x = self.conv1_5(x)
        if hasattr(self, "bn"):
            x = self.bn(x)
        return self.act(x)

    def switch_to_deploy(self):
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w4, b4 = self.conv1_4.get_weight()
        self.conv1_5.weight = nn.Parameter(w1 + w2 + w3 + w4 + self.conv1_5.weight)
        self.conv1_5.bias = nn.Parameter(b1 + b2 + b3 + b4 + self.conv1_5.bias)
        del self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4


__all__ = ("Conv2d_cd", "Conv2d_ad", "Conv2d_rd", "Conv2d_hd", "Conv2d_vd", "DEConv")
