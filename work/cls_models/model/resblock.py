"""
# File       : resblock.py
# Time       ：2023/5/7 22:34
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

"""
    3D-ResNet
"""

import paddle
import paddle.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
from paddle import ParamAttr


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, padding: int = 1,
            groups: int = 1) -> nn.Conv2D:
    """3x3x3 convolution padding"""
    return nn.Conv2D(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding,
                     groups=groups, dilation=dilation, weight_attr=ParamAttr(initializer=KaimingNormal()))


def conv1x1(in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1, padding: int = 0,
            groups: int = 1) -> nn.Conv2D:
    """1x1x1 convolution padding"""
    return nn.Conv2D(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding,
                     groups=groups, dilation=dilation, weight_attr=ParamAttr(initializer=KaimingNormal()))


class BasicBlock(nn.Layer):
    expansion: int = 1

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Layer] = None,
                 groups: int = 1,
                 padding: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Layer]] = None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding

        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, padding=padding)
        self.bn1 = norm_layer(out_channels, weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                              bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, padding=padding)
        self.bn2 = norm_layer(out_channels, weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                              bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.active = nn.Hardswish()

        if self.downsample is None:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride, padding=0),
                nn.BatchNorm2D(out_channels, weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                               bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
            )

    def forward(self, x):
        residual = x

        # print('1',x.shape)
        out = self.conv1(x)
        # print(out.shape)
        out = self.bn1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.active(out)
        return out


class Bottleneck(nn.Layer):
    expansion: int = 4

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 downsample: Optional[nn.Layer] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 padding: int = 1,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Layer]] = None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding

        self.conv1 = conv1x1(in_channels, out_channels, stride=self.stride, dilation=dilation, padding=padding)
        self.bn1 = norm_layer(out_channels, weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                              bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.conv2 = conv3x3(out_channels, out_channels, stride=self.stride, dilation=dilation, padding=padding)
        self.bn2 = norm_layer(out_channels, weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                              bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion, stride=self.stride, dilation=dilation,
                             padding=padding)
        self.bn3 = norm_layer(out_channels * self.expansion, weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                              bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.active = nn.Hardswish()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.active(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # connection
        out += residual
        out = self.active(out)
        return out


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inputs):
        return inputs
