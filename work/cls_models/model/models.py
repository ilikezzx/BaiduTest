"""
# File       : models.py
# Time       ：2023/5/7 21:59
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import os
import time
import random
import numpy as np
import paddle
import paddle.nn as nn

from resblock import BasicBlock, Bottleneck, conv3x3, conv1x1

from paddle.regularizer import L2Decay
from paddle.nn.initializer import KaimingNormal
from paddle import ParamAttr

seed = 0

np.random.seed(seed)
paddle.seed(seed)


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Module(nn.Layer):
    """分类器模块抽象类"""

    def __init__(self):
        super(Module, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        state = paddle.load(path)
        self.set_state_dict(state['model_state_dict'])

    def save(self, name=None):
        if name is None:
            suffix = 'checkpoints' + os.sep + self.model_name + "_"
            name = time.strftime(suffix + "%m%d_%H:%M:%S.pth")
        paddle.save(self.state_dict(), name)


class Flatten(nn.Layer):
    """将切块拍平"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        # print(x.shape)
        batch_size = x.shape[0]
        return x.flatten(start_axis=1, stop_axis=-1)


# 统一版Model
class ResNet(Module):
    def __init__(self, img_shape, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=nn.MaxPool2D, class_num: int = 1):
        """
        :param in_channel:   输入图像的通道
        :param every_layers_message:
            a list of [(layer_num, out_channel),(...),...,]
                --  layer_num: 特征提取，数值越大，模型越深
                --  out_channel: 每个block的输出通道数
        :param class_num: 分类数，二分类数值 == 1
        """
        super(ResNet, self).__init__()
        if class_num == 2:
            class_num = 1
        # isBasic = 1 if layer_type == BasicBlock else 4
        self.h, self.w = img_shape, img_shape

        assert len(every_layers_message) >= 3, "please check layers_num"
        last_out_channel = None
        self.blocks = []
        for index, (layer_num, out_channel) in enumerate(every_layers_message):
            # appending extract features layer
            for _ in range(layer_num):
                self.blocks.append(layer_type(in_channel, out_channel))
                # print(in_channel, out_channel)
                in_channel = out_channel

            last_out_channel = out_channel

            # appending down sample layer
            if downsample_type == nn.MaxPool2D:
                self.blocks.append(downsample_type(kernel_size=3, stride=2, padding=1))
            else:
                self.blocks.append(downsample_type(out_channel, out_channel))

        self.blocks = nn.LayerList(self.blocks)
        self.ef = layer_type(last_out_channel, last_out_channel)
        # last_layer
        self.downsample = nn.Sequential(conv1x1(last_out_channel, last_out_channel),
                                        nn.BatchNorm2D(last_out_channel),
                                        nn.ReLU(True),
                                        Flatten())

        self.h, self.w = self.h // (2 ** len(every_layers_message)), self.w // (2 ** len(every_layers_message))

        self.fc = nn.Sequential(
            nn.Linear(self.h * self.w * last_out_channel, 32),
            nn.Hardswish(),
            nn.Linear(32, class_num)
        )
        # active
        # self.active = None
        # if class_num == 1:
        #     self.active = nn.Sigmoid()
        # else:
        #     self.active = nn.Softmax()

    def forward(self, x):
        for block in self.blocks:
            # print(x.shape)
            x = block(x)
        out = self.downsample(self.ef(x))
        out = self.fc(out)
        # out = self.active(out)
        return out


class ResNet_basic111_maxpool(ResNet):
    def __init__(self, img_shape, in_channel: int, every_layers_message: list, layer_type=BasicBlock,
                 downsample_type=nn.MaxPool2D, class_num: int = 1):
        super(ResNet_basic111_maxpool, self).__init__(img_shape, in_channel, every_layers_message,
                                                      layer_type=layer_type,
                                                      downsample_type=downsample_type, class_num=class_num)


if __name__ == '__main__':
    # images = np.random.rand(64, 64, 64)
    # # print(images.shape)
    # images = torch.tensor(images, dtype=torch.float32)
    # images = images.unsqueeze(0)
    # images = images.unsqueeze(0)
    # model = Classifier_2(1, [16, 32, 64, 64])
    # pred = model(images)

    # sum1 = sum(x.numel() for x in Classifier_1(1, [32, 64, 64, 64]).parameters())
    # sum2 = sum(x.numel() for x in Classifier_2(1, [32, 64, 64, 64]).parameters())
    #
    # print(sum1, sum2)
    print(paddle.fluid.install_check.run_check())
    print(paddle.device.get_device())
    paddle.device.set_device('cpu')  # 把get—device的结果直接复制进去
    x = paddle.rand([2, 3, 320, 320])
    model = ResNet_basic111_maxpool(x.shape, x.shape[1], [(1, 32), (1, 64), (1, 64), (1, 64)], class_num=7)
    pred = model(x)
    print(pred.shape, x.shape)
