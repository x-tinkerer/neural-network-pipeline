#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn
import math
import brevitas.onnx as bo
from brevitas.core.restrict_val import RestrictValueType
from brevitas.quant import IntBias
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d, QuantIdentity

MobileFaceNet_BottleNeck_Setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

FIRST_LAYER_BIT_WIDTH = 8

class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class BottleNeck(nn.Module):
    # def __init__(self, inp, oup, stride, expansion):
    def __init__(
            self, in_channels, out_channels,stride, expansion,
            weight_bit_width=8, act_bit_width=8, bn_eps=1e-5,
            activation_scaling_per_channel=False):
        super(BottleNeck, self).__init__()
        self.connect = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            # 1*1 conv
            # nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            QuantConv2d(
                in_channels=in_channels,
                out_channels=in_channels * expansion,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=weight_bit_width),
            # nn.BatchNorm2d(inp * expansion),
            nn.BatchNorm2d(num_features=in_channels * expansion, eps=bn_eps),
            #nn.ReLU(inp * expansion),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                per_channel_broadcastable_shape=(1, in_channels * expansion, 1, 1),
                scaling_per_channel=activation_scaling_per_channel,
                return_quant_tensor=True),

            # 3*3 depth wise conv
            # nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            QuantConv2d(
                in_channels=in_channels * expansion,
                out_channels=in_channels * expansion,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * expansion,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=weight_bit_width),

            # nn.BatchNorm2d(inp * expansion),
            nn.BatchNorm2d(num_features=in_channels * expansion, eps=bn_eps),

            #nn.ReLU(inp * expansion),
            QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                per_channel_broadcastable_shape=(1, in_channels * expansion, 1, 1),
                scaling_per_channel=activation_scaling_per_channel,
                return_quant_tensor=True),

            # 1*1 conv
            # nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            QuantConv2d(
                in_channels=in_channels * expansion,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=weight_bit_width),

            # nn.BatchNorm2d(oup),
            nn.BatchNorm2d(num_features=out_channels, eps=bn_eps),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class ConvBlock(nn.Module):
    # def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            stride=1, padding=0,
            weight_bit_width=8, act_bit_width=8,
            bn_eps=1e-5, activation_scaling_per_channel=False,
            dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            #self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
            self.conv = QuantConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=weight_bit_width)
        else:
            # self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
            self.conv = QuantConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                weight_quant=CommonIntWeightPerChannelQuant,
                weight_bit_width=weight_bit_width)

        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        if not linear:
            # self.relu = nn.ReLU(oup)
            self.relu = QuantReLU(
                act_quant=CommonUintActQuant,
                bit_width=act_bit_width,
                per_channel_broadcastable_shape=(1, out_channels, 1, 1),
                scaling_per_channel=activation_scaling_per_channel,
                return_quant_tensor=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.relu(x)


class MobileFaceNet(nn.Module):
    def __init__(self, feature_dim=512, bottleneck_setting=MobileFaceNet_BottleNeck_Setting):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.cur_channel = 64
        block = BottleNeck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t))
                else:
                    layers.append(block(self.cur_channel, c, 1, t))
                self.cur_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x


def mobilenet():
    model = MobileFaceNet()
    return model

if __name__ == "__main__":
    input = torch.Tensor(2, 3, 112, 112)
    net = MobileFaceNet(feature_dim=128)
    print(net)

    x = net(input)
    print(x.shape)

    model_for_export = "quant_mobilenet_v1_b4.pth"
    torch.save(net.state_dict(),model_for_export)

    onnx_model_export = "quant_mobilenet_v1_b4.onnx"
    input_shape = (2, 3, 112, 112)
    bo.export_finn_onnx(net, input_shape, export_path=onnx_model_export,)
    print("Model saved to %s" % model_for_export)