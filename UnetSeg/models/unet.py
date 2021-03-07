#!/usr/bin/env python3
# coding: utf-8

"""
@File      : unet.py
@Author    : alex
@Date      : 2021/1/23
@Desc      : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet"]


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class ResizeModel(nn.Module):
    def __init__(self):
        super(ResizeModel, self).__init__()

    def forward(self, x):
        if self.training:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            # hack in order to generate a simpler onnx
            x = F.interpolate(x, size=[int(2 * x.shape[2]), int(2 * x.shape[3])], mode='nearest')
        return x


class UNet(nn.Module):
    NAME = "UNet"

    def __init__(self, n_cls, scale=8):
        super().__init__()

        self.dconv_down1 = double_conv(3, 64 // scale)
        self.dconv_down2 = double_conv(64 // scale, 128 // scale)
        self.dconv_down3 = double_conv(128 // scale, 256 // scale)
        self.dconv_down4 = double_conv(256 // scale, 512 // scale)

        self.maxpool = nn.MaxPool2d(2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample = ResizeModel()

        self.dconv_up3 = double_conv((256 + 512) // scale, 256 // scale)
        self.dconv_up2 = double_conv((128 + 256) // scale, 128 // scale)
        self.dconv_up1 = double_conv((128 + 64) // scale, 64 // scale)

        self.conv_last = nn.Conv2d(64 // scale, n_cls, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        out = self.conv_last(x)
        return out


if __name__ == "__main__":
    import torch as t

    x = t.randn(1, 3, 256, 256)
    model = UNet(1)
    print("models:", model)
    y = model(x)
    print("y:", y.shape)
    pass
