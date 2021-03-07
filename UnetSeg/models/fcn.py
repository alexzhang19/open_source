#!/usr/bin/env python3
# coding: utf-8

"""
@File      : fcn.py
@Author    : alex
@Date      : 2021/3/6
@Desc      : 
"""

import torch
from torch import nn
from torchvision import models


class FCN(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        base_model = models.resnet18(pretrained=True)
        layers = list(base_model.children())
        self.layer1 = nn.Sequential(*layers[:5])  # size=(N, 64, x.H/2, x.W/2)

        self.upsample1 = nn.Upsample(scale_factor=4, mode='nearest')
        self.layer2 = layers[5]  # size=(N, 128, x.H/4, x.W/4)
        self.upsample2 = nn.Upsample(scale_factor=8, mode='nearest')
        self.layer3 = layers[6]  # size=(N, 256, x.H/8, x.W/8)
        self.upsample3 = nn.Upsample(scale_factor=16, mode='nearest')
        self.layer4 = layers[7]  # size=(N, 512, x.H/16, x.W/16)
        self.upsample4 = nn.Upsample(scale_factor=32, mode='nearest')
        self.conv1k = nn.Conv2d(64 + 128 + 256 + 512, n_class, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        print("backbone:", x.shape)  # [1, 64, 56, 56]

        up1 = self.upsample1(x)
        x = self.layer2(x)
        up2 = self.upsample2(x)
        x = self.layer3(x)
        up3 = self.upsample3(x)
        x = self.layer4(x)
        up4 = self.upsample4(x)
        merge = torch.cat([up1, up2, up3, up4], dim=1)
        merge = self.conv1k(merge)
        out = self.sigmoid(merge)
        return out


if __name__ == "__main__":
    import onnx
    from onnxsim import simplify

    model = FCN(4)
    print("model:", model)

    example = torch.rand(1, 3, 224, 224)
    model(example)

    # input_names = ["input_0"]
    # output_names = ["output_0"]
    # output_path = "fcn.onnx"
    # torch.onnx.export(model, example,
    #                   output_path,
    #                   verbose=True,
    #                   opset_version=11,
    #                   input_names=input_names,
    #                   output_names=output_names)
    # print("out_put_shape:", model(example).shape)
    #
    # onnx_model = onnx.load(output_path)  # load onnx model
    # model_simp, check = simplify(onnx_model)
    # assert check, "Simplified ONNX model could not be validated"
    # onnx.save(model_simp, output_path)
    pass
