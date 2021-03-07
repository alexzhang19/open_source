#!/usr/bin/env python3
# coding: utf-8

"""
@File      : test.py
@Author    : alex
@Date      : 2021/1/24
@Desc      :
"""

import cv2
import torch
import numpy as np
from models.unet import UNet
from alcore.common import *
from data_loader import valid_set, valid_loader

g_ignore_label = 255


def reverse_transform_img(inp):
    inp = inp.data.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)
    return inp


def reverse_transform_maks(pred, th=0.5):
    pred = torch.sigmoid(pred).data.cpu().numpy().transpose((1, 2, 0))
    # print("pred:", pred.shape, np.min(pred), np.max(pred))

    height, width, n_cls = pred.shape
    bk_label = np.ones((height, width, 1), dtype=pred.dtype) * th
    output = np.append(pred, bk_label, axis=-1)
    output = np.asarray(np.argmax(output, axis=-1), dtype=np.uint8)
    # print("output:", output.shape, np.min(output), np.max(output))

    output[output == n_cls] = g_ignore_label
    return output


def vis_result(inputs, preds, key, labels=None):
    # print(key, inputs.shape, labels.shape, preds.shape)
    # preds = torch.sigmoid(preds).data.cpu().numpy()

    th = 0.3
    for i in range(inputs.shape[0]):
        input, pred, label = inputs[i], preds[i], labels[i]
        # print(key, i, input.shape, pred.shape, label.shape)

        # input vis
        img = reverse_transform_img(input)
        img_ret_dir = path.join(vis_ret_dir, "imgs")
        mkdir(img_ret_dir)
        cv2.imwrite(path.join(img_ret_dir, "%s_%02d.jpg" % (key, i)), img)

        # label vis
        mask = reverse_transform_maks(pred, th)
        # blank_img = np.ones(img.shape, dtype=np.uint8) * 255
        label_img = draw_mask(img, mask, alfa=0.6)
        label_ret_dir = path.join(vis_ret_dir, "labels")
        mkdir(label_ret_dir)
        cv2.imwrite(path.join(label_ret_dir, "%s_%02d.jpg" % (key, i)), label_img)


def test():
    n_cls = len(valid_set.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_cls).to(device)
    model.load_state_dict(torch.load("best_model_wts.pth", map_location=device))
    model.eval()

    for idx, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        print(idx, inputs.shape, labels.shape, preds.shape)
        vis_result(inputs, preds, "%04d" % idx, labels=labels)


if __name__ == "__main__":
    vis_ret_dir = r"C:\Users\Administrator\Desktop\vis_result"
    rm(vis_ret_dir)

    test()
    pass
