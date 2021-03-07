#!/usr/bin/env python3
# coding: utf-8

"""
@File      : data_trans.py
@Author    : alex
@Date      : 2021/1/24
@Desc      : 
"""

import cv2
import math
import random
import numpy as np
from addict import Dict
from enum import Enum, unique
from alcore.common.utils import *
from torch.utils.data import Dataset


class SegmentData(Dataset):
    NAME = "SegmentData"

    IMAGES_DIR = "imgs"
    TRAIN_FILE = "train.txt"
    VALID_FILE = "valid.txt"

    def __init__(self, data_dir, test_mode: bool = True, shape=(128, 128), transform=None):
        self.shape = shape
        self.data_dir = data_dir
        self.transform = transform
        self.test_mode = test_mode

        # mesh_square, mesh_circle
        self.classes = ["scraper"]
        self._data_dicts = self._data_prepare()

    @property
    def class_idx(self):
        if self.classes is None:
            return
        return {i: v for i, v in enumerate(self.classes)}

    def __getitem__(self, index):
        scale = 4

        img_path = self._data_dicts[index].img_path
        label_path = self._data_dicts[index].label_path
        img = cv2.imread(img_path)

        height, width = self.shape
        img = cv2.resize(img, (width, height))
        if self.transform:
            img = self.transform(img)
        mask = cv2.imread(label_path, -1)
        # print("info:", self._data_dicts[index], mask)
        assert len(mask.shape) == 2, f"mask shape should like (height, width), {mask.shape}"

        label = np.zeros(mask.shape, np.uint8)
        label[mask == 0] = 255
        label = cv2.resize(label, (width, height))
        label[label <= 128] = 0
        label[label > 128] = 1
        label = np.reshape(label, (1, height, width))
        return img, label.astype(np.float32)

    def __len__(self):
        return len(self._data_dicts)

    def _data_prepare(self):
        data_set = self.VALID_FILE if self.test_mode else self.TRAIN_FILE
        file_paths = CText(path.join(self.data_dir, data_set)).read_lines()
        # print(file_paths, len(file_paths))

        dicts = []
        for idx, img_path in enumerate(file_paths):
            anno_path = path.join(self.data_dir, "labels", keyname(img_path) + ".png")
            # print(idx, img_path, anno_path)
            if not path.exists(anno_path):
                continue

            r = Dict({
                "img_path": img_path,
                "label_path": anno_path
            })
            dicts.append(r)
        return dicts

    def reload(self):
        self._data_dicts = self._data_prepare()

    def split(self, train_rate=0.95, shuffle=True, reload=True):
        """
        将原始数据，分成训练、测试数据集
        :param train_rate: 训练集样本比例
        :param shuffle: 是否打乱数据集
        """

        assert train_rate <= 1 + 1e-6, "train rate should be < 1."

        img_paths = listdir(path.join(self.data_dir, self.IMAGES_DIR), filter=".jpg$", real_path=True)
        # print("img_paths:", img_paths, len(img_paths))

        file_paths = []
        for idx, img_path in enumerate(img_paths):
            anno_path = path.join(self.data_dir, "labels", keyname(img_path) + ".png")
            if not path.exists(anno_path):
                print("warn: anno_path '%s' not exists, it will be ignored." % anno_path)
                continue
            file_paths.append(img_path)
        total_cnt = len(file_paths)
        print("total img cnt:", len(img_paths), "total cnt:", total_cnt)

        if shuffle:
            random.shuffle(file_paths)

        train_file = CText(path.join(self.data_dir, self.TRAIN_FILE), is_clear=True)
        valid_file = CText(path.join(self.data_dir, self.VALID_FILE), is_clear=True)
        for idx, file_path in enumerate(file_paths):
            if idx < int(total_cnt * train_rate):
                train_file.append(file_path + "\n")
            else:
                valid_file.append(file_path + "\n")

        if reload:
            self.reload()


if __name__ == "__main__":
    data_dir = r"F:\dataset\scraper_seg"
    dataset = SegmentData(data_dir, test_mode=False, shape=(256, 512))
    # dataset.split(train_rate=0.9)
    print(len(dataset))
    img, mask = dataset[0]
    print("xxx:", img.shape)
    print("gt_mask:", np.min(mask), np.max(mask), mask.shape, type(mask), mask.dtype)

    cv2.imwrite(path.join(data_dir, "img.jpg"), img)
    mask = mask[0, :, :]
    cv2.imwrite(path.join(data_dir, "mask.jpg"), cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR))
    pass
