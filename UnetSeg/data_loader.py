#!/usr/bin/env python3
# coding: utf-8

"""
@File      : data_loader.py
@Author    : alex
@Date      : 2021/1/23
@Desc      : 
"""

from dataset import SegmentData
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ["train_loader", "valid_loader", "n_cls"]

trans = transforms.Compose([transforms.ToTensor(), ])

data_dir = r"F:\dataset\scraper_seg"
# use same transform for train/val for this example
train_set = SegmentData(data_dir, test_mode=False, shape=(256, 512), transform=trans)
valid_set = SegmentData(data_dir, test_mode=True, shape=(256, 512), transform=trans)
n_cls = len(train_set.classes)

train_loader = DataLoader(train_set,
                          batch_size=48,
                          shuffle=True,
                          num_workers=6)

valid_loader = DataLoader(valid_set,
                          batch_size=4,
                          num_workers=4)

if __name__ == "__main__":
    # test_set = SegmentData(data_dir, shape=(192, 192), type=ShapeType.segment, transform=trans)
    # test_loader = DataLoader(valid_set,
    #                          batch_size=6,
    #                          num_workers=0)

    print("data_dir:", data_dir)
    images, masks = next(iter(valid_loader))
    print(images.shape, masks.shape)
    for x in [images.numpy(), masks.numpy()]:
        print(x.min(), x.max(), x.mean(), x.std())
    pass
