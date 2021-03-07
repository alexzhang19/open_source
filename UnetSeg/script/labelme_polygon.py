#!/usr/bin/env python3
# coding: utf-8

"""
@File      : labelme_polygon.py
@Author    : alex
@Date      : 2021/1/24
@Desc      : https://www.cnblogs.com/bjxqmy/p/13462863.html
"""

import cv2
import numpy as np
from alcore.common import *


def read_polygon(json_path):
    info = CJson(json_path).load()
    # print("info:", info)

    points = []
    # height, width = info["imageHeight"], info["imageWidth"]
    for shape in info["shapes"]:
        label = shape["label"]
        if label not in names:
            continue

        region = []
        for pts in shape["points"]:
            region.append([np.array(pts).astype(np.int32)])
        points.append(np.array(region))
    return points


def main():
    file_paths = walk_file(data_dir, filter=".json$")
    print(f"total label cnt is: {len(file_paths)}")

    ret_img_dir = path.join(ret_dir, "imgs")
    ret_label_dir = path.join(ret_dir, "labels")
    ret_vis_dir = path.join(ret_dir, "virs")
    mkdir(ret_img_dir)
    mkdir(ret_label_dir)
    mkdir(ret_vis_dir)

    for idx, file_path in enumerate(file_paths):
        print(idx, file_path)
        img_path = keyname(file_path, real_path=True) + ".jpg"
        cp(img_path, path.join(ret_img_dir, path.basename(img_path)))

        img = cv2.imread(img_path)
        points = read_polygon(file_path)
        mask = np.ones(img.shape[:2], np.uint8) * 255
        cv2.drawContours(mask, points, contourIdx=-1, color=0, thickness=-1)
        ret_mask_path = path.join(ret_label_dir, keyname(img_path) + ".png")
        cv2.imwrite(ret_mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        mask = cv2.imread(ret_mask_path, -1)
        vis_img = draw_mask(img, mask)
        cv2.imwrite(path.join(ret_vis_dir, path.basename(img_path)), vis_img)


if __name__ == "__main__":
    data_dir = r"F:\data\20201015_scraper1"
    names = ["scraper", "1"]

    ret_dir = r"F:\dataset\scraper_seg"
    mkdir(ret_dir)

    main()
    pass
