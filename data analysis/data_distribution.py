"""
# File       : data_distribution.py
# Time       ：2023/5/13 21:53
# Author     ：zzx
# version    ：python 3.10
# Description：

得到数据的分布，并画图
包括 每个类别的数量 （颜色、型体）   以及 每个boxes的面积，长宽比分布
"""
import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

data_dir = r'../data/train/train'

boxes_num = 0
attribute = {'red': 0, "blue": 0, "green": 0, "bold": 0, "italic": 0, "delete_line": 0, "underline": 0}
boxes_area = []
boxes_width = []
boxes_height = []
boxes_wh_ratio = []

for file_name in tqdm(os.listdir(data_dir)):
    if not file_name.endswith('txt'):
        continue

    label_path = os.path.join(data_dir, file_name)

    with open(label_path, 'r') as f:
        json_data = json.load(f)
        # 可能有多个目标
        for index, label in enumerate(json_data):
            box_points = np.array(label["box_points"])
            box_attribute = label["box_attribute"]

            # print(box_points, box_attribute)

            # 切割出矩形框切割图形
            x_min = np.min(box_points[:, 0])
            x_max = np.max(box_points[:, 0])
            y_min = np.min(box_points[:, 1])
            y_max = np.max(box_points[:, 1])

            # 写入label
            x1, y1 = box_points[0, :]
            x2, y2 = box_points[1, :]
            x3, y3 = box_points[2, :]
            x4, y4 = box_points[3, :]
            rbox = [x1, y1, x2, y2, x3, y3, x4, y4]  # 可能是带旋转的
            rbox = [int(rb) for rb in rbox]

            w_box = x_max - x_min
            h_box = y_max - y_min
            box_ann = [x_min, y_min, w_box, h_box]
            box_ann = [int(b) for b in box_ann]

            attribute['red'] += int(box_attribute['red'])
            attribute['green'] += int(box_attribute['green'])
            attribute['blue'] += int(box_attribute['blue'])
            attribute['bold'] += int(box_attribute['bold'])
            attribute['italic'] += int(box_attribute['italic'])
            attribute['delete_line'] += int(box_attribute['delete_line'])
            attribute['underline'] += int(box_attribute['underline'])

            boxes_width.append(w_box)
            boxes_height.append(h_box)

            boxes_area.append(w_box * h_box)
            boxes_wh_ratio.append(float(1.0 * w_box / h_box))
            boxes_num += 1

print("attribute=", attribute)
print("box num: ", boxes_num)

plt.subplot(1, 4, 1)
plt.hist(boxes_area, bins=40)
plt.xlabel("area")

plt.subplot(1, 4, 2)
plt.hist(boxes_width, bins=40)
plt.xlabel("width")

plt.subplot(1, 4, 3)
plt.hist(boxes_height, bins=10)
plt.xlabel("height")

plt.subplot(1, 4, 4)
plt.hist(boxes_wh_ratio, bins=10)
plt.xlabel("ratio")

plt.show()

