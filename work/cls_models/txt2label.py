"""
# File       : txt2label.py
# Time       ：2023/5/5 22:39
# Author     ：zzx
# version    ：python 3.10
# Description：
"""


import os
import cv2
import json
import json
import numpy as np
from tqdm import tqdm

labels_txt = r'./multi_label.txt'
train_dir = r'../../data/train/train'
save_dir = r'../../data/train/train_crop_img'

os.makedirs(save_dir, exist_ok=True)
f1 = open(labels_txt, 'w')

for img_name in tqdm(os.listdir(train_dir)):
    if not img_name.endswith('jpg'):
        continue

    base_name = img_name.replace('.jpg', '')
    img_path = os.path.join(train_dir, img_name)
    label_path = os.path.join(train_dir, img_name.replace('.jpg', '.txt'))
    img = cv2.imread(img_path)

    H, W, C = img.shape
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

            # 保存截图
            crop_img = img[y_min:min(y_max+1, H), x_min: min(x_max+1, W), :]
            save_path = os.path.join(save_dir, base_name+"_"+str(index)+".jpg")
            cv2.imwrite(save_path, crop_img)

            # 写入label
            x1, y1 = box_points[0, :]
            x2, y2 = box_points[1, :]
            x3, y3 = box_points[2, :]
            x4, y4 = box_points[3, :]
            rbox = [x1,y1, x2, y2, x3, y3, x4, y4]  # 可能是带旋转的
            rbox = [int(rb) for rb in rbox]

            w_box = x_max - x_min
            h_box = y_max - y_min
            box_ann = [x_min, y_min, w_box, h_box]
            box_ann = [int(b) for b in box_ann]

            red = box_attribute['red']
            green = box_attribute['green']
            blue = box_attribute['blue']
            bold = box_attribute['bold']
            italic = box_attribute['italic']
            delete_line = box_attribute['delete_line']
            underline = box_attribute['underline']

            f1.writelines(
                '{}\t{},{},{},{},{},{},{}\n'.format(base_name+"_"+str(index)+".jpg", red, green, blue, bold, italic, delete_line,
                                                    underline, ))

f1.close()


