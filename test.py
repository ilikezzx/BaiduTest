"""
# File       : test.py
# Time       ：2023/4/21 21:19
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import os
import cv2
import json
import numpy as np


if __name__ == '__main__':
    test_dir = r'./data/test/test'
    result = r'./data/save_result'

    for image_name in os.listdir(test_dir):
        if not image_name.endswith('jpg'): continue
        image_path = os.path.join(test_dir, image_name)
        image = cv2.imread(image_path)

        bbox = []
        pred_path = os.path.join(result, image_name.replace('.jpg', '.txt'))
        with open(pred_path, 'r') as f:
            json_data = json.load(f)

            for anno in json_data:
                box_point = np.array(anno['box_points'])
                box_attribute = anno['box_attribute']
                xmin = np.min(box_point[:, 0])
                xmax = np.max(box_point[:, 0])
                ymin = np.min(box_point[:, 1])
                ymax = np.max(box_point[:, 1])
                # x1, y1 = box_point[0, :]
                # x2, y2 = box_point[1, :]
                # x3, y3 = box_point[2, :]
                # x4, y4 = box_point[3, :]
                # rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
                # rbox = [int(rb) for rb in rbox]

                w_box = xmax - xmin
                h_box = ymax - ymin

                box_ann = [xmin, ymin, xmax, ymax]
                box_ann = [int(b) for b in box_ann]
                bbox.append(box_ann)
                print(box_ann)
                print(box_attribute)

        for box in bbox:
            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255))

        cv2.imshow('result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print('-'*20)






