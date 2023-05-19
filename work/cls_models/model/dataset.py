"""
# File       : dataset.py
# Time       ：2023/5/7 23:24
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import os
from PIL import Image
import numpy as np
from transform import ResizePad

import paddle
from paddle.vision.transforms import to_tensor
from paddle.io import Dataset


def process_image(img, mode, rotate, resize_with_pad):
    img = img.convert('RGB')
    img = np.array(img)
    img = ResizePad(img, target_size=resize_with_pad)
    img = np.array(img).astype('float32').transpose((2, 0, 1))
    img = img / 255.0

    return img


class AngleClass(Dataset):

    def __init__(self, Config, mode, isFine):
        super().__init__()
        base_dataset_path = Config["dataset"]["base_dataset_path"]
        base_train_label_path = Config["dataset"]["base_train_label_path"]
        base_test_label_path = Config["dataset"]["base_test_label_path"]

        self.train_img, self.train_label, self.mode = [], [], mode

        self.resize_with_pad = Config["dataset"]["resize_pad"]
        self.isFineTuning = False
        if mode == 'train':
            load_p = base_train_label_path
        else:
            load_p = base_test_label_path
        self.labels = []
        with open(load_p, 'r') as f:
            lines = f.readlines()
        for (lid, line) in enumerate(lines):
            img_name, multi_label = line.split('\t')

            if not img_name.endswith(".jpg"):
                continue
            file_path = os.path.join(base_dataset_path, img_name)
            img = Image.open(file_path)

            if img is not None:
                labels = multi_label.split(',')
                labels = [np.int64(i) for i in labels]
                self.train_img.append(file_path)
                self.train_label.append(labels)

    def __len__(self):
        return len(self.train_img)

    def __getitem__(self, idx):
        img_path = self.train_img[idx]
        img = Image.open(img_path)
        label = np.array(self.train_label[idx]).astype("float32")
        img = np.array(img)
        h1, w1 = img.shape[:-1]
        img = Image.fromarray(img)

        img = process_image(img, self.mode, True, self.resize_with_pad)
        label = paddle.to_tensor(label)
        return img, label