"""
# File       : label_split.py
# Time       ：2023/5/5 23:13
# Author     ：zzx
# version    ：python 3.10
# Description：
"""

import numpy as np
from tqdm import tqdm

np.random.seed(5)

labels_txt = r'./multi_label.txt'
train_label_path = r'train_label.txt'
valid_label_path = r'valid_label.txt'
f_train = open(train_label_path, 'w')
f_valid = open(valid_label_path, 'w')

with open(labels_txt, 'r') as f:
    lines = f.readlines()

    rand_num = np.random.randint(10, size=len(lines))
    for index, line in tqdm(enumerate(lines)):
        if rand_num[index] in [1]:
            f_valid.writelines(line)
        else:
            f_train.writelines(line)
f_valid.close()
f_train.close()
