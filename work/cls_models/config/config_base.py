"""
# File       : config_base.py
# Time       ：2023/5/5 23:21
# Author     ：zzx
# version    ：python 3.10
# Description：
"""
Config = {
    'dataset': {
        'base_dataset_path': r'../../data/train/train_crop_img',
        "base_ratio": 1,
        "fine_ratio": 1,
        'base_train_label_path': '../train_label.txt',
        'base_test_label_path': '../valid_label.txt',
        "aug_dataset_list": [
        ],
        "aug_ratio_list": [],
        "train_batch": 512,
        "test_batch": 256,
        "resize_pad": 224,
    },
    'solver': {
        'seed': 0,
        'base_lr': 0.5 * 1e-2,
        'milestones': [10, 20, 30, 40, 50],

        'gamma': 0.5,
        'num_epoch': 60,
        'loss_print_freq': 20,
    },
    'models': {
        'scale': 1,
        'pretrain_path': '',
        'save_path': 'mutilabel_base',
        'interrupt_path': 'multilabel_interrupt',
    }
}