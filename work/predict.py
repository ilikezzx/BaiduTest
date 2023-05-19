import os
import sys
import glob
import json
import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from cls_model import PPLCNet


def ResizePad(img, target_size):
    img = np.array(img)
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img1


def process_mutilabel(img, resize_with_pad):
    img = np.array(img)
    img = ResizePad(img, target_size=resize_with_pad)
    img = np.array(img).astype('float32').transpose((2, 0, 1))
    img = img / 255.0

    return img


def generate_scale(im, resize_shape, keep_ratio):
    """
    Args:
        im (np.ndarray): image (np.ndarray)
    Returns:
        im_scale_x: the resize ratio of X
        im_scale_y: the resize ratio of Y
    """
    target_size = (resize_shape[0], resize_shape[1])
    origin_shape = im.shape[:2]

    if keep_ratio:
        im_size_min = np.min(origin_shape)
        im_size_max = np.max(origin_shape)
        target_size_min = np.min(target_size)
        target_size_max = np.max(target_size)
        im_scale = float(target_size_min) / float(im_size_min)
        if np.round(im_scale * im_size_max) > target_size_max:
            im_scale = float(target_size_max) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale
    else:
        resize_h, resize_w = target_size
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
    return im_scale_y, im_scale_x


def resize(im, im_info, resize_shape, keep_ratio):
    interp = 2
    im_scale_y, im_scale_x = generate_scale(im, resize_shape, keep_ratio)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=interp)
    # print("scale,", im_scale_y, im_scale_x)
    im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
    im_info['scale_factor'] = np.array(
        [im_scale_y, im_scale_x]).astype('float32')

    return im, im_info


def process_yoloe(im, im_info, resize_shape):
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    # print(im)

    im, im_info = resize(im, im_info, resize_shape, False)
    h_n, w_n = im.shape[:-1]
    im = im / 255.0
    im = im.transpose((2, 0, 1)).copy()

    im = paddle.to_tensor(im, dtype='float32')
    im = im.unsqueeze(0)
    factor = paddle.to_tensor(im_info['scale_factor']).reshape((1, 2)).astype('float32')
    im_shape = paddle.to_tensor(im_info['im_shape'].reshape((1, 2)), dtype='float32')
    return im, im_shape, factor


MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    model = paddle.jit.load('ppyoloe_plus_crn_analysis/model')
    model.eval()
    multilabel_model = PPLCNet(scale=1, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"])
    load_path = 'base_best_model'
    multilabel_model.set_dict(paddle.load(load_path))
    multilabel_model.eval()
    result = {}
    for image_path in image_paths:
        filename = os.path.basename(image_path).replace(".jpg", ".txt")
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        im = np.asarray(im)
        im_copy = im.copy()

        im_info = {
            'scale_factor': np.array(
                [1., 1.], dtype=np.float32),
            'im_shape': None,
        }

        h1, w1 = im.shape[:-1]
        im, im_shape, factor = process_yoloe(im, im_info, [576, 576])
        with paddle.no_grad():
            pre = model(im, factor)

        result = []

        for item in pre[0].numpy():
            cls, value, xmin, ymin, xmax, ymax = list(item)
            cls, xmin, ymin, xmax, ymax = [int(x) for x in [cls, xmin, ymin, xmax, ymax]]
            if value > 0.3:
                # print(im_copy.shape)
                cls_img = im_copy[ymin:ymax, xmin:xmax, :]
                cls_img = process_mutilabel(cls_img, 224)
                cls_img = paddle.to_tensor(cls_img)
                cls_img = cls_img.unsqueeze(0)
                # print("cls",cls_img.shape)
                with paddle.no_grad():
                    label = multilabel_model(cls_img)
                label = F.sigmoid(label)
                label = label.squeeze()
                # print(label)
                label = label > 0.6
                label = label.numpy()
                # print(label.shape,label)
                # label = label.squeeze.numpy()

                # print(label.shape)
                # print(label)

                # cv2.imwrite('/home/aistudio/work/predict_code/test.jpg',cls_img)
                box_dict = {}

                box_dict["box_points"] = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
                box_dict["box_attribute"] = {
                    "red": int(label[0]),  # 目标框中文字的颜色是否为红色，是为1，否为0
                    "green": int(label[1]),  # 目标框中文字的颜色是否为绿色，是为1，否为0
                    "blue": int(label[2]),  # 目标框中文字的颜色是否为蓝色，是为1，否为0
                    "bold": int(label[3]),  # 目标框中文字是否加粗，是为1，否为0  
                    "italic": int(label[4]),  # 目标框中文字是否倾斜，是为1，否为0  
                    "delete_line": int(label[5]),  # 目标框中文字上是否有删除线，是为1，否为0
                    "underline": int(label[6]),  # 目标框中文字上是否有下划线，是为1，否为0
                }
                result.append(box_dict)
        result_str = json.dumps(result)
        # break
        with open(os.path.join(save_dir, filename), "w") as writeF:
            writeF.write(result_str)


if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)
