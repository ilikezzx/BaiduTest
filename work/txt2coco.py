import os
import cv2
import json
import numpy as np

images = []
categories = []
annotations = []
class_ids = {'text': 1}
class_id = 1
coco = {}
annid = 0
train_dir = r'../data/train/train'


def get_annotations(box, image_id, ann_id, calss_name, segment):
    annotation = {}
    w, h = box[2], box[3]
    area = w * h
    annotation['segmentation'] = [segment]
    annotation['iscrowd'] = 0
    # 第几张图像，从0开始
    annotation['image_id'] = image_id
    annotation['bbox'] = box
    annotation['area'] = float(area)
    # category_id=0
    annotation['category_id'] = class_ids[calss_name]
    # 第几个标注，从0开始
    annotation['id'] = ann_id
    # print(annotation)
    return annotation


def get_categories(name, class_id):
    category = {}
    # id=0
    category['id'] = class_id
    # name=1
    category['name'] = name
    # print(category)
    return category


def get_images(filename, height, width, image_id):
    image = {}
    image["height"] = height
    image['width'] = width
    image["id"] = image_id
    image["file_name"] = filename
    return image


idx = 0
img_id = 0

for filename in os.listdir(train_dir):
    if filename.endswith('.jpg'):
        img_p = os.path.join(train_dir, filename)
        img = cv2.imread(img_p)
        anno_p = os.path.join(train_dir, filename.replace('.jpg', '.txt'))
        h_img, w_img = img.shape[:-1]

        images.append(get_images(filename, h_img, w_img, img_id))
        with open(anno_p, 'r') as f:
            json_data = json.load(f)
            # print(json_data)
            for anno in json_data:
                box_point = np.array(anno['box_points'])
                box_attribute = anno['box_attribute']
                xmin = np.min(box_point[:, 0])
                xmax = np.max(box_point[:, 0])
                ymin = np.min(box_point[:, 1])
                ymax = np.max(box_point[:, 1])
                x1, y1 = box_point[0, :]
                x2, y2 = box_point[1, :]
                x3, y3 = box_point[2, :]
                x4, y4 = box_point[3, :]
                rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
                rbox = [int(rb) for rb in rbox]

                w_box = xmax - xmin
                h_box = ymax - ymin

                box_ann = [xmin, ymin, w_box, h_box]
                box_ann = [int(b) for b in box_ann]
                annotations.append(get_annotations(box_ann, img_id, annid, 'text', rbox))
                annid += 1
        img_id += 1

coco['images'] = images
categories.append(get_categories('text', 1))
coco['categories'] = categories
coco['annotations'] = annotations
instance = json.dumps(coco)
f = open('./train_all.json', 'w')
f.write(instance)
f.close()
