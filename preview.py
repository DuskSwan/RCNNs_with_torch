# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:36:38 2023

@author: 24112
"""
import os
import numpy as np
import torch
from PIL import Image

'''
#%% 查看掩码

root = r'.\data\PennFudanPed'
idx = 88

imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages")))) # img names
masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

img_path = os.path.join(root, "PNGImages", imgs[idx])
mask_path = os.path.join(root, "PedMasks", masks[idx])
img = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path)

img.show()
# mask.show()

# np.unique(mask)

# 将灰度图转换为NumPy数组
mask_array = np.array(mask)

# 使用NumPy的函数来找到非零像素的索引
# non_zero_indices = np.argwhere(mask_array != 0)

# non_zero_indices是一个包含非零像素位置的NumPy数组
# 每行包含一个非零像素的行列号，例如 (row, col)

# 打印非零像素的索引
# for index in non_zero_indices:
#     row, col = index
#     print(f"非零像素位置：行 {row}, 列 {col}")

# 将掩码灰度拉伸
stretched_mask = Image.fromarray(mask_array*(255/np.max(mask_array)))
stretched_mask.show()

#%% 查看target类型

# mask = Image.open(mask_path)

mask = np.array(mask)
obj_ids = np.unique(mask)
obj_ids = obj_ids[1:]

masks = mask == obj_ids[:, None, None]

num_objs = len(obj_ids)
boxes = []
for i in range(num_objs):
    pos = np.where(masks[i])
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    boxes.append([xmin, ymin, xmax, ymax])

boxes = torch.as_tensor(boxes, dtype=torch.float32)

labels = torch.ones((num_objs,), dtype=torch.int64)
masks = torch.as_tensor(masks, dtype=torch.uint8)

image_id = torch.tensor([idx])
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
'''

#%% 查看model结构

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

os.environ['TORCH_HOME'] = r'.\premodel'
num_classes = 2

def add_txt_in_file(txt, file_path):
    with open(file_path, 'a') as f:
        f.write(txt)
        f.write('\n')
fpath = r'res.txt'

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

with open(fpath, 'w') as f:
    f.truncate(0)
txt = str(model)
add_txt_in_file(txt, fpath)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

txt = str(model.roi_heads.box_predictor) 
add_txt_in_file(txt, fpath)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)
txt = str(model.roi_heads.mask_predictor)
add_txt_in_file(txt, fpath)