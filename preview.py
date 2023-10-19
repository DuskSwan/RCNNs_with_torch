# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:36:38 2023

@author: 24112
"""
import os
import numpy as np
# import torch
from PIL import Image

# from data_manager import PennFudanDataset

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

# 将数组转换回PIL图像
stretched_mask = Image.fromarray(mask_array*(255/3))
stretched_mask.show()
