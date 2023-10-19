# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:59:57 2023

@author: 24112
"""
import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model_instance_segmentation(num_classes, download_path = None):
    
    # change the content where models will be download in
    if(download_path): os.environ['TORCH_HOME'] = download_path
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

if __name__ == '__main__':
    curdir = os.getcwd() #获得当前工作目录
    model = get_model_instance_segmentation(2, os.path.join(curdir,'pre_model'))
    print(model)