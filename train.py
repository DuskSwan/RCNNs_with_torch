# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:39:39 2023

@author: 24112
"""

import os
import torch
from PIL import Image

from packs.engine import train_one_epoch, evaluate
from packs import utils
from packs import transforms as T

from data_manager import PennFudanDataset
from model_loader import get_model_instance_segmentation

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor()) # PIL image to tensor
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5)) # flip img 
    return T.Compose(transforms)

def test(img,model,device):
    model.eval()
    with torch._no_grad(): prediction = model([img.to(device)])
    Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy) # 原图
    Image.fromarray(prediction[0]['masks'][0,0].mul(255).byte().cpu().numpy) # mask
 
def main(workdir):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset_cont = os.path.join(workdir,'data','PennFudanPed')
    dataset = PennFudanDataset(dataset_cont, get_transform(train=True))
    dataset_test = PennFudanDataset(dataset_cont, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    pre_model_sav_path = os.path.join(workdir,'premodel')
    model = get_model_instance_segmentation(num_classes, pre_model_sav_path)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler, lr becomes 0.1 times every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    
    img = dataset_test[0]
    test(img,model,device)
    
   
if __name__ == "__main__":
    curdir = os.getcwd() #当前工作目录
    main(curdir)
