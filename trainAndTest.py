# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:39:39 2023

@author: 24112
"""

import os
import sys,datetime
import torch
from PIL import Image

from packs.engine import train_one_epoch, evaluate
from packs import utils
from packs import transforms as T

from data_manager import PennFudanDataset, hangcai03Dataset
from model_loader import get_model_instance_segmentation

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor()) # PIL image to tensor
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5)) # flip img 
    return T.Compose(transforms)

def test_model(dataset,model,device,save_path=r'./res'):
    model.eval()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for idx, (img, target) in enumerate(dataset):
        print(f'dealing idx:{idx}')
        originImg =  Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
        trueMask = Image.fromarray(target['masks'][0].mul(255).byte().cpu().numpy())
        with torch.no_grad(): prediction = model( [img.to(device)] )
        predMask = Image.fromarray(prediction[0]['masks'][0,0].mul(255).byte().cpu().numpy())
        originImg.save(os.path.join(save_path,f'origin_{idx}.jpg'))
        trueMask.save(os.path.join(save_path,f'trueMask_{idx}.jpg'))
        predMask.save(os.path.join(save_path,f'predMask_{idx}.jpg'))
        print(f'dealing idx:{idx} done')

def train(workdir,device,
          dataset,dataset_test,num_classes=2):
    
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

    # let's train it for some epochs
    num_epochs = 10

    print('start training')
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    model_dir = os.path.join(workdir, 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_save_name = save_time + 'model.pth'
    weight_save_name = save_time + 'model_weights.pth'
    torch.save(model, os.path.join(model_dir, model_save_name))
    torch.save(model.state_dict(), os.path.join(model_dir, weight_save_name))

    return (model,model_dir,save_time)

def log_decorator(func):
    def wrapper(*args, **kwargs):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = f'.\\log\\training_log_{current_time}.txt'
        
        original_stdout = sys.stdout # 保存标准输出流到一个文件
        log_file = open(log_file_path, 'w')
        sys.stdout = log_file
        func(*args, **kwargs)

        sys.stdout = original_stdout # 恢复标准输出流
        log_file.close() # 关闭 log_file
    return wrapper

def model_load(name):
    cont = r'.\model'
    model = torch.load(os.path.join(cont,name))
    return model

@log_decorator
def main(workdir):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset_cont = os.path.join(workdir,'data','PennFudanPed')
    dataset = PennFudanDataset(dataset_cont, get_transform(train=True))
    dataset_test = PennFudanDataset(dataset_cont, get_transform(train=False))

    model,_,_ = train(workdir,device,dataset,dataset_test,num_classes)

    save_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    res_save_path = r'./res/{}'.format(save_time)
    test_model(dataset_test,model,device,res_save_path)
   
if __name__ == "__main__":
    curdir = os.getcwd()  # 当前工作目录
    print(curdir,'project start')
    
    main(curdir)

