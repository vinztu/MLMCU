"""
IntelNet dataset
"""
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms, datasets
from torchvision.io import read_image

import ai8x

import os
import pandas as pd

import matplotlib.pyplot as plt


"""
Dataloader function
"""
def Intel_get_datasets(data, load_train=False, load_test=False):
    
    traindir = "./data/intel/seg_train"
    testdir = "./data/intel/seg_test"
    validation_split = 0.2
    BATCH_SIZE = 32
    image_size = (64,64)
   
    (data_dir, args) = data
    
    
    if load_train:
        train_transform = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                                              #transforms.RandomAffine(degrees=30, translate=(0.5, 0.5), scale=(0.5,1.5), fill=0),
                                              transforms.RandomVerticalFlip(0.5),
                                              transforms.Resize(image_size),
                                              transforms.ToTensor(),
                                              ai8x.normalize(args=args)])
      
    
        train_data = torchvision.datasets.ImageFolder(traindir, transform = train_transform)
        trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size = BATCH_SIZE)
        trainloader = trainloader.dataset
    else:
        train_data = None
    
    
    if load_test:
        test_transform = transforms.Compose([transforms.Resize(image_size),
                                         transforms.ToTensor(),
                                         ai8x.normalize(args=args)])
        
        test_data = torchvision.datasets.ImageFolder(testdir, transform = test_transform)
        testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size = BATCH_SIZE)
        testloader = testloader.dataset
    else:
        test_data = None
    
    

    return trainloader, testloader


"""
Dataset description
"""
datasets = [
    {
        'name': 'intel',
        'input': (3, 64, 64),
        'output': list(map(str, range(6))),
        'loader': Intel_get_datasets,
    }
]
