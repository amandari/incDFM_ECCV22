from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms
import numpy as np
import time
import os
from torch.utils.data import Dataset
import sys
from PIL import Image
from matplotlib import pyplot as plt

def default_loader(path):
    try:
        img = Image.open(path)
        #return img
        #print(img)
        return img.convert('RGB')
        #return img
    except:
        print("Cannot read image: {}".format(path))

class MY_FLOWERS(Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, target_transform=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            # self.img_name = [os.path.join(img_path, line.strip().split(',')[0]) for line in lines]
            # self.img_label = [int(line.strip().split(',')[-1]) for line in lines]
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        
        self.target_transform = target_transform

        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        
        #plt.imshow(img)
        #plt.show()
        if self.data_transforms is not None:
            try:
                #print("I enter here")
                #print(img)
                img = self.data_transforms(img)
            except:
                #e = sys.exc_info()[0]
                #print("exception",e)
                print("Cannot transform image: {}".format(img_name))
                
        #print("the shape of img is".img.shape)
#         print(img.shape)

        if self.target_transform is not None:
            
            label = self.target_transform(label)
        
        return img, label