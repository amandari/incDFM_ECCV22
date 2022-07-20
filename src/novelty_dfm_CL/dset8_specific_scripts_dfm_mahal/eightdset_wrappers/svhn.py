from __future__ import print_function
import os
import os.path
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import gzip
from torch.utils.data import Dataset
import sys
import pickle
import random


        
class MY_SVHN(Dataset):
    def __init__(self, img_path, tasklist=None, data_transforms=None, target_transform=None, split='train', balance=False):
       
        
        
        self.train_map = {'train':True, 'val':False}
        
        self.balance=balance
        
        self.max_labels=10

        self.images, self.img_label = load_SVHN(img_path, self.train_map[split], balance=self.balance)
        
        # subset 
        if tasklist is not None:
            tasklist = np.load(tasklist)
            self.images = self.images[tasklist,...]
            self.img_label = self.img_label[tasklist]
                
        self.data_transforms = data_transforms
        self.target_transform=target_transform
        

    def __len__(self):
        return self.img_label.shape[0]

    def __getitem__(self, item):
        
        img = self.images[item,:]
        label = self.img_label[item]
        

        if self.data_transforms is not None:
            try:

                img = self.data_transforms(img)
                
            except:

                print("Cannot transform image")
                
        if self.target_transform is not None:
            
            label = self.target_transform(label)
                

        return img, label


def load_SVHN(root, train=True, balance=True):

    root = os.path.expanduser(root)
    
    if train==True:
        filename = "train_32x32.mat"
    else:
        filename = "test_32x32.mat"

        
    import scipy.io as sio
    # reading(loading) mat file as array
    loaded_mat = sio.loadmat(os.path.join(root, filename))

    data = loaded_mat['X']
    # loading from the .mat file gives an np array of type np.uint8
    # converting to np.int64, so that we have a LongTensor after
    # the conversion from the numpy array
    # the squeeze is needed to obtain a 1D tensor
    labels = loaded_mat['y'].astype(np.int64).squeeze()

    # the svhn dataset assigns the class label "10" to the digit 0
    # this makes it inconsistent with several loss functions
    # which expect the class labels to be in the range [0, C-1]
    np.place(labels, labels == 10, 0)
    data = np.transpose(data, (3, 2, 0, 1))

    if train==True:
        max_ind_b = 4948
    else:
        max_ind_b = 1595
    
    
    if balance==True:
        
        inds_b = []
        random.seed(999)

        for i in range(10):

            arr = np.where(labels==i)[0]
            np.random.shuffle(arr)
            inds = arr[:max_ind_b]
            inds_b.extend(list(inds))

        inds_b = np.array(inds_b)
        inds_b = inds_b.astype(int)

        np.random.shuffle(inds_b)

        data = data[inds_b,...]
        labels = labels[inds_b]
        
        
    data = torch.from_numpy(data).type(torch.FloatTensor)
    # y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)
        
    return data, labels