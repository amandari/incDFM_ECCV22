

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as TF
from collections import Counter
from tqdm import tqdm 
from torchvision import datasets, models, transforms
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import tforms
import utils
from eightdset_wrappers.aircraft import MY_AIRCRAFT
from eightdset_wrappers.birds import MY_BIRDS
from eightdset_wrappers.cars import MY_CARS
from eightdset_wrappers.voc import MY_VOC
from eightdset_wrappers.char import MY_CHAR
from eightdset_wrappers.flowers import MY_FLOWERS
from eightdset_wrappers.scenes import MY_SCENES
from eightdset_wrappers.svhn import MY_SVHN

import copy


class eightdsetTask():
    def __init__(self, dataroot, dset_name, split='train', tasklist='task_indices.txt', transform=None, \
        returnIDX=False,  preload=False):
        '''
        dataroot - for 8dset /lab/arios/ProjIntel/incDFM/data/8dset
        dset_name - among the 8 datasets (flowers, aircrafts, birds, cars, char, scenes, voc, svhn)
        '''

        self.transform = transform
        
        self.dset_name = dset_name
        
        self.individual_wrappers={'flowers':MY_FLOWERS, 'aircraft':MY_AIRCRAFT, 'birds': MY_BIRDS, \
            'voc':MY_VOC, 'cars': MY_CARS, 'svhn': MY_SVHN, 'chars':MY_CHAR, 'scenes':MY_SCENES}
        
        self.order_tasks = {'flowers':0, 'scenes':1, 'birds':2, 'cars':3, 'aircraft':4, 'voc':5,  'chars':6, 'svhn':7}
        
        self.task_lb = self.order_tasks[self.dset_name]
        
        if dset_name !='svhn':
            self.intern_dset = self.individual_wrappers[dset_name](img_path='%s/%s'%(dataroot, dset_name),
                                        txt_path=tasklist,
                                        data_transforms=transform)
        else:
            self.intern_dset = self.individual_wrappers[dset_name]('%s/%s'%(dataroot, dset_name),
                                        tasklist=tasklist,
                                        data_transforms=transform,
                                        split=split)
            
        self.returnIDX = returnIDX
        
        self.indices_task_init = np.arange(self.intern_dset.__len__())
        
        self.indices_task = copy.deepcopy(self.indices_task_init)

            
    def __len__(self):
        return self.indices_task.shape[0]

    def select_random_subset(self, random_num):

        inds_keep = np.random.permutation(np.arange(self.indices_task_init.shape[0]))[:random_num]

        self.indices_task = self.indices_task_init[inds_keep]
        
    def select_specific_subset(self, indices_select):
        
        self.indices_task = self.indices_task_init[indices_select]
        
    def __getitem__(self, idx):
        
        idx = self.indices_task[idx]

        im, class_lbl = self.intern_dset.__getitem__(idx)
        
        if self.returnIDX:
            return im, class_lbl, self.task_lb, idx
            
        return im, class_lbl, self.task_lb





def eightdset_Experiments_w_holdout_w_validation_trainset(dataroot, outdir, experiment_name, dset_name='8dset',\
    holdout_percent=0.2, validation_percent=0.1):
    """ 
    Only do holdout for Train
    Args:
        holdout_percent (float): percent of train data to leave out for later
        max_holdout (float): maximum holdout_percent allowed. Usually not changed 
        root (string): Root directory of the dataset where images and paths file are stored
        outdir (string): Out directory to store experiment task files (txt sequences of objects)
        train (bool, optional): partition set. If train=True then it is train set. Else, test. 
        scenario (string, optional): What tye of CL learning regime. 'nc' stands for class incremental,\
             where each task contains disjoint class subsets
    """
    
    list_tasks = ['flowers', 'scenes', 'birds', 'cars', 'aircraft', 'voc', 'chars', 'svhn']
    individual_wrappers={'flowers':MY_FLOWERS, 'aircraft':MY_AIRCRAFT, 'birds': MY_BIRDS, \
            'voc':MY_VOC, 'cars': MY_CARS, 'svhn': MY_SVHN, 'chars':MY_CHAR, 'scenes':MY_SCENES}
    
    tasklists_train = ['%s/%s/labels/train.txt'%(dataroot,d) for d in list_tasks[:-1]]+['train']

    dsets = []
    for i, dname in enumerate(list_tasks):
        if dname =='svhn':
            dataset = MY_SVHN(img_path='/lab/arios/ProjIntel/incDFM/data/svhn/',
                                    split='train')
        else:
            dataset = individual_wrappers[dname](img_path='%s/%s'%(dataroot, dname),
                                            txt_path=tasklists_train[i],
                                            dataset='train')
        dsets.append(dataset)
        
                
    # --- Set up Task sequences 8dset
    tasks_list_train=[]
    tasks_list_val=[]
    tasks_list_holdout=[]
    for t, task in enumerate(list_tasks):
        
        labels = np.array(dsets[t].img_label)
        print('Dset %s - Num data %d Num labels %d'%(task, labels.shape[0], np.unique(labels).shape[0]))
        num_samples_task = labels.shape[0]
        indices_task = np.arange(num_samples_task)

        # for each label subset
        print('num_samples_task', num_samples_task)
        
        inds_shuf = np.random.permutation(indices_task)
        
        # divide the train/val/holdout split
        split_val = int(np.floor(validation_percent*num_samples_task))
        tasks_list_val.append(inds_shuf[:split_val])
        inds_train = inds_shuf[split_val:]


        split_holdout = int(np.floor(holdout_percent*num_samples_task))
        tasks_list_train.append(inds_train[split_holdout:])
        tasks_list_holdout.append(inds_train[:split_holdout])

        print(tasks_list_train[-1].shape, tasks_list_val[-1].shape, tasks_list_holdout[-1].shape)

    # sys.exit()

    # --- save sequences to text files 
    task_filepaths_train = saveTasks_to_txt(tasks_list_train, tasklists_train, dset_name, 'train', outdir, experiment_name, scenario='nc')
    task_filepaths_train_holdout = saveTasks_to_txt(tasks_list_holdout, tasklists_train, dset_name, 'holdout', outdir, experiment_name, scenario='nc')
    task_filepaths_val = saveTasks_to_txt(tasks_list_val, tasklists_train, dset_name, 'validation', outdir, experiment_name, scenario='nc')


    return task_filepaths_train, task_filepaths_train_holdout, task_filepaths_val
        



def eightdset_Experiments_testset(dataroot, outdir, experiment_name, dset_name='8dset'):
    """ 
    Only do holdout for Train
    Args:
        holdout_percent (float): percent of train data to leave out for later
        max_holdout (float): maximum holdout_percent allowed. Usually not changed 
        root (string): Root directory of the dataset where images and paths file are stored
        outdir (string): Out directory to store experiment task files (txt sequences of objects)
        train (bool, optional): partition set. If train=True then it is train set. Else, test. 
        scenario (string, optional): What tye of CL learning regime. 'nc' stands for class incremental,\
             where each task contains disjoint class subsets
    """
    
    list_tasks = ['flowers', 'scenes', 'birds', 'cars', 'aircraft', 'voc', 'chars', 'svhn']
    individual_wrappers={'flowers':MY_FLOWERS, 'aircraft':MY_AIRCRAFT, 'birds': MY_BIRDS, \
            'voc':MY_VOC, 'cars': MY_CARS, 'svhn': MY_SVHN, 'chars':MY_CHAR, 'scenes':MY_SCENES}
    
    tasklists_train = ['%s/%s/labels/val.txt'%(dataroot,d) for d in list_tasks[:-1]]+['test']

    dsets = []
    for i, dname in enumerate(list_tasks):
        if dname =='svhn':
            dataset = MY_SVHN(img_path='/lab/arios/ProjIntel/incDFM/data/svhn/',
                                    split='val')
        else:
            dataset = individual_wrappers[dname](img_path='%s/%s'%(dataroot, dname),
                                            txt_path=tasklists_train[i],
                                            dataset='val')
        dsets.append(dataset)
        
                
    # --- Set up Task sequences 8dset
    tasks_list_test=[]
    for t, task in enumerate(list_tasks):
        
        labels = np.array(dsets[t].img_label)
        print('Dset %s - Num data %d Num labels %d'%(task, labels.shape[0], np.unique(labels).shape[0]))
        num_samples_task = labels.shape[0]
        indices_task = np.arange(num_samples_task)

        # for each label subset
        print('num_samples_task', num_samples_task)

        tasks_list_test.append(indices_task)
        print(tasks_list_test[-1].shape)

    # --- save sequences to text files 
    task_filepaths_test = saveTasks_to_txt(tasks_list_test, tasklists_train, dset_name, 'test', outdir, experiment_name, scenario='nc')

    return task_filepaths_test
        



def saveTasks_to_txt(tasklists_subset_indices, tasklists_orig, dset_name, partition, outdir, experiment_name, scenario='nc'):
    
    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir, dset_name, experiment_name)
    utils.makedirectory(dest_dir)
    dest_dir = dest_dir + '/%s'%(partition)
    utils.makedirectory(dest_dir)

    # Create directory for experiment 
    task_filepaths=[]
    for task in range(len(tasklists_orig)):
        subset_indices = tasklists_subset_indices[task]
        if task<7:
            task_filepaths.append("%s/%s_%s_task_%d.txt"%(dest_dir, scenario, partition, task))
            write_to_file_subset_8dset(tasklists_orig[task], subset_indices, task_filepaths[-1])
        else:
            task_filepaths.append("%s/%s_%s_task_%d.npy"%(dest_dir, scenario, partition, task))
            np.save(task_filepaths[-1], subset_indices)
        
    return task_filepaths



def write_to_file_subset_8dset(original_txt, subset_indices, new_txt):
    with open(original_txt, "r") as file_input:
        with open(new_txt, "w") as output: 
            for i, line in enumerate(file_input):
                if i in subset_indices:
                    output.write(line)
        