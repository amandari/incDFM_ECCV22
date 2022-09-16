import os
import os.path
import pickle as pkl
import sys
import glob 
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import image
import fnmatch
from pathlib import Path
import PIL

import utils
import random





def loadExperiments_emnist(experiment_filepath):
    tasks_filepaths_train = glob.glob('%s/train/*.npy'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)
    tasks_filepaths_test = glob.glob('%s/test/*.npy'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    return tasks_filepaths_train, tasks_filepaths_test




def emnisttasks(num_per_class, num_tasks=25, first_task_num=None, shuffle=False):
    '''
    if num_classes is not divisible by num_per_class, then make extra classes go into first task. 
    '''
    num_classes = 26
    classes = np.arange(0,num_classes,1)
    rest= num_classes%num_per_class
    if shuffle:
        classes = classes[np.random.permutation(num_classes)]
    classes = list(classes)

    if first_task_num is not None:
        first_task=classes[:first_task_num]
    else:
        first_task = classes[:(num_per_class+rest)]

    print('first_task',first_task_num, first_task, num_per_class, rest)
    other_tasks = utils.list_to_2D(classes[len(first_task):],num_per_class)
    tasks = [first_task]+other_tasks

    return tasks[:num_tasks]
    



def load_data_dict(filepath):
    with open(filepath, 'rb') as f:
        return pkl.load(f)



def load_emnist(root, train=True, split='letters'):
    '''
    Meant for use in 3D pretrained architectures
    So tripled dimension --> RGB 
    '''

    root = os.path.expanduser(root)
    
    if train==True:
        filename = 'training_%s.pt'%split
        per_class_max=4800
    else:
        filename = 'test_%s.pt'%split
        per_class_max=800     


    data, labels = torch.load('%s/processed/%s'%(root, filename))

    
    data = data.type(torch.FloatTensor)
    labels = labels.type(torch.LongTensor)

    data = data/255.0
    labels = labels-1

    data = torch.stack([data,data,data], dim=1)

    data = torch.flip(data, [2])
    data = torch.rot90(data, -1, [2,3])

    # data = torch.unsqueeze(data, dim=1)

    # print('data', data.shape)
    # print('labels', labels.shape)


        
    return data, labels
            
            


class emnistTask():
    def __init__(self, dataroot, dset_name='emnist', train=True, tasklist='task_indices.npy', transform=None, returnIDX=False, preload=True):
        """ 
        dataset for each individual task
        """
        data, labels = load_emnist(dataroot, train=train)
        
        self.dset_name = dset_name
        

        # have option of loading multiple tasklists if teh case
        if isinstance(tasklist, list):
            self.indices_task_init=[]
            for l in tasklist:
                self.indices_task_init.append(np.load(l))
            self.indices_task_init = np.concatenate(self.indices_task_init)
        else:
            self.indices_task_init = np.load(tasklist)

        self.x = data 
        self.y = labels 
        self.transform = transform
        self.preload = preload 
        self.returnIDX = returnIDX
        
        # get appropriate task data
        self.x = self.x[self.indices_task_init, ...]
        self.y = self.y[self.indices_task_init]
        self.indices_task_init = np.arange(self.indices_task_init.shape[0])
        self.indices_task = np.copy(self.indices_task_init)

    def __len__(self):
        return self.indices_task.shape[0]

    def select_random_subset(self, random_num):

        inds_keep = np.random.permutation(np.arange(self.indices_task_init.shape[0]))[:random_num]

        self.indices_task = self.indices_task_init[inds_keep]
        
    def select_specific_subset(self, indices_select):
        
        self.indices_task = self.indices_task_init[indices_select]
        

        
    def __getitem__(self, idx):

        idx = self.indices_task[idx]

        im = self.x[idx,...]
        
        if self.transform is not None: ##input the desired tranform 

            im = self.transform(im)

        class_lbl = self.y[idx]

        if self.returnIDX:
            return im, class_lbl, class_lbl, idx
            
        return im, class_lbl, class_lbl





def emnistExperiments_w_holdout(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, holdout_percent=0.25, max_holdout=0.75, train=True, scenario='nc', shuffle=False):
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
    if train==True:
        partition='train'
    else:
        partition='test'
    
    # root = os.path.expanduser(root)
    dataroot = Path(dataroot)
    data, labels = load_emnist(dataroot, train=train)
    data = data.numpy()
    labels = labels.numpy()


    # --- Set up Task sequences (seq of labels)
    tasks_list=[]
    tasks_list_holdout = []
    for task in labels_per_task:
        tasks_list.append([])
        tasks_list_holdout.append([])
        for lb in task:
            tasks_list[-1].extend(list(np.where(labels==lb)[0]))
        tasks_list[-1] = np.array(tasks_list[-1])
        size_task = tasks_list[-1].shape[0]
        if holdout_percent>0:
            size_holdout = int(size_task*min(holdout_percent, max_holdout))
            inds_all = np.arange(size_task)
            indices_holdout = np.random.permutation(inds_all)[:size_holdout]
            indices_keep = np.array(list(set(list(inds_all)) - set(list(indices_holdout))))
            tasks_list_holdout[-1] = tasks_list[-1][indices_holdout]
            tasks_list[-1] = tasks_list[-1][indices_keep]
        

    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir,'emnist', experiment_name)
    utils.makedirectory(dest_dir)
    dest_dir_keep = dest_dir + '/%s'%(partition)
    utils.makedirectory(dest_dir_keep)
    if holdout_percent>0:
        dest_dir_holdout = dest_dir + '/%s'%('holdout')
        utils.makedirectory(dest_dir_holdout)

    # Create directory for experiment 
    task_filepaths=[]
    task_filepaths_holdout=[]
    for task in range(len(tasks_list)):
        task_filepaths.append("%s/%s_%s_task_%d.npy"%(dest_dir_keep, 'nc', partition, task))
        np.save(task_filepaths[-1], tasks_list[task])
        if holdout_percent>0:
            task_filepaths_holdout.append("%s/%s_%s_task_%d.npy"%(dest_dir_holdout, 'nc', 'holdout', task))
            np.save(task_filepaths_holdout[-1], tasks_list_holdout[task])


    return task_filepaths, task_filepaths_holdout
        


def emnistExperiments(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, train=True, scenario='nc'):
    """ 
    Args:
        root (string): Root directory of the dataset where images and paths file are stored
        outdir (string): Out directory to store experiment task files (txt sequences of objects)
        train (bool, optional): partition set. If train=True then it is train set. Else, test. 
        scenario (string, optional): What tye of CL learning regime. 'nc' stands for class incremental,\
             where each task contains disjoint class subsets
        run (int, optional):  with different runs shuffle order of tasks.
    """
    if train:
        partition='train'
    else:
        partition='test'
    
    # root = os.path.expanduser(root)
    dataroot = Path(dataroot)
    data, labels = load_emnist(dataroot, train=train)
    data = data.numpy()
    labels = labels.numpy()
    num_samples = data.shape[0]
    indices = np.arange(0,num_samples)

    # TODO shuffle for different runs 
    # labels_per_task = cifar10tasks(num_per_class, shuffle=shuffle)

    # --- Set up Task sequences (seq of labels)
    tasks_list=[]
    if scenario=='nc':
        for task in labels_per_task:
            tasks_list.append([])
            for lb in task:
                tasks_list[-1].extend(list(np.where(labels==lb)[0]))
        tasks_list[-1] = np.array(tasks_list[-1])
    elif scenario == 'joint_nc':
        tasks_list = [list(indices)]


    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir,'emnist', experiment_name)
    utils.makedirectory(dest_dir)
    dest_dir = dest_dir + '/%s'%(partition)
    utils.makedirectory(dest_dir)

    # Create directory for experiment 
    task_filepaths=[]
    for task in range(len(tasks_list)):
        task_filepaths.append("%s/%s_%s_task_%d.npy"%(dest_dir, scenario, partition, task))
        np.save(task_filepaths[-1], tasks_list[task])


    return task_filepaths
    #tasks_list
        

def emnistExperiments_w_holdout_w_validation(labels_per_task, dataroot, outdir, experiment_name, \
    target_ind=1, holdout_percent=0.2, validation_percent=0.1, train=True, scenario='nc'):
    """ 
    CORe50 Experiments setup
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
    # root = os.path.expanduser(root)
    dataroot = Path(dataroot)
    data, labels = load_emnist(dataroot, train=train)
    data = data.numpy()
    labels = labels.numpy()
    indices = np.arange(labels.shape[0])

        
    # --- Set up Task sequences (seq of labels)
    tasks_list=[]
    if scenario=='nc':
        for task in labels_per_task:
            tasks_list.append([])
            for lb in task:
                inds_task = np.where(labels==lb)[0]
                inds_task = indices[inds_task]
                tasks_list[-1].extend(list(inds_task))
            tasks_list[-1] = np.array(tasks_list[-1])
    elif scenario == 'joint_nc':
        tasks_list = [list(indices)]
        
        
    tasks_list_train=[]
    tasks_list_val=[]
    tasks_list_holdout=[]
    for task_ in tasks_list:
        # for each label subset
        num_samples_task = task_.shape[0]
        print('num_samples_task', num_samples_task)
        
        inds_shuf = np.random.permutation(task_)
        
        # divide the train/test split
        split_val = int(np.floor(validation_percent*num_samples_task))
        tasks_list_val.append(inds_shuf[:split_val])
        inds_train = inds_shuf[split_val:]

        split_holdout = int(np.floor(holdout_percent*num_samples_task))
        tasks_list_train.append(inds_train[split_holdout:])
        tasks_list_holdout.append(inds_train[:split_holdout])

        

    # --- save sequences to text files 
    task_filepaths_train = saveTasks(tasks_list_train, 'train', outdir, experiment_name, scenario='nc')
    task_filepaths_train_holdout = saveTasks(tasks_list_holdout, 'holdout', outdir, experiment_name, scenario='nc')
    task_filepaths_val = saveTasks(tasks_list_val, 'validation', outdir, experiment_name, scenario='nc')


    return task_filepaths_train, task_filepaths_train_holdout, task_filepaths_val
        



def saveTasks(tasks_list, partition, outdir, experiment_name, scenario='nc'):
    
    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir,'emnist', experiment_name)
    utils.makedirectory(dest_dir)
    dest_dir = dest_dir + '/%s'%(partition)
    utils.makedirectory(dest_dir)

    # Create directory for experiment 
    task_filepaths=[]
    for task in range(len(tasks_list)):
        task_filepaths.append("%s/%s_%s_task_%d.npy"%(dest_dir, scenario, partition, task))
        # print('labels', labels[tasks_list[task]])
        np.save(task_filepaths[-1], tasks_list[task])
        
    return task_filepaths