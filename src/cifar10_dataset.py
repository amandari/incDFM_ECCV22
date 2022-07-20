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





def loadExperiments_cifar10(experiment_filepath):
    tasks_filepaths_train = glob.glob('%s/train/*.npy'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)
    tasks_filepaths_test = glob.glob('%s/test/*.npy'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    return tasks_filepaths_train, tasks_filepaths_test




def cifar10tasks(num_per_class, num_tasks=9, first_task_num=None, shuffle=False):
    '''
    if 10 is not divisible by num_per_class, then make extra classes go into first task. 
    '''
    num_classes = 10
    classes = np.arange(0,num_classes,1)
    rest= 10%num_per_class
    if shuffle:
        classes = classes[np.random.permutation(10)]
    classes = list(classes)

    if first_task_num is not None:
        first_task=classes[:first_task_num]
    else:
        first_task = classes[:(num_per_class+rest)]

    print('first_task', first_task, num_per_class, rest)
    other_tasks = utils.list_to_2D(classes[len(first_task):],num_per_class)
    tasks = [first_task]+other_tasks

    return tasks[:num_tasks]
    


def load_cifar(root, train=True):
    
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    
    root = os.path.expanduser(root)

    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list

    data = []
    labels = []

    # now load the picked numpy arrays
    for file_name, checksum in downloaded_list:
        file_path = os.path.join(root, file_name)
        with open(file_path, 'rb') as f:

            entry = pkl.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                labels.extend(entry['labels'])
            else:
                labels.extend(entry['fine_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    data = np.array(data)
    labels = np.array(labels)
    data = np.transpose(data, (0, 3, 1, 2))
    data = torch.from_numpy(data)
    # y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    
    return data, labels
            
        


class cifar10Task():
    def __init__(self, dataroot, dset_name='cifar10', train=True, tasklist='task_indices.npy', transform=None, returnIDX=False, preload=True):
        """ 
        dataset for each individual task
        """
        data, labels = load_cifar(dataroot, train=train)
        
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
        

    def __getitem__(self, idx, apply_transform=True):

        idx = self.indices_task[idx]

        im = self.x[idx,...]
        
        if apply_transform == True:
            if self.transform is not None: ##input the desired tranform 
                im = self.transform(im)

        class_lbl = self.y[idx]

        if self.returnIDX:
            return im, class_lbl, class_lbl, idx
            
        return im, class_lbl, class_lbl



def cifar10Experiments(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, train=True, scenario='nc', shuffle=False):
    """ 
    CORe50 Experiments setup
    Args:
        root (string): Root directory of the dataset where images and paths file are stored
        outdir (string): Out directory to store experiment task files (txt sequences of objects)
        train (bool, optional): partition set. If train=True then it is train set. Else, test. 
        scenario (string, optional): What tye of CL learning regime. 'nc' stands for class incremental,\
             where each task contains disjoint class subsets
        run (int, optional):  with different runs shuffle order of tasks.
    """
    if train==True:
        partition='train'
    else:
        partition='test'
    
    # root = os.path.expanduser(root)
    dataroot = Path(dataroot)
    data, labels = load_cifar(dataroot, train=train)
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
    dest_dir = '%s/%s/%s'%(outdir,'cifar10', experiment_name)
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
    #tasks_list




def cifar10Experiments_w_holdout(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, holdout_percent=0.25, max_holdout=0.75, train=True, scenario='nc', shuffle=False):
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
    if train==True:
        partition='train'
    else:
        partition='test'
    
    # root = os.path.expanduser(root)
    dataroot = Path(dataroot)
    data, labels = load_cifar(dataroot, train=train)
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
    dest_dir = '%s/%s/%s'%(outdir,'cifar10', experiment_name)
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
        





def cifar10Experiments_w_holdout_w_validation(labels_per_task, dataroot, outdir, experiment_name, \
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
    data, labels = load_cifar(dataroot, train=train)
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
    dest_dir = '%s/%s/%s'%(outdir,'cifar10', experiment_name)
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