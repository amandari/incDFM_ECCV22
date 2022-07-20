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





def loadExperiments_svhn(experiment_filepath):
    tasks_filepaths_train = glob.glob('%s/train/*.npy'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)
    tasks_filepaths_test = glob.glob('%s/test/*.npy'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    return tasks_filepaths_train, tasks_filepaths_test




def svhntasks(num_per_class, num_tasks=9, first_task_num=None, shuffle=False):
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

    print('first_task',first_task_num, first_task, num_per_class, rest)
    other_tasks = utils.list_to_2D(classes[len(first_task):],num_per_class)
    tasks = [first_task]+other_tasks

    return tasks[:num_tasks]
    



def load_data_dict(filepath):
    with open(filepath, 'rb') as f:
        return pkl.load(f)



# def load_svhn(root, train=True):

#     root = os.path.expanduser(root)
    
#     if train==True:
#         filename = "train_32x32.pkl"
#     else:
#         filename = "test_32x32.pkl"

        
#     # reading(loading) mat file as array
#     loaded_mat = load_data_dict('%s/%s'%(root, filename))

#     data = loaded_mat['data']
#     # loading from the .mat file gives an np array of type np.uint8
#     # converting to np.int64, so that we have a LongTensor after
#     # the conversion from the numpy array
#     # the squeeze is needed to obtain a 1D tensor
#     labels = loaded_mat['labels'].astype(np.int64).squeeze()

#     # the svhn dataset assigns the class label "10" to the digit 0
#     # this makes it inconsistent with several loss functions
#     # which expect the class labels to be in the range [0, C-1]

        
#     data = torch.from_numpy(data).type(torch.FloatTensor)
#     labels = torch.from_numpy(labels).type(torch.LongTensor)


#     # print('data', data.shape)
#     # print('labels', labels.shape)
#     # print(labels)

        
#     return data, labels
            
     
def load_svhn(root, train=True, balance=False):

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


class svhnTask():
    def __init__(self, dataroot, dset_name='svhn', train=True, tasklist='task_indices.npy', transform=None, returnIDX=False, preload=True):
        """ 
        dataset for each individual task
        """
        data, labels = load_svhn(dataroot, train=train, balance=False)
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





def svhnExperiments_w_holdout(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, holdout_percent=0.25, max_holdout=0.75, train=True, scenario='nc', shuffle=False):
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
    data, labels = load_svhn(dataroot, train=train, balance=False)
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
    dest_dir = '%s/%s/%s'%(outdir,'svhn', experiment_name)
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
        


def svhnExperiments(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, train=True, scenario='nc', shuffle=False):
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
    data, labels = load_svhn(dataroot, train=train, balance=False)
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
    dest_dir = '%s/%s/%s'%(outdir,'svhn', experiment_name)
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
        

def svhnExperiments_w_holdout_w_validation(labels_per_task, dataroot, outdir, experiment_name, \
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
    data, labels = load_svhn(dataroot, train=train, balance=False)
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
    dest_dir = '%s/%s/%s'%(outdir,'svhn', experiment_name)
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