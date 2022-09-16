import os
import os.path
import pickle as pkl
import sys
import glob 
import numpy as np
from numpy.core.numeric import indices
import torch
from torch.utils.data import Dataset

from matplotlib import image
import fnmatch
from pathlib import Path
import PIL
from torchvision import datasets, transforms
import utils
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import tforms


def loadExperiments_inaturalist(experiment_filepath):
    tasks_filepaths_train = glob.glob('%s/train/*.npy'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)
    tasks_filepaths_test = glob.glob('%s/test/*.npy'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    return tasks_filepaths_train, tasks_filepaths_test




def INATURALISTtasks(num_per_class, num_classes, num_tasks=5, first_task_num=None, shuffle=False):
    '''
    6 super classes - 2019 version 
    '''
    classes = np.arange(0,num_classes,1)
    rest= num_classes%num_per_class
    if shuffle:
        classes = classes[np.random.permutation(num_classes)]
    classes = list(classes)

    if first_task_num is not None:
        first_task=classes[:first_task_num]
    else:
        first_task = classes[:(num_per_class+rest)]

    print('first_task', first_task, num_per_class, rest)
    other_tasks = utils.list_to_2D(classes[len(first_task):],num_per_class)
    tasks = [first_task]+other_tasks

    return tasks[:num_tasks]

    
    
def exclude_problem_indices_inaturalist_2019(dataroot, indices_save_filepath, transform=None):
    
    dataset = Dataset.inaturalist(dataroot, version='2019', target_type=['super'], \
                                transform=transform, target_transform=None, download=False)
    
    indices_all = np.arange(dataset.__len__())
    exclude_inds = []
    
    for i, ind in enumerate(indices_all):
        try:
            _ = dataset.__getitem__(ind)
        except RuntimeError:
            exclude_inds.append(i)
    
    if len(exclude_inds)>0:
        keep = np.setdiff1d(np.arange(indices_all.shape[0]), np.array(exclude_inds))
        indices_all = indices_all[keep]
        
    np.save(indices_save_filepath, indices_all)
    
    return indices_all
    
    
    
def exclude_inds_2021(labels, indices_save_filepath):
    exclude_lbls = [0,9,10,11]
    
    indices_all = np.arange(labels.shape[0])
    exclude_inds = []
    for lb in exclude_lbls:
        inds = np.where(labels==lb)[0].astype(int)
        exclude_inds.extend(inds.tolist())
    
    indices_keep = np.setdiff1d(indices_all, np.array(exclude_inds))
    
    np.save(indices_save_filepath, indices_keep)

    return indices_keep
    
    
    

class WrapNaturalist(datasets.INaturalist):
    
    def __init__(self, root: str, version: str = "2021_train", target_type: Union[List[str], str] = "full", \
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, version, target_type, transform, target_transform, download)
        
        self.target_transform = target_transform
        self.version = version
        
        if '19' in self.version:
            indices_path = root+'/'+'2019_valid_indices.npy'
        elif '21' in self.version:
            self.map_target_2021 = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,12:8}
            self.target_transform =  transforms.Lambda(lambda x: self.map_target_2021[x])
            indices_path = root+'/'+'2021_valid_indices.npy'
            
        self.indices_path = indices_path
        # print('indices_path', indices_path)
        self.indices_all = np.load(indices_path)
            
    def __len__(self):
        return self.indices_all.shape[0]
        
    def __getindex__(self, idx):
        
        idx = self.indices_all[idx]
        
        cat_id, _ = self.index[idx]

        target = []
        # target_name = []
        for t in self.target_type:
            if t == "full":
                target.append(cat_id)
            else:
                target.append(self.categories_map[cat_id][t])
                # if '21' in self.version:
                #     target_name.append(self.all_categories[cat_id].split('_')[2])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        
        return target
    
    
    def __getitem_wrap__(self, idx):
        
        idx = self.indices_all[idx]
        
        im, target = self.__getitem__(idx)
        
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
            
        return im, target
    
    
    def __getlabels_2019__(self):
        
        self.labels = np.zeros((self.__len__(),)).astype(int)
        for i in range(self.__len__()):
            target = self.__getindex__(i)            
            self.labels[i]=target
            
        return self.labels
            
            
    def __getlabels_2021__(self):
        self.labels = np.zeros((self.__len__(),)).astype(int)
        for i in range(self.__len__()):
            target = self.__getindex__(i)            
            self.labels[i]=target
            # self.labels_map.append((target, name))
                    
        return self.labels
    
    
 
    




class inaturalistTask():
    def __init__(self, dataroot, dset_name='inaturalist19', tasklist='task_indices.npy', transform=None, \
        returnIDX=False, train=True, preload=False):
        """ 
        dataset for each individual task
        """
        self.preload = preload
        self.transform = transform
        

        self.transform_preload = transform
        self.dset_name = dset_name
        
        if self.dset_name == 'inaturalist19':
            self.dataset = WrapNaturalist(dataroot, version='2019', target_type=['super'], \
                                    transform=transform, download=False)
            _ = self.dataset.__getlabels_2019__()
        elif self.dset_name == 'inaturalist21':
            self.dataset = WrapNaturalist(dataroot, version='2021_train_mini', target_type=['phylum'], \
                                    transform=transform, download=False)
            _ = self.dataset.__getlabels_2021__()

        # have option of loading multiple tasklists if the case
        if isinstance(tasklist, list):
            self.indices_task_init=[]
            for l in tasklist:
                self.indices_task_init.append(np.load(l))
            self.indices_task_init = np.concatenate(self.indices_task_init)
        else:
            self.indices_task_init = np.load(tasklist)
            

        self.returnIDX = returnIDX
        
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
        # print('idx', idx)
        im, class_lbl = self.dataset.__getitem_wrap__(idx)

        # assert self.dataset.labels[idx] == class_lbl
        
        if self.returnIDX:
            return im, class_lbl, class_lbl, idx
            
        return im, class_lbl, class_lbl








def INATURALISTExperiments(labels_per_task, dataroot, outdir, experiment_name, dset_name='inaturalist19',\
                           scenario='nc', test_split=0.2, shuffle=False, equalize_labels=False, clip_labels=False, clip_max=50000):
    """ 
    inaturalist Experiments setup
    Args:
        root (string): Root directory of the dataset where images and paths file are stored
        outdir (string): Out directory to store experiment task files (txt sequences of objects)
        train (bool, optional): partition set. If train=True then it is train set. Else, test. 
        scenario (string, optional): What tye of CL learning regime. 'nc' stands for class incremental,\
             where each task contains disjoint class subsets
        run (int, optional):  with different runs shuffle order of tasks.
        
        equalize_labels: True if every label has to have same counts 
    """

    if dset_name == 'inaturalist19':
        dataset = WrapNaturalist(dataroot, version='2019', target_type=['super'],download=False)
        labels = dataset.__getlabels_2019__()
    elif dset_name == 'inaturalist21':
        dataset = WrapNaturalist(dataroot, version='2021_train_mini', target_type=['phylum'], download=False)
        labels = dataset.__getlabels_2021__()
    

    num_samples = dataset.__len__()
    indices = np.arange(0,num_samples)
    
    
    # labels = []
    # for pair in dataset.index:
    #     labels.append(dataset.categories_map[pair[0]]['super'])
    # labels = np.array(labels)
    
    if equalize_labels:
        indices = equilize_label_counts(labels)
        labels = labels[indices]
        num_samples = indices.shape[0]
        
    if clip_labels:
        indices = clip_top_labels(labels, max_count=clip_max)
        labels = labels[indices]
        num_samples = indices.shape[0]




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
    tasks_list_test=[]
    for task_ in tasks_list:
        num_samples_task = task_.shape[0]
        # divide the train/test split
        split = int(np.floor(test_split * num_samples_task))
        inds_shuf = np.random.permutation(task_)
        tasks_list_train.append(inds_shuf[split:])
        tasks_list_test.append(inds_shuf[:split])


    # save the files
    task_filepaths_train = saveTasks(tasks_list_train, dset_name, 'train', outdir, experiment_name, scenario='nc')
    task_filepaths_test = saveTasks(tasks_list_test, dset_name, 'test', outdir, experiment_name, scenario='nc')


    return task_filepaths_train, task_filepaths_test






def INATURALISTExperiments_w_holdout(labels_per_task, dataroot, outdir, experiment_name, dset_name='inaturalist19',\
    holdout_percent=0.2, test_split=0.2, scenario='nc', shuffle=False, equalize_labels=False, clip_labels=False, clip_max=50000):
    """ 
    inaturalist Experiments setup
    Only do holdout for Train
    Args:
        holdout_percent (float): percent of train data to leave out for later
        max_holdout (float): maximum holdout_percent allowed. Usually not changed 
        root (string): Root directory of the dataset where images and paths file are stored
        outdir (string): Out directory to store experiment task files (txt sequences of objects)
        train (bool, optional): partition set. If train=True then it is train set. Else, test. 
        scenario (string, optional): What tye of CL learning regime. 'nc' stands for class incremental,\
             where each task contains disjoint class subsets
             
        equalize_labels: True if every label has to have same counts 
    """
    
    if dset_name == 'inaturalist19':
        dataset = WrapNaturalist(dataroot, version='2019', target_type=['super'],download=False)
        labels = dataset.__getlabels_2019__()
    elif dset_name == 'inaturalist21':
        dataset = WrapNaturalist(dataroot, version='2021_train_mini', target_type=['phylum'], download=False)
        labels = dataset.__getlabels_2021__()
    
    num_samples = dataset.__len__()
    indices = np.arange(0,num_samples)
    
    
    print('counts labels', Counter(labels))
    
        
    if equalize_labels:
        print('equilize counts')
        indices = equilize_label_counts(labels)
        labels = labels[indices]
        num_samples = indices.shape[0]
        print('counts labels', Counter(labels))
        
    if clip_labels:
        print('clip counts')
        indices = clip_top_labels(labels, max_count=clip_max)
        labels = labels[indices]
        num_samples = indices.shape[0]
        print('counts labels', Counter(labels))

        
        
        

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


    print('tasks_list', tasks_list, len(tasks_list))
    

    tasks_list_train=[]
    tasks_list_test=[]
    tasks_list_holdout=[]
    for task_ in tasks_list:
        # for each label subset
        num_samples_task = task_.shape[0]
        print('num_samples_task', num_samples_task)
        # divide the train/test split
        split = int(np.floor(test_split * num_samples_task))
        inds_shuf = np.random.permutation(task_)
        tasks_list_test.append(inds_shuf[:split])
        
        inds_train = inds_shuf[split:]
        # split_holdout = int(np.floor(holdout_percent * inds_train.shape[0]))
        inds_train = np.random.permutation(inds_train)
        tasks_list_train.append(inds_train[split:])
        tasks_list_holdout.append(inds_train[:split])


    # save the files
    task_filepaths_train = saveTasks(tasks_list_train, dset_name, 'train', outdir, experiment_name, scenario='nc')
    task_filepaths_train_holdout = saveTasks(tasks_list_holdout, dset_name, 'holdout', outdir, experiment_name, scenario='nc')
    task_filepaths_test = saveTasks(tasks_list_test, dset_name, 'test', outdir, experiment_name, scenario='nc')


    return task_filepaths_train, task_filepaths_train_holdout, task_filepaths_test
        




def INATURALISTExperiments_w_holdout_w_validation(labels_per_task, dataroot, outdir, experiment_name, dset_name='inaturalist19',\
    target_ind=1, holdout_percent=0.2, test_split=0.2, validation_percent=0.1, train=True, scenario='nc', equalize_labels=False, clip_labels=False, clip_max=50000):
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
    
    if dset_name == 'inaturalist19':
        print('Get 19')
        dataset = WrapNaturalist(dataroot, version='2019', target_type=['super'],download=False)
        labels = dataset.__getlabels_2019__()
    elif dset_name == 'inaturalist21':
        print('Get 21')
        dataset = WrapNaturalist(dataroot, version='2021_train_mini', target_type=['phylum'], download=False)
        labels = dataset.__getlabels_2021__()
    
    num_samples = dataset.__len__()
    indices = np.arange(0,num_samples)
    
    
    print('counts labels', Counter(labels))
    
        
    if equalize_labels:
        print('equilize counts')
        indices = equilize_label_counts(labels)
        labels = labels[indices]
        num_samples = indices.shape[0]
        print('counts labels', Counter(labels))
        
    if clip_labels:
        print('clip counts')
        indices = clip_top_labels(labels, max_count=clip_max)
        labels = labels[indices]
        num_samples = indices.shape[0]
        print('counts labels', Counter(labels))

                
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
    tasks_list_test=[]
    for task_ in tasks_list:
        # for each label subset
        num_samples_task = task_.shape[0]
        print('num_samples_task', num_samples_task)
        
        inds_shuf = np.random.permutation(task_)
        
        # divide the train/test split
        split_val = int(np.floor(validation_percent*num_samples_task))
        tasks_list_val.append(inds_shuf[:split_val])
        inds_train = inds_shuf[split_val:]
        
        split_test = int(np.floor(test_split*num_samples_task))
        tasks_list_test.append(inds_train[:split_test])
        inds_train = inds_train[split_test:]

        split_holdout = int(np.floor(holdout_percent*num_samples_task))
        tasks_list_train.append(inds_train[split_holdout:])
        tasks_list_holdout.append(inds_train[:split_holdout])

        

    # --- save sequences to text files 
    task_filepaths_train = saveTasks(tasks_list_train, dset_name, 'train', outdir, experiment_name, scenario='nc')
    task_filepaths_train_holdout = saveTasks(tasks_list_holdout, dset_name, 'holdout', outdir, experiment_name, scenario='nc')
    task_filepaths_val = saveTasks(tasks_list_val, dset_name, 'validation', outdir, experiment_name, scenario='nc')
    task_filepaths_test = saveTasks(tasks_list_test, dset_name, 'test', outdir, experiment_name, scenario='nc')

    return task_filepaths_train, task_filepaths_train_holdout, task_filepaths_val, task_filepaths_test
        





def saveTasks(tasks_list, dset_name, partition, outdir, experiment_name, scenario='nc'):
    
    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir,dset_name, experiment_name)
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






def clip_top_labels(labels, max_count=50000):
    
    keys = np.unique(labels)
    
    indices = []
    # Now adjust counts 
    for k in keys:
        inds_k = np.where(labels==k)[0]
        inds_k = np.random.permutation(inds_k)
        inds_k = list(inds_k[:max_count])
        indices.extend(inds_k)
    
    indices = np.array(indices)
    
    return indices




def equilize_label_counts(labels, min_order=1):
    
    counts = dict(Counter(labels))
    
    counts = list(counts.items())

    keys, vals = zip(*counts)
    
    vals = np.array(list(vals))
    keys = np.array(list(keys))
    
    print(vals, keys)
    
    min_count = vals[np.argsort(vals)][min_order]
    
    print('min_count', min_count)
    
    indices = []
    # Now adjust counts 
    for k in keys:
        inds_k = np.where(labels==k)[0]
        inds_k = np.random.permutation(inds_k)
        inds_k = list(inds_k[:min_count])
        indices.extend(inds_k)
    
    indices = np.array(indices)
    
    return indices