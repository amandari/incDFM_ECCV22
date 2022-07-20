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
import pickle

import utils

def remap_label(label, mapper):
        label = int(label.item())
        return int(mapper[label])



def superclass_remap_labels(data_dir, num_tasks, classes_per_task=5):
    # returns a fine-class to super-class dictionary
    
    classes_cifar=pickle.load(open(data_dir+'/cifar100/meta', 'rb'))
    fineclasses=classes_cifar['fine_label_names']
    superclasses=classes_cifar['coarse_label_names']
    superclasses_complete = [['beaver', 'dolphin', 'otter', 'seal', 'whale'], ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                            ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],['bottle', 'bowl', 'can', 'cup', 'plate'],
                            ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                            ['bed', 'chair', 'couch', 'table', 'wardrobe'], ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                            ['bear', 'leopard', 'lion', 'tiger', 'wolf'], ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                            ['cloud', 'forest', 'mountain', 'plain', 'sea'], ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                            ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'], ['crab', 'lobster', 'snail', 'spider', 'worm'],
                            ['baby', 'boy', 'girl', 'man', 'woman'], ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'], 
                            ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'], ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'], 
                            ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'], ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

    num2fineclasses_dict={}
    for i in range(len(fineclasses)):
        num2fineclasses_dict[i]=fineclasses[i]

    fineclasses2num_dict={v: k for k, v in num2fineclasses_dict.items()}

    fine2super={}
    finenum2super={}
    finenum2supernum={}

    for i in range(len(superclasses_complete)):
        for j in range(len(superclasses_complete[i])):
            fine2super[superclasses_complete[i][j]] = superclasses[i]
            finenum2super[fineclasses2num_dict[superclasses_complete[i][j]]] = superclasses[i]
            finenum2supernum[fineclasses2num_dict[superclasses_complete[i][j]]] = i


    finenum2supernum_list=[[] for i in range(20)]
    for key in finenum2supernum.keys():
        finenum2supernum_list[finenum2supernum[key]].append(key)

    seqindices=finenum2supernum_list
    ## later implement the incremental option 
    
    task_nclass_list = [classes_per_task for i in range(num_tasks)]
    
    remap_label_dict={}
    for t_l in range(num_tasks):
        for l in range(classes_per_task):
            remap_label_dict[seqindices[t_l][l]]=t_l
    
    return seqindices[:num_tasks], remap_label_dict, task_nclass_list
    

def flatten(t):
    return [item for sublist in t for item in sublist]


def cifar100tasks(data_dir, num_per_task=1, num_tasks=19, num_tasks_cifar=20, type_l='fine', first_task_num=2, shuffle=False):
    '''
    if 100 is not divisible by num_per_class, then make extra classes go into first task. 
    '''
    
    num_classes = 100
    if type_l=='super':
        tasks, obj2class,_ = superclass_remap_labels(data_dir, num_tasks_cifar, classes_per_task=5)
        tasks = [flatten(tasks[:first_task_num])] + tasks[first_task_num:]
        # TODO rest like shuffling
    else:
        classes = np.arange(0,num_classes,1)
        rest= num_classes%num_per_task
        if shuffle:
            classes = classes[np.random.permutation(num_classes)]
        classes = list(classes)

        if first_task_num is not None:
            first_task=classes[:first_task_num]
        else:
            first_task = classes[:(num_per_task+rest)]
        other_tasks = utils.list_to_2D(classes[len(first_task):],num_per_task)
        tasks = [first_task]+other_tasks
        obj2class={}
        for i in range(0,num_classes):
            obj2class[i]=i

    return tasks[:num_tasks], obj2class
    













def loadExperiments_cifar100(experiment_filepath):
    tasks_filepaths_train = glob.glob('%s/train/*.npy'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)
    tasks_filepaths_test = glob.glob('%s/test/*.npy'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    return tasks_filepaths_train, tasks_filepaths_test


  
def load_cifar100(root, train=True):
    
    # base_folder = 'cifar100'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    root = os.path.expanduser(root)

    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list
        
    data = []
    labels = []
    labels_coarse = []

    # now load the picked numpy arrays
    for file_name, checksum in downloaded_list:
        file_path = os.path.join(root, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                labels.extend(entry['labels'])
            else:
                labels.extend(entry['fine_labels'])
                labels_coarse.extend(entry['coarse_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC

    data = np.array(data)
    labels = np.array(labels)
    labels_coarse = np.array(labels_coarse)
    
    
    data = np.transpose(data, (0, 3, 1, 2))
    
    data = torch.from_numpy(data).type(torch.FloatTensor)
    # y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    labels_coarse = torch.from_numpy(labels_coarse).type(torch.LongTensor)
    
    return data, labels, labels_coarse
            



class cifar100Task():
    def __init__(self, dataroot, dset_name='cifar100', train=True, tasklist='task_indices.npy', transform=None, returnIDX=False, preload=True):
        """ 
        dataset for each individual task
        """
        data, labels, labels_coarse = load_cifar100(dataroot, train=train)
        
        self.dset_name = dset_name
        
        # have option of loading mltiple tasklists if teh case
        if isinstance(tasklist, list):
            self.indices_task_init=[]
            for l in tasklist:
                self.indices_task_init.append(np.load(l))
            self.indices_task_init = np.concatenate(self.indices_task_init)
        else:
            self.indices_task_init = np.load(tasklist)

        self.x = data 
        self.y = labels 
        self.y_coarse = labels_coarse
        self.transform = transform
        self.preload = preload 
        self.returnIDX = returnIDX
        
        # get appropriate task data
        self.x = self.x[self.indices_task_init, ...]
        self.y = self.y[self.indices_task_init]
        self.y_coarse = self.y_coarse[self.indices_task_init]
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

        fine_lbl = self.y[idx]
        coarse_lbl = self.y_coarse[idx]

        if self.returnIDX:
            return im, fine_lbl, coarse_lbl, idx
            
        return im, fine_lbl, coarse_lbl




def cifar100Experiments_w_holdout(labels_per_task, dataroot, outdir, experiment_name,  target_ind=1, holdout_percent=0.25, max_holdout=0.75, \
    train=True, scenario='nc', shuffle=False):
    """ 
    cifar100 Experiments setup
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
    data, labels, labels_coarse = load_cifar100(dataroot, train=train)
    # if target_ind==2:
    #     labels = labels_coarse
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
    dest_dir = '%s/%s/%s'%(outdir,'cifar100', experiment_name)
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
        


    

def cifar100Experiments_w_holdout_w_validation(labels_per_task, dataroot, outdir, experiment_name, \
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
    data, labels, labels_coarse = load_cifar100(dataroot, train=train)
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
        





def cifar100Experiments(labels_per_task, dataroot, outdir, experiment_name, target_ind=1, train=True, scenario='nc', shuffle=False):
    """ 
    cifar100 with fine labels or coarse (sperclass labels) incrementally
    labels_per_task (in form of fine label)
    saves out indices for each task in outdir in form of numpy arrays 
    """
    if train:
        partition='train'
    else:
        partition='test'
    
    # root = os.path.expanduser(root)
    dataroot = Path(dataroot)
    # get on basis of fine labels 
    data, labels, labels_coarse = load_cifar100(dataroot, train=train)
    # if target_ind==2:
    #     labels = labels_coarse
    data = data.numpy()
    labels = labels.numpy()
    num_samples = data.shape[0]
    indices = np.arange(0,num_samples)


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
    dest_dir = '%s/%s/%s'%(outdir,'cifar100', experiment_name)
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
        



def saveTasks(tasks_list, partition, outdir, experiment_name, scenario='nc'):
    
    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir,'cifar100', experiment_name)
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






