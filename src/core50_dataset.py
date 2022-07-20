from __future__ import print_function

import os
import os.path
import pickle as pkl
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import image
import fnmatch
from pathlib import Path
import glob
import fileinput

import utils 


def loadExperiments_core50(experiment_filepath):
    tasks_filepaths_train = glob.glob('%s/train/*.txt'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)
    tasks_filepaths_test = glob.glob('%s/test/*.txt'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    return tasks_filepaths_train, tasks_filepaths_test



def Core50_classdict(scenario_classif='class', per_class_step=5, num_total_instances=50):
    obj2class={}
    for i in range(1,num_total_instances+1,1):
        if scenario_classif=='class':
            obj2class[i]=int(np.floor((i-1)/per_class_step))
        else:
            obj2class[i]=i
    return obj2class


def core50tasks(scenario='nc', exp_type='class', num_classes_first=2, num_class_per_task=1, shuffle=False):
    '''
    scenario (str): Type of experiment to run, nc=incremental and joint_nc is multitask joint training. 
    per_task ()
    '''
    # what classes to learn in what order 
    num_classes = 10
    per_class_step=5# core50 has 5 instances per class  
    num_total_instances = 50

    class_list=np.arange(num_classes) #there are 10 classes, with 5 instances per class
    if shuffle:
        class_list = class_list[np.random.permutation(class_list.shape[0])]
    class_list = list(class_list)
    instances = []
    for i in class_list:
        instances.extend([(i*per_class_step)+j for j in range(per_class_step)])
    instances = [i+1 for i in instances]
    # print(instances)
    
    if scenario=='nc':
        first_task = [instances[i] for i in range(num_classes_first*per_class_step)]
        if exp_type=='instance':
            other_tasks=[[instances[j]] for j in range(len(first_task),num_total_instances,1)]
        elif exp_type=='class':
            classes_inc = utils.list_to_2D(class_list[num_classes_first:],num_class_per_task)
            # print(classes_inc)
            other_tasks = []
            for task in classes_inc:
                other_tasks.append([])
                for c in task:
                    other_tasks[-1].extend([(c*5)+k+1 for k in range(5)])
            # print(other_tasks)
        list_tasks = [first_task]+other_tasks
    elif scenario=='joint_nc':
        list_tasks = [[i for i in range(1,num_total_instances+1,1)]]
    else:
        list_tasks = None

    return list_tasks



def core50Experiments(list_tasks, root, outdir, experiment_name, train=True, scenario='nc', shuffle=False):
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
    # root = os.path.expanduser(root)
    root = Path(root)
    # with open(os.path.join(root, 'additional', 'paths.pkl'), 'rb') as f:
    with open(root /'additional' / 'paths.pkl', 'rb') as f:
        paths = pkl.load(f)

    # --- Hardcoded functions for core50 
    obj2class = Core50_classdict()
    if train:
        envs = [0, 1, 3, 4, 5, 7, 8, 10] 
        partition = 'train'
    else:
        envs = [2, 6, 9]
        partition = 'test'

    # list_tasks = core50tasks(scenario, shuffle=shuffle)

    # TODO shuffle for different runs 
    
    paths_=[]
    for env in envs:
        paths_.extend([filename for filename in paths if fnmatch.fnmatch(filename,'s%d/*'%env)])

    # --- Set up Task sequences 
    tasks_list=[]
    if scenario=='nc':
        for task in list_tasks:
            tasks_list.append([])
            # seq_wilds = [('s*/o%d/*'%i, i-1) for i in list_objects]
            for obj in task:
                pattern = 's*/o%d/*'%(obj)
                lbl = obj2class[obj]
                tasks_list[-1].extend([(filename, obj, lbl) for filename in paths_ if fnmatch.fnmatch(filename,pattern)])
    elif scenario == 'joint_nc':
        tasks_list.append([])
        for filename in paths_:
            obj = int(filename.split('/')[1][1:])
            lbl = obj2class[obj]
            tasks_list[-1].append((filename, obj, lbl)) 


    # --- save sequences to text files 
    dest_dir = '%s/%s/%s'%(outdir,'core50', experiment_name)
    utils.makedirectory(dest_dir)
    dest_dir = dest_dir + '/%s'%(partition)
    utils.makedirectory(dest_dir)

    # Create directory for experiment 
    task_filepaths=[]
    for task in range(len(tasks_list)):
        task_filepaths.append("%s/%s_%s_task_%d.txt"%(dest_dir, scenario, partition, task))
        textfile = open(task_filepaths[-1], "w")
        for tup in tasks_list[task]:
            path_obj, obj, lbl_obj = tup
            textfile.write("%s %d %d\n"%(path_obj, obj-1, lbl_obj))
        textfile.close()

    return task_filepaths
    # tasks_list
        

def read_lines_filetxt(file):
    file_content = open(file, "r")
    lines = file_content.read().splitlines()
    return lines



class core50Task(Dataset):
    def __init__(self, root, tasklist='tasklist.txt', preload=True, transform=None, returnIDX=False):
        """ 
        CORe50 Pytorch Dataset wrapper for each task
        Args:
            root (string): Root directory of the dataset where images and paths file are stored
            tasklist (string): filepath for paths of images belonging to a given task as well as the label of the object. 
            preload (bool, optional): If True data is pre-loaded with look-up
                tables. RAM usage may be high.
            multilabel (bool): If outputting more then one label per input 
        """
        self.root = Path(root)
        self.preload = preload
        
        # transform into list of strings
        self.tasklist = []
        if isinstance(tasklist, list):
            for file in tasklist: # read all txt files in working directory
                lines = read_lines_filetxt(file)
                for line in lines:
                    self.tasklist.append(line)
        else:
            lines = read_lines_filetxt(tasklist)
            for line in lines:
                self.tasklist.append(line)


        self.transform = transform
        self.returnIDX = returnIDX

        self.x_paths=[]
        self.y = [[],[]]
        # loop through list of strings 
        for l in self.tasklist:
            obj_path, obj, lbl_obj = l.split(' ')
            self.y[0].append(obj)
            self.y[1].append(lbl_obj)
            self.x_paths.append(obj_path)



        self.y = np.array(self.y).astype('int')
        self.y = torch.from_numpy(self.y)

        self.indices_task_init = np.arange(self.y.shape[1])
        self.indices_task = np.copy(self.indices_task_init)

        if self.preload:
            # print("Loading data...")
            self.x = np.zeros((self.y.shape[1], 3, 128, 128))
            # self.x = np.zeros((len(self.y), 128, 128, 3))
            for i, path_ in enumerate(self.x_paths):
                # raw = image.imread(self.root / 'images'/ path_)
                raw = utils.image2arr(self.root / 'images'/ path_)
                self.x[i,...] = raw

            self.x = self.x / 255
            self.x[:, 0, :, :] = ((self.x[:, 0, :, :] - 0.485) / 0.229)
            self.x[:, 1, :, :] = ((self.x[:, 1, :, :] - 0.456) / 0.224)
            self.x[:, 2, :, :] = ((self.x[:, 2, :, :] - 0.406) / 0.225)
            self.x = torch.from_numpy(self.x).type(torch.FloatTensor)


    def __len__(self):
        return self.indices_task.shape[0]


    def select_random_subset(self, random_num):

        inds_keep = np.random.permutation(np.arange(self.indices_task_init.shape[0]))[:random_num]

        self.indices_task = self.indices_task_init[inds_keep]
        
    def select_specific_subset(self, indices_select):
        
        self.indices_task = self.indices_task_init[indices_select]
        


    def __getitem__(self, idx):

        idx = self.indices_task[idx]

        if self.preload:
            im = self.x[idx,...]
        else:
            # im = image.imread(self.root / 'images'/ self.x_paths[idx])
            im = utils.image2arr(self.root / 'images'/ self.x_paths[idx])
            im = torch.from_numpy(im).type(torch.FloatTensor)/255

        if self.transform is not None: ##input the desired tranform 

            im = self.transform(im)

        obj_lbl = self.y[0,idx]
        class_lbl = self.y[1,idx]

        if self.returnIDX:
            return im, obj_lbl, class_lbl, idx
            
        return im, obj_lbl, class_lbl


