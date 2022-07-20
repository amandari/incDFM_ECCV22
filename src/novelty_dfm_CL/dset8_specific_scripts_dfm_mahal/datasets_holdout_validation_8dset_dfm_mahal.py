import numpy as np
import glob 

import os
import time
import sys
import torch
from torch.utils.data import Dataset
import src.novelty_dfm_CL.dset8_specific_scripts_dfm_mahal.dset8_dfm_mahal as dset

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import tforms
import utils
from datasets_utils import combine_for_replay_all



def loadExperiments_w_holdout_w_validation(experiment_filepath):

    tasks_filepaths_train = glob.glob('%s/train/*'%(experiment_filepath))
    tasks_filepaths_train.sort(key=utils.natural_keys)

    tasks_filepaths_train_holdout = glob.glob('%s/holdout/*'%(experiment_filepath))
    tasks_filepaths_train_holdout.sort(key=utils.natural_keys)

    tasks_filepaths_test = glob.glob('%s/test/*'%(experiment_filepath))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    
    tasks_filepaths_val = glob.glob('%s/validation/*'%(experiment_filepath))
    tasks_filepaths_val.sort(key=utils.natural_keys)


    return tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_val, tasks_filepaths_test



def call_8dset_holdout_w_validation(data_dir, experiment_dir, experiment_filepath=None, experiment_name=None, 
                                            num_tasks=8, target_ind=1,
                                            holdout_percent=0.2, val_holdout=0.1):

    start = time.time()

    dset_name = '8dset'
    seq_tasks = [[0], [1], [2], [3], [4], [5], [6], [7]]
    total_classes = 8
    list_tasks = ['flowers', 'scenes', 'birds', 'cars', 'aircraft', 'voc', 'chars', 'svhn']
    seq_tasks = seq_tasks[:num_tasks]
    print('seq_tasks: ', seq_tasks, target_ind)



    dset_prep={'use_coarse_data':True, 'scenario_classif':target_ind, 'homog_ind':2, 'total_classes':total_classes, \
        }
    # # Generate sequence of classes for experiment 

    print('*****Prep Data*****')
    data_dir = '%s/%s'%(data_dir, dset_name)


    if experiment_name is None:
        experiment_name = 'holdout_%.2f_val_%.2f'%( holdout_percent, val_holdout)
        
    if experiment_filepath is None:
        experiment_filepath = '%s/%s/%s/'%(experiment_dir, dset_name, experiment_name)




    # if experiment_filepath is not None:
    if os.path.exists(experiment_filepath):
        print('Load existing Experiment', experiment_filepath)
        tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_val, tasks_filepaths_test = loadExperiments_w_holdout_w_validation(experiment_filepath)
    else:
        tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_val = dset.eightdset_Experiments_w_holdout_w_validation_trainset(data_dir, \
            experiment_dir, experiment_name, holdout_percent=holdout_percent, validation_percent=val_holdout)
        
        tasks_filepaths_test = dset.eightdset_Experiments_testset(data_dir, experiment_dir, experiment_name)
        

    print(experiment_filepath)
    num_tasks = len(tasks_filepaths_train)
    print('number tasks', num_tasks)
    
    # sys.exit()


    transform_train =  [tforms.eightdset_train() for i in range(num_tasks) if i!=7]+[tforms.eightdset_train_svhn()]
    transform_test =  [tforms.eightdset_test() for i in range(num_tasks) if i!=7]+[tforms.eightdset_test_svhn()]



    train_datasets = [dset.eightdsetTask(data_dir, list_tasks[i], split='train', tasklist=tasks_filepaths_train[i],\
        transform=transform_train[i]) for i in range(num_tasks)]
    
    
    if len(tasks_filepaths_train_holdout):
        train_holdout_datasets = [dset.eightdsetTask(data_dir, list_tasks[i], split='train', tasklist=tasks_filepaths_train_holdout[i],\
        transform=transform_train[i]) for i in range(num_tasks)]
    else:
        train_holdout_datasets=[]
        
    if len(tasks_filepaths_val):
        val_datasets = [dset.eightdsetTask(data_dir, list_tasks[i], split='train', tasklist=tasks_filepaths_val[i],\
        transform=transform_train[i]) for i in range(num_tasks)]
    else:
        val_datasets = []

    test_datasets = [dset.eightdsetTask(data_dir, list_tasks[i], split='val', tasklist=tasks_filepaths_test[i],\
        transform=transform_test[i]) for i in range(num_tasks)]
    

    print('prepared datasets', time.time()-start)
    

    return train_datasets, train_holdout_datasets, val_datasets, test_datasets, seq_tasks, dset_prep

