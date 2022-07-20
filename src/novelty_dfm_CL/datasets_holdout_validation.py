import numpy as np
import glob 

import os
import time
import sys
import torch
from torch.utils.data import Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tforms
import utils
from datasets_utils import combine_for_replay_all


def Num_Tasks_CL(dset_name, num_tasks=-1):
    'Based on 1-1 class experiment'
    if num_tasks>0:
        return num_tasks
    else:
        task_map ={'cifar10':9, 'svhn':9, 'emnist':25, 'cifar100':19, 'inaturalist19':5, 'inaturalist21':8}
        num_tasks=task_map[dset_name]
        return num_tasks


def loadExperiments_w_holdout_w_validation(experiment_filepath, dset_name):
    filetype={'cifar10':'npy', 'svhn':'npy', 'cifar100': 'npy', 'core50': 'npy', \
        'emnist':'npy', 'inaturalist19':'npy', 'inaturalist21':'npy'}

    tasks_filepaths_train = glob.glob('%s/train/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_train.sort(key=utils.natural_keys)

    tasks_filepaths_train_holdout = glob.glob('%s/holdout/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_train_holdout.sort(key=utils.natural_keys)

    tasks_filepaths_test = glob.glob('%s/test/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_test.sort(key=utils.natural_keys)
    
    tasks_filepaths_val = glob.glob('%s/validation/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_val.sort(key=utils.natural_keys)


    return tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_val, tasks_filepaths_test



def call_dataset_holdout_w_validation(dset_name, data_dir, experiment_dir, experiment_filepath=None, experiment_name=None, 
                                            type_l_cifar='fine', num_tasks_cifar=20, num_tasks=9,
                                            holdout_percent=0.2, val_holdout=0.1, scenario='nc', 
                                            num_per_task=1, num_classes_first=2, keep_all_data=False,
                                            shuffle=False, preload=False, equalize_labels=False, clip_labels=True, clip_max=50000):

    start = time.time()

    if dset_name=='cifar10':
        import cifar10_dataset as dset
        total_classes=10
        obj2class={}
        for i in range(0,10):
            obj2class[i]=i

        target_ind=1

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'cifar10Task', 'transform_train_name':'cifar_train', \
            'transform_test_name':'cifar_test',  'tasks_gen':'cifar10tasks', 'experiment_gen':'cifar10Experiments_w_holdout_w_validation', 'experiment_gen_test':'cifar10Experiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.cifar10tasks(num_per_task, num_tasks=num_tasks, first_task_num=num_classes_first,  shuffle=shuffle)
        seq_tasks_targets = seq_tasks 

    
    elif dset_name=='cifar100':
        import cifar100_dataset as dset
        total_classes=100
        if type_l_cifar=='super':
            target_ind=2
            use_coarse=True
        elif type_l_cifar=='fine':
            target_ind=1
            use_coarse=False
        seq_tasks, obj2class = dset.cifar100tasks(data_dir, num_per_task, num_tasks=num_tasks, num_tasks_cifar=num_tasks_cifar, type_l=type_l_cifar, first_task_num=num_classes_first, shuffle=shuffle)
        dset_prep={'total_classes':total_classes, 'use_coarse':use_coarse, 'scenario_classif':target_ind, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'cifar100Task', 'transform_train_name':'cifar_train', \
            'transform_test_name':'cifar_test',  'tasks_gen':'cifar100tasks', 'experiment_gen':'cifar100Experiments_w_holdout_w_validation', 'experiment_gen_test':'cifar100Experiments'}
        
        print('seq_tasks', seq_tasks, obj2class)

        seq_tasks_targets = []
        for t_ in seq_tasks:
            temp=[]
            for i_ in t_:
                temp.append(obj2class[i_])
            seq_tasks_targets.append(list(set(temp)))

        print('seq_tasks_targets', seq_tasks_targets)



    elif dset_name=='svhn':
        import svhn_dataset as dset
        target_ind = 1

        total_classes=10
        obj2class={}
        for i in range(0,10):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'svhnTask', 'transform_train_name':'svhn_train', \
            'transform_test_name':'svhn_test',  'tasks_gen':'svhntasks', 'experiment_gen':'svhnExperiments_w_holdout_w_validation', 'experiment_gen_test':'svhnExperiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.svhntasks(num_per_task, num_tasks=num_tasks, first_task_num=num_classes_first, shuffle=shuffle)
        seq_tasks_targets = seq_tasks 



    elif dset_name=='emnist':
        import emnist_dataset as dset
        target_ind = 1

        total_classes=26
        obj2class={}
        for i in range(0,total_classes):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, \
            'dataset_wrap':'emnistTask', 'transform_train_name':'emnist_train', \
            'transform_test_name':'emnist_test',  'tasks_gen':'emnisttasks', 'experiment_gen':'emnistExperiments_w_holdout_w_validation', \
                'experiment_gen_test':'emnistExperiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.emnisttasks(num_per_task, num_tasks=num_tasks, first_task_num=num_classes_first, shuffle=shuffle)
        seq_tasks_targets = seq_tasks 


    
    elif 'inaturalist' in dset_name:

        import inaturalist_dataset as dset
        target_ind = 1

        if '19' in dset_name:
            total_classes=6
        elif '21' in dset_name:
            total_classes=9
            
        obj2class={}
        for i in range(0,total_classes):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, \
            'dataset_wrap':'inaturalistTask', 'transform_train_name':'inaturalist_train', \
            'transform_test_name':'inaturalist_test',  'tasks_gen':'INATURALISTtasks', 'experiment_gen':'INATURALISTExperiments_w_holdout_w_validation', \
                }
        # # Generate sequence of classes for experiment 
        seq_tasks = dset.INATURALISTtasks(num_per_task, total_classes, num_tasks=num_tasks, first_task_num=num_classes_first, shuffle=shuffle)
        seq_tasks_targets = seq_tasks 


    
        seq_tasks_targets = seq_tasks

    print('seq_tasks: ', seq_tasks, target_ind)



    print('*****Prep Data*****')
    data_dir = '%s/%s'%(data_dir, dset_name)


    if experiment_name is None:
        experiment_name = 'holdout_%.2f_val_%.2f'%( holdout_percent, val_holdout)
        
    if experiment_filepath is None:
        experiment_filepath = '%s/%s/%s/'%(experiment_dir, dset_name, experiment_name)

    # if experiment_filepath is not None:
    if os.path.exists(experiment_filepath):
        print('Load existing Experiment', experiment_filepath)
        tasks_filepaths_train, tasks_filepaths_train_holdout, task_filepaths_val, tasks_filepaths_test = loadExperiments_w_holdout_w_validation(experiment_filepath, \
            dset_name)
    else:
    #     print('Did not find experiment to load')
    # else:
        # try:
        #     assert experiment_name is not None
        # except:
        #     print('Need to give experiment a name: set args.experiment_name as a string')
        #     sys.exit()
        print('Create New Experiment')
        if 'inaturalist' in dset_name:
            tasks_filepaths_train, tasks_filepaths_train_holdout, task_filepaths_val, tasks_filepaths_test = dset.__getattribute__(dset_prep['experiment_gen'])(seq_tasks, data_dir, \
                experiment_dir, experiment_name, dset_name=dset_name, holdout_percent=holdout_percent, validation_percent=val_holdout, test_split=0.2, \
                    scenario=scenario, equalize_labels=equalize_labels, clip_labels=clip_labels, clip_max=clip_max)
        else:
            tasks_filepaths_train, tasks_filepaths_train_holdout, task_filepaths_val = dset.__getattribute__(dset_prep['experiment_gen'])(seq_tasks, data_dir, \
                experiment_dir, experiment_name, holdout_percent=holdout_percent, validation_percent=val_holdout, target_ind=target_ind, train=True, scenario=scenario)
            tasks_filepaths_test = dset.__getattribute__(dset_prep['experiment_gen_test'])(seq_tasks, data_dir, \
                experiment_dir, experiment_name, target_ind=target_ind, train=False, scenario=scenario)
        # print('finished preparing', time.time()-start)



    print(experiment_filepath)
    num_tasks = len(tasks_filepaths_train)
    print('number tasks', num_tasks)
    
    # sys.exit()


    transform_train =  tforms.__getattribute__(dset_prep['transform_train_name'])()
    transform_test =  tforms.__getattribute__(dset_prep['transform_test_name'])()


    if keep_all_data:
        # list of lists 
        train_filepaths_new_only = tasks_filepaths_train
        tasks_filepaths_train = combine_for_replay_all(tasks_filepaths_train)
        train_datasets_new_only = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, dset_name=dset_name, tasklist=train_filepaths_new_only[i], preload=preload, \
                    transform=transform_train, train=True) for i in range(num_tasks)]



    train_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, dset_name=dset_name, tasklist=tasks_filepaths_train[i], preload=preload, \
        transform=transform_train, train=True) for i in range(num_tasks)]

    if len(tasks_filepaths_train_holdout):
        train_holdout_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, dset_name=dset_name, tasklist=tasks_filepaths_train_holdout[i], preload=preload, \
            transform=transform_train, train=True) for i in range(num_tasks)]
    else:
        train_holdout_datasets=[]
        
    if len(task_filepaths_val):
        val_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, dset_name=dset_name, tasklist=task_filepaths_val[i], preload=preload, \
        transform=transform_test, train=True) for i in range(num_tasks)]
    else:
        val_datasets = []

    test_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, dset_name=dset_name, tasklist=tasks_filepaths_test[i], preload=preload, \
        transform=transform_test, train=False) for i in range(num_tasks)]
    

    print('prepared datasets', time.time()-start)
    
    
    # print('Dset sizes', train_datasets[1].__len__(),  train_holdout_datasets[1].__len__(), val_datasets[1].__len__(), test_datasets[1].__len__())

    # sys.exit()
    if keep_all_data:
        return train_datasets, train_holdout_datasets, train_datasets_new_only, val_datasets, test_datasets, seq_tasks, seq_tasks_targets, dset_prep

    return train_datasets, train_holdout_datasets, val_datasets, test_datasets, seq_tasks, seq_tasks_targets, dset_prep

