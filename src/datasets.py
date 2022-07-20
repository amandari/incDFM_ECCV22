import numpy as np
import glob 
import utils
import os
import time
import sys
import torch
from torch.utils.data import Dataset

from datasets_utils import combine_for_replay_all

import tforms





def loadExperiments(experiment_filepath, dset_name):
    filetype={'cifar10':'npy', 'svhn':'npy', 'cifar100': 'npy', 'core50': 'npy', 'emnist':'npy'}

    tasks_filepaths_train = glob.glob('%s/train/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_train.sort(key=utils.natural_keys)

    tasks_filepaths_test = glob.glob('%s/test/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_test.sort(key=utils.natural_keys)

    return tasks_filepaths_train, tasks_filepaths_test




# make script to call datasets (all so far)
def call_dataset(dset_name, data_dir, experiment_dir, experiment_filepath=None, 
                                            type_l_cifar='fine', num_tasks_cifar=20, 
                                            experiment_name=None, scenario='nc', keep_all_data=False, 
                                            scenario_classif='class', exp_type='class', num_per_task=1, num_classes_first=2, 
                                            shuffle=False, online=False, preload=False):

    start = time.time()

    if dset_name=='cifar10':
        import cifar10_dataset as dset
        total_classes=10
        target_ind = 1
        obj2class={}
        for i in range(0,10):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'cifar10Task', 'transform_train_name':'cifar_train', \
            'transform_test_name':'cifar_test',  'tasks_gen':'cifar10tasks', 'experiment_gen':'cifar10Experiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.cifar10tasks(num_per_task, num_tasks_cifar, first_task_num=num_classes_first,  shuffle=shuffle)

        seq_tasks_targets = seq_tasks 
    
    elif dset_name=='cifar100':
        import cifar100_dataset as dset
        total_classes=100
        if type_l_cifar=='super':
            target_ind=2
        elif type_l_cifar=='fine':
            target_ind=1
        seq_tasks, obj2class = dset.cifar100tasks(data_dir, num_per_task, num_tasks_cifar, type_l=type_l_cifar, first_task_num=num_classes_first, shuffle=shuffle)
        dset_prep={'total_classes':total_classes, 'scenario_classif':target_ind, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'cifar100Task', 'transform_train_name':'cifar_train', \
            'transform_test_name':'cifar_test',  'tasks_gen':'cifar100tasks', 'experiment_gen':'cifar100Experiments'}

        seq_tasks_targets = []
        for t_ in seq_tasks:
            temp=[]
            for i_ in t_:
                temp.append(obj2class[i_])
            seq_tasks_targets.append(list(set(temp)))
                
    elif dset_name=='svhn':
        import svhn_dataset as dset
        target_ind = 1

        total_classes=10
        obj2class={}
        for i in range(0,10):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'svhnTask', \
            'transform_train_name':'svhn_train', \
            'transform_test_name':'svhn_test',  'tasks_gen':'svhntasks', 'experiment_gen':'svhnExperiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.svhntasks(num_per_task, first_task_num=num_classes_first, shuffle=shuffle)
        seq_tasks_targets = seq_tasks 


    
    elif dset_name=='emnist':
        import emnist_dataset as dset
        target_ind = 1

        total_classes=26
        obj2class={}
        for i in range(0,total_classes):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, \
            'dataset_wrap':'emnistTask', 'transform_train_name':'emnist_train', \
            'transform_test_name':'emnist_test',  'tasks_gen':'emnisttasks', 'experiment_gen':'emnistExperiments', \
                'experiment_gen_test':'emnistExperiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.emnisttasks(num_per_task, first_task_num=num_classes_first, shuffle=shuffle)
        seq_tasks_targets = seq_tasks 



    
    elif dset_name=='core50':
        import core50_dataset as dset
        total_classes = {'class':10, 'instance':50}
        # Generate sequence of classes for experiment 
        obj2class = dset.Core50_classdict()
        total_classes = total_classes[scenario_classif]
        seq_tasks = dset.core50tasks(scenario,  exp_type=exp_type, num_classes_first=num_classes_first, num_class_per_task=num_per_task, shuffle=shuffle)
        if scenario_classif=='class':
            target_ind=2
        elif scenario_classif=='instance':
            target_ind=1
        dset_prep={'total_classes':total_classes,  'scenario_classif':target_ind, 'obj2class':obj2class, 'dataset_wrap':'core50Task', 'transform_train_name':'core50_train', \
            'transform_test_name':'core50_test','tasks_gen':'core50tasks', 'experiment_gen':'core50Experiments'}

        seq_tasks_targets = seq_tasks 
        
    print('seq_tasks: ', seq_tasks)

    print('*****Prep Data*****')
    data_dir = '%s/%s'%(data_dir, dset_name)
    # if args.experiment_filepath:
    if experiment_filepath is not None:
        if os.path.exists(experiment_filepath):
            tasks_filepaths_train, tasks_filepaths_test = loadExperiments(experiment_filepath, dset_name)
            # list_tasks = dset.__getattribute__(dset_prep['tasks_gen'])(scenario)
    else:
        try:
            assert experiment_name is not None
        except:
            print('Need to give experiment a name: set args.experiment_name as a string')
            sys.exit()
        tasks_filepaths_train = dset.__getattribute__(dset_prep['experiment_gen'])(seq_tasks, data_dir, experiment_dir, experiment_name, target_ind=target_ind, train=True, scenario=scenario,shuffle=shuffle)
        tasks_filepaths_test = dset.__getattribute__(dset_prep['experiment_gen'])(seq_tasks, data_dir, experiment_dir, experiment_name, target_ind=target_ind, train=False, scenario=scenario, shuffle=shuffle)
        print('prepared filelists', time.time()-start)




    if preload:
        transform_train=None
        transform_test=None
    else:
        transform_train =  tforms.__getattribute__(dset_prep['transform_train_name'])()
        transform_test =  tforms.__getattribute__(dset_prep['transform_test_name'])()

    print(experiment_filepath)
    num_tasks = len(tasks_filepaths_train)
    print('tasks_filepaths_train', tasks_filepaths_train)
    print('number tasks', num_tasks)

    # If flag finetune_back=='all' then we need 
    if keep_all_data:
        # list of lists 
        train_filepaths_new_only = tasks_filepaths_train
        tasks_filepaths_train = combine_for_replay_all(tasks_filepaths_train)
        # tasks_filepaths_train_old = [tasks_filepaths_train[k][:-1] for k in range(len(tasks_filepaths_train))]

        train_datasets_new_only = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=train_filepaths_new_only[i], preload=preload, \
                    transform=transform_train, train=True) for i in range(num_tasks)]

        # train_datasets_old = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_train_old[i], preload=preload, \
        # transform=transform_train) for i in range(1,num_tasks)]
        # train_datasets_old = [[]]+train_datasets_old

         

    train_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_train[i], preload=preload, \
        transform=transform_train, train=True) for i in range(num_tasks)]

    test_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_test[i], preload=preload, \
        transform=transform_test, train=False) for i in range(num_tasks)]

    print('prepared datasets', time.time()-start)

    if keep_all_data:
        return train_datasets, train_datasets_new_only, test_datasets, seq_tasks, seq_tasks_targets, dset_prep

    return train_datasets, test_datasets, seq_tasks, seq_tasks_targets, dset_prep


# have 3rd dataset loader only for current task in train mode for keep_all_data option (because of novelty_detection score function)


