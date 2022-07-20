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



def loadExperiments_w_holdout(experiment_filepath, dset_name):
    filetype={'cifar10':'npy', 'svhn':'npy', 'cifar100': 'npy', 'core50': 'npy', \
        'emnist':'npy', 'inaturalist':'npy'}

    tasks_filepaths_train = glob.glob('%s/train/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_train.sort(key=utils.natural_keys)

    tasks_filepaths_train_holdout = glob.glob('%s/holdout/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_train_holdout.sort(key=utils.natural_keys)

    tasks_filepaths_test = glob.glob('%s/test/*.%s'%(experiment_filepath, filetype[dset_name]))
    tasks_filepaths_test.sort(key=utils.natural_keys)

    return tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_test



def call_dataset_holdout(dset_name, data_dir, experiment_dir, experiment_filepath=None, experiment_name=None, 
                                            type_l_cifar='fine', num_tasks_cifar=19, num_tasks=9,
                                            holdout_percent=0.25,  max_holdout=0.75,  scenario='nc', 
                                            scenario_classif='class', exp_type='class', num_per_task=1, num_classes_first=2, keep_all_data=False,
                                            shuffle=False, preload=False, equalize_labels=True):

    start = time.time()

    if dset_name=='cifar10':
        import cifar10_dataset as dset
        total_classes=10
        obj2class={}
        for i in range(0,10):
            obj2class[i]=i

        target_ind =1

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'cifar10Task', 'transform_train_name':'cifar_train', \
            'transform_test_name':'cifar_test',  'tasks_gen':'cifar10tasks', 'experiment_gen':'cifar10Experiments_w_holdout', 'experiment_gen_test':'cifar10Experiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.cifar10tasks(num_per_task, num_tasks_cifar, first_task_num=num_classes_first,  shuffle=shuffle)
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
        seq_tasks, obj2class = dset.cifar100tasks(data_dir, num_per_task, num_tasks_cifar, type_l=type_l_cifar, first_task_num=num_classes_first, shuffle=shuffle)
        dset_prep={'total_classes':total_classes, 'use_coarse':use_coarse, 'scenario_classif':target_ind, 'homog_ind':2, 'obj2class':obj2class, 'dataset_wrap':'cifar100Task', 'transform_train_name':'cifar_train', \
            'transform_test_name':'cifar_test',  'tasks_gen':'cifar100tasks', 'experiment_gen':'cifar100Experiments_w_holdout', 'experiment_gen_test':'cifar100Experiments'}
        
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
            'transform_test_name':'svhn_test',  'tasks_gen':'svhntasks', 'experiment_gen':'svhnExperiments_w_holdout', 'experiment_gen_test':'svhnExperiments'}
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

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, \
            'dataset_wrap':'emnistTask', 'transform_train_name':'emnist_train', \
            'transform_test_name':'emnist_test',  'tasks_gen':'emnisttasks', 'experiment_gen':'emnistExperiments_w_holdout', \
                'experiment_gen_test':'emnistExperiments'}
        # Generate sequence of classes for experiment 
        seq_tasks = dset.emnisttasks(num_per_task, first_task_num=num_classes_first, shuffle=shuffle)
        seq_tasks_targets = seq_tasks 


    
    elif dset_name=='inaturalist':

        import inaturalist_dataset as dset
        target_ind = 1

        total_classes=6
        obj2class={}
        for i in range(0,total_classes):
            obj2class[i]=i

        dset_prep={'total_classes':total_classes, 'use_coarse':False, 'scenario_classif':1, 'homog_ind':2, 'obj2class':obj2class, \
            'dataset_wrap':'inaturalistTask', 'transform_train_name':'inaturalist_train_afterload', \
            'transform_test_name':'inaturalist_test_afterload',  'tasks_gen':'INATURALISTtasks', 'experiment_gen':'INATURALISTExperiments_w_holdout', \
                'experiment_gen_test':'INATURALISTExperiments'}
        # # Generate sequence of classes for experiment 
        seq_tasks = dset.INATURALISTtasks(num_tasks, num_per_task, first_task_num=num_classes_first, shuffle=shuffle)
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
            use_coarse=True
        elif scenario_classif=='instance':
            target_ind=1
            use_coarse=False
        dset_prep={'total_classes':total_classes, 'use_coarse':use_coarse, 'scenario_classif':target_ind, 'obj2class':obj2class, 'dataset_wrap':'core50Task', 'transform_train_name':'core50_train', \
            'transform_test_name':'core50_test','tasks_gen':'core50tasks', 'experiment_gen':'core50Experiments_w_holdout', 'experiment_gen_test':'core50Experiments'}
        

        seq_tasks_targets = seq_tasks

    print('seq_tasks: ', seq_tasks, target_ind)



    print('*****Prep Data*****')
    data_dir = '%s/%s'%(data_dir, dset_name)



    if experiment_filepath is not None:
        if os.path.exists(experiment_filepath):
            tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_test = loadExperiments_w_holdout(experiment_filepath, dset_name)
    else:
        try:
            assert experiment_name is not None
        except:
            print('Need to give experiment a name: set args.experiment_name as a string')
            sys.exit()
        print('Prepare DSET npy paths')
        
        if dset_name=='inaturalist':
            tasks_filepaths_train, tasks_filepaths_train_holdout, tasks_filepaths_test = dset.__getattribute__(dset_prep['experiment_gen'])(seq_tasks, data_dir, \
                experiment_dir, experiment_name, holdout_percent=holdout_percent, test_split=0.2, scenario=scenario, shuffle=shuffle, equalize_labels=equalize_labels)
        else:
            tasks_filepaths_train, tasks_filepaths_train_holdout = dset.__getattribute__(dset_prep['experiment_gen'])(seq_tasks, data_dir, \
                experiment_dir, experiment_name, holdout_percent=holdout_percent, max_holdout=max_holdout, target_ind=target_ind, train=True, scenario=scenario,shuffle=shuffle)
            tasks_filepaths_test = dset.__getattribute__(dset_prep['experiment_gen_test'])(seq_tasks, data_dir, \
                experiment_dir, experiment_name, target_ind=target_ind, train=False, scenario=scenario, shuffle=shuffle)
        print('prepared filelists', time.time()-start)


    print(experiment_filepath)
    num_tasks = len(tasks_filepaths_train)
    print('number tasks', num_tasks)


    # if preload:
    #     transform_train=None
    #     transform_test=None
    # else:
    transform_train =  tforms.__getattribute__(dset_prep['transform_train_name'])()
    transform_test =  tforms.__getattribute__(dset_prep['transform_test_name'])()


    if keep_all_data:
        # list of lists 
        train_filepaths_new_only = tasks_filepaths_train
        tasks_filepaths_train = combine_for_replay_all(tasks_filepaths_train)
        train_datasets_new_only = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=train_filepaths_new_only[i], preload=preload, \
                    transform=transform_train, train=True) for i in range(num_tasks)]



    test_dset = dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_train[1], preload=preload, \
        transform=transform_train, train=True)
    
    
    # print(test_dset.transform, test_dset.x.shape, test_dset.y.shape)
    
    # print(test_dset.__getitem__(0)[0].shape)
    
    # sys.exit()



    train_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_train[i], preload=preload, \
        transform=transform_train, train=True) for i in range(num_tasks)]

    if len(tasks_filepaths_train_holdout):
        train_holdout_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_train_holdout[i], preload=preload, \
            transform=transform_train, train=True) for i in range(num_tasks)]
    else:
        train_holdout_datasets=[]

    test_datasets = [dset.__getattribute__(dset_prep['dataset_wrap'])(data_dir, tasklist=tasks_filepaths_test[i], preload=preload, \
        transform=transform_test, train=False) for i in range(num_tasks)]

    print('prepared datasets', time.time()-start)




    if keep_all_data:
        return train_datasets, train_holdout_datasets, train_datasets_new_only, test_datasets, seq_tasks, seq_tasks_targets, dset_prep

    return train_datasets, train_holdout_datasets, test_datasets, seq_tasks,  seq_tasks_targets, dset_prep

