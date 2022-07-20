'''
Main Script (Novelty Detection) Mix Old + New without distribution shift 


--------- To run (currently with flag only_novelty which skips training, change in yaml file)---------
===for cifar10:
===class incremental (task = all instances for 1 class)
CUDA_VISIBLE_DEVICES=2 python3 -i ./src/novelty_level1/main_novelty_level1.py --config_file ./src/configs/dfm_noveltyl1_cifar10_classinc_config.yaml --test_num 0

'''

from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot
import glob
import pickle
import sys
import os
import argparse
from tqdm import tqdm
import time 
import random 
import json
import shutil
from collections import Counter
from datetime import datetime
import re

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

general_seed = 0
seed_torch(general_seed)

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import datasets as dset
import classifier as clf
import memory as mem
from train import train_main_epoch
from test import test_main
import tforms
import utils

from datasets_holdout import call_dataset_holdout

import novelty_detector as novel 

import feature_extraction.quantization as quant
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents
import feature_extraction.latent_layer_functions as latent

parser = argparse.ArgumentParser(description='Main Script')

#---- general 
parser.add_argument('--scenario', type=str, default='nc', help='If incremental (nc) or joint (joint_nc)')
parser.add_argument('--scenario_classif', type=str, default='class', help='If doing class based classification or instance based. Valid for core50.')
parser.add_argument('--exp_type', type=str, default='class', help='If doing incremental tasks based on class label or instance label (valid for core50)')
parser.add_argument('--dir_results', type=str, default='./src/novelty_level1/results', help='where to store results')
parser.add_argument('--test_num', type=int, default=-1)
parser.add_argument('--experiment_dir', type=str, default='./src/novelty_level1/experiments')
parser.add_argument('--experiment_filepath', type=str, default=None)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--input_layer_name', type=str, default='base.8', help='input from frozen backbone')
parser.add_argument("--config_file", help="YAML config file. Use --save_config to generate sample configs")
parser.add_argument('--num_threads', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=4)

# Novelty 
parser.add_argument('--holdout_percent', type=float, default=0.25, help='num of images to holdout from train set - to be used as old mixing images for each new task')
parser.add_argument('--max_holdout', type=float, default=0.75, help='max number of holdout images during experiment building')
parser.add_argument('--percent_old', type=float, default=0.5, help='percent of mixing old with respect to new each task: num_old = percent_old*num_new')
parser.add_argument('--novelty_threshold', type=float, default=2.2, help='novelty threshold for binary classification new vs old')
parser.add_argument('--novelty_detector', choices={'dfm', 'odin'})
parser.add_argument('--detector_params', help='Dictionary of parameters for novelty detector')
parser.add_argument('--only_novelty', action='store_true', default=False, help='Only evaluate novelty')


# ----data 
parser.add_argument('--dset_name', type=str, default='core50')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--preload', action='store_true', default=False, help='store the data during dataset init')
parser.add_argument('--num_per_task', type=int, default=1, help='used right now only for for cifar')
parser.add_argument('--num_classes_first', type=int, default=2, help='number classes in first task')
parser.add_argument('--shuffle_order', action='store_true', default=False, help='shuffle order of classes in inc learning')


# ---- Main training 
parser.add_argument('--net_type', type=str, default='resnet18')
parser.add_argument('--fc_sizes', type=str, default='7680,4096')
parser.add_argument('--resnet_base', type=int, default=-1)
parser.add_argument('--head_type', type=str, default='single')
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--batchsize_test', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10, help='print interval')
parser.add_argument('--retrain_base', action='store_true', default=False, help='retrain backbone')



args = parser.parse_args()
if args.config_file:
    args = utils.get_args_from_yaml(args, parser)

print(vars(args))
start = time.time()


torch.set_num_threads(args.num_threads)
pin_memory=False
#================================================================================================================
# 0) ----- Directories and config
utils.makedirectory(args.dir_results)
if args.test_num>=0:
    save_name = 'Test_%d'%(args.test_num)
else:
    save_name = datetime.now().strftime(r'%d%m%y_%H%M%S')
if args.save_name:
    save_name = args.save_name + '_' + save_name
dir_save = '%s/%s/%s/'%(args.dir_results, args.dset_name, save_name)
# check if exists, if yes, overwrite. 
if os.path.exists(dir_save) and os.path.isdir(dir_save):
    shutil.rmtree(dir_save)
utils.makedirectory(dir_save)
# save config of experiment
dict_args = vars(args)
with open('%sconfig_model.json'%(dir_save), 'w') as fp:
    json.dump(dict_args, fp, sort_keys=True, indent=1)


# 1) ----- Dataset 
train_datasets, train_holdout_datasets, test_datasets, list_tasks, dset_prep = call_dataset_holdout(args.dset_name, args.data_dir,
                                            args.experiment_dir, experiment_filepath=args.experiment_filepath, 
                                            experiment_name=args.experiment_name, holdout_percent=args.holdout_percent,  max_holdout=args.max_holdout, scenario=args.scenario, 
                                            scenario_classif=args.scenario_classif, exp_type=args.exp_type, num_per_task=args.num_per_task, num_classes_first=args.num_classes_first, 
                                            shuffle=args.shuffle_order, preload=args.preload)


num_tasks = len(train_datasets)
target_ind = dset_prep['scenario_classif'] 
homog_ind = -2
batchsize = args.batchsize
batchsize_new_types = {0:args.batchsize}
for t in range(1,num_tasks):
    batchsize_new_types[t]= int(0.5*batchsize)


test_loaders = [torch.utils.data.DataLoader(test_datasets[t], batch_size=args.batchsize_test,
                                            shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]

train_loaders = [torch.utils.data.DataLoader(train_datasets[t], batch_size=batchsize_new_types[t],
                                            shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]



# sys.exit()
# 2) ------ Main network (feat extractor + classifier)
fc_sizes = args.fc_sizes.split(",")
fc_sizes = [int(x) for x in fc_sizes]
print('total_classes', dset_prep['total_classes'])
network = clf.Resnet(dset_prep['total_classes'], resnet_arch=args.net_type, FC_layers=fc_sizes,  
            resnet_base=args.resnet_base, multihead_type=args.head_type, base_freeze=not args.retrain_base)
if args.retrain_base is False:
    network.base.train(False)
    print('freeze backbone')

network = network.cuda()
list_layers_extract = [args.input_layer_name]
pooling_factors_layers ={args.input_layer_name:-1}
network_inner = NetworkLatents(network, list_layers_extract, pool_factors=pooling_factors_layers)
print('prepare classifier/feat_extractor', time.time()-start)


# 4) Novelty
novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector, params=args.detector_params)
noveltyResults = novel.save_novelty_results(num_tasks, args.detector_params['score_type'], dir_save)



def num_mix_old_novelty(percent_old, train_dataset, train_holdout_datasets, current_task, list_tasks):

    num_new = train_dataset.__len__()
    num_old_total = sum([train_holdout_datasets[i].__len__() for i in range(current_task)])

    num_old = min(int(percent_old*num_new), num_old_total)
    real_percent_old = num_old/num_new
    print('real_percent_old', real_percent_old)
    # uniform sampling accross old tasks, can be changed after
    num_old_per_task = int(num_old/len(flatten(list_tasks[:current_task])))

    return num_old, num_old_total, num_old_per_task, num_new



def flatten(t):
    return [item for sublist in t for item in sublist]

# ---------------------------------------------------------------
# ---------------------------- Train ----------------------------
# ---------------------------------------------------------------
error_old_to_new=[]
error_new_to_old=[]
start = time.time()
for t in range(num_tasks):
    print('###############################')
    print('######### task %d ############'%(t))
    print('###############################')

    current_task = list_tasks[t]
    print(current_task)

    # continue

    # Evaluate Novelty Detector 
    if t>0:
        print('Get scores for novelty detector')
        num_old, num_old_total, num_old_per_task, num_new = num_mix_old_novelty(args.percent_old, train_datasets[t], train_holdout_datasets, t, list_tasks)

        print('num_old', num_old,  'num_new', num_new, 'num_old_total', num_old_total, 'num_old_per_task', num_old_per_task)


        new_gt, new_scores = novelty_detector.score(network_inner, args.input_layer_name, train_loaders[t])
        print('new_scores', new_scores)

        # old_gt, old_scores = novelty_detector.score(network_inner, args.input_layer_name, train_loaders[0])
        # print('old_scores', old_scores)


        # Review this for corrected scores 
        inds_below_th_new = np.arange(new_scores.shape[0])[new_scores<args.novelty_threshold]
        num_fake_old = inds_below_th_new.shape[0]

        error_new=num_fake_old/new_scores.shape[0]
        error_new_to_old.append(error_new)

        print('error mistaking new data for old: ', error_new, np.mean(np.array(error_new_to_old)))

        old_gt = []
        old_scores = []
        num_samples_old = 0
        for t_old in range(t):
            train_holdout_datasets[t_old].select_random_subset(num_old_per_task)
            num_samples_old += train_holdout_datasets[t_old].__len__()
            t_loader_old = torch.utils.data.DataLoader(train_holdout_datasets[t_old], batch_size=100,
                                                    shuffle=True, num_workers=4)
            gt, scores = novelty_detector.score(network_inner, args.input_layer_name, t_loader_old)
            old_gt.append(gt)
            old_scores.append(scores)


        old_scores = np.array(old_scores).flatten()
        # print('old_scores', old_scores)
        # filtered indices accross all old 
        inds_above_th_old = np.arange(old_scores.shape[0])[old_scores>=args.novelty_threshold]
        num_fake_new = inds_above_th_old.shape[0]
        # inds_above_th_old = np.arange(l.shape[0])[l>=args.novelty_threshold for l in old_scores]
        # num_fake_new = sum([l.shape[0] for l in inds_above_th_old])

        error_old = num_fake_new/num_samples_old
        error_old_to_new.append(error_old)
        print('error mistaking old data for new: ', error_old, np.mean(np.array(error_old_to_new)))

        
        scores_concat = np.concatenate((new_scores, old_scores))
        gt_concat = np.ones(len(scores_concat))
        gt_concat[:len(new_scores)] = 0
        noveltyResults.compute(t, gt_concat, scores_concat)

        

        # get binary classification accuracy with threhsolding
        accuracy, sk_fn, sk_tp, sk_tn, sk_fp = noveltyResults.compute_from_preds(t, scores_concat, args.novelty_threshold, gt_concat)
        

    #### Run to get results 


    # Generate all new data
    print('Generate features for novelty evaluation and coreset')
    current_features = futils.extract_features(network_inner, train_loaders[t], target_ind=target_ind, homog_ind=homog_ind, device=args.device)
   
    # ----append memory
    if t<num_tasks-1:
        print('Update Novelty detector')
        # Update Novelty detector 
        novelty_detector.fit(current_features[0][args.input_layer_name].T, current_features[1])
        

print('Runtime: ', time.time()-start)
