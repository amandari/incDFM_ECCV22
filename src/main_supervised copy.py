'''
Main Script for Simple Supervised Instance per instance training core50 or cifar10. 
    Instance level x class level classification
    1. Simple dynamic-buffer replay storing embeddings
    2. quantizing the embeddings and storing the codebooks 


--------- To run (currently with flag only_novelty which skips training, change in yaml file)---------
===for cifar10:
===class incremental (task = all instances for 1 class)
CUDA_VISIBLE_DEVICES=1 python3 -i ./src/main_supervised.py --config_file ./src/configs/dfm_cifar10_classinc_config.yaml

====for core50:
====class incremental (task = all instances for 1 class)
CUDA_VISIBLE_DEVICES=1 python3 -i ./src/main_supervised.py --config_file ./src/configs/dfm_core50_classinc_config.yaml

====instance incremental (task = 1 instance)
CUDA_VISIBLE_DEVICES=0 python3 -i ./src/main_supervised.py --config_file ./src/configs/dfm_core50_instinc_config.yaml


'''

# TODO maybe edit cifar10 training parameters like lr with patience etc or number of epochs per task 

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

import utils
import datasets as dset
import classifier as clf
import memory as mem
from train import train_main_epoch
from test import test_main
import tforms

import novelty_OOD.novelty_detector as novel 
import novelty_OOD.novelty_eval as novelval 


import feature_extraction.quantization as quant
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents
import feature_extraction.latent_layer_functions as latent

parser = argparse.ArgumentParser(description='Main Script')

#---- general 
parser.add_argument('--scenario', type=str, default='nc', help='If incremental (nc) or joint (joint_nc)')
parser.add_argument('--scenario_classif', type=str, default='class', help='If doing class based classification or instance based. Valid for core50.')
parser.add_argument('--exp_type', type=str, default='class', help='If doing incremental tasks based on class label or instance label (valid for core50)')
parser.add_argument('--dir_results', type=str, default='./Results', help='where to store results')
parser.add_argument('--test_num', type=int, default=-1)
parser.add_argument('--experiment_dir', type=str, default='./experiments')
parser.add_argument('--experiment_filepath', type=str, default=None)
parser.add_argument('--experiment_name', type=str, default=None)
parser.add_argument('--file_acc_pertask', type=str, default='acc_pertask', help='test accuracy per each cummulative tasks')
parser.add_argument('--file_acc_avg', type=str, default='acc_avg', help='test average accuracy over all cummulative tasks')
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--skip_train', action='store_true', default=False, help='skip training in task loop')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--input_layer_name', type=str, default='base.8', help='input from frozen backbone')
parser.add_argument("--config_file", help="YAML config file. Use --save_config to generate sample configs")
parser.add_argument('--cut_epoch_short', action='store_true', default=False, help='cut epoch short with break in train function')
parser.add_argument('--weight_old', type=float, default=0.5, help='weight for old samples versus new')
parser.add_argument('--num_threads', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--only_novelty', action='store_true', default=False, help='Only evaluate novelty')
parser.add_argument('--only_experiment_gen', action='store_true', default=False, help='Only run to generate experiment setup and save it')


# ----data 
parser.add_argument('--dset_name', type=str, default='core50')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--preload', action='store_true', default=False, help='store the data during dataset init')
parser.add_argument('--num_per_task', type=int, default=1, help='used right now only for for cifar')
parser.add_argument('--shuffle_order', action='store_true', default=False, help='shuffle order of classes in inc learning')


# ----memory for replay
parser.add_argument('--coreset_size', type=int, default=1000, help='Number of embedding images to store. Later switch to GB measure')
parser.add_argument('--coreset_size_MB', type=float, default=None, help='coreset size in Megabytes - optional to coreset_size')
parser.add_argument('--exp_type_latent', type=str, default='random', help='how to pick latent nodes to keep in memory for replay')
parser.add_argument('--percent_latent', type=float, default=0.5, help='percentage of original latent layer output to keep in memory for replay')
parser.add_argument('--latent_layer_name', type=str, default=None, help='latent layer name, ex FC.3')
parser.add_argument('--latent_pooling_factor', type=int, default=None, help='how much to pool latent by. None does no pooling. -1 does global avg pooling')
parser.add_argument('--weight_latent', type=float, default=0.5, help='loss weight for latent')



# ---- Main training 
parser.add_argument('--net_type', type=str, default='resnet18')
parser.add_argument('--fc_sizes', type=str, default='7680,4096')
parser.add_argument('--retrain_base', action='store_true', default=False, help='leave backbone plastic.')
parser.add_argument('--quantize_buffer', action='store_true', default=False, help='if quantizing buffer')
parser.add_argument('--num_codebooks', type=int, default=32, help='number of codebooks for PQ, .i.e. number of centroids ')
parser.add_argument('--codebook_size', type=int, default=256, help=' size of each codebook for PQ, .i.e. dimension of encoded quantized input ')

parser.add_argument('--resnet_base', type=int, default=-1)
parser.add_argument('--head_type', type=str, default='single')
parser.add_argument('--batchsize', type=int, default=50)
parser.add_argument('--online', action='store_true', default=False, help='if true, doing online training. One input at a time')
parser.add_argument('--batchsize_test', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10, help='print interval')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train each PSP key for a given task')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01)')
parser.add_argument('--schedule_patience_perepoch', type=float, default=0.25, help='for scheduler lr decrease after int(np.ceil(num_epochs*shedule_patience_perepoch)) epochs')
parser.add_argument('--schedule_decay', type=float, default=0.5, help='decay multiplier for lr in scheduling')

# TODO knowledge distillation with replay
parser.add_argument('--temperature', type=float, default=1, help='temperature for knowledge distillation of old samples; 1 = No temperature effect')
parser.add_argument('--distill_old', action='store_true', default=False, help='perform knowledge distillation of old samples')

# Novelty Detection
parser.add_argument('--novelty_detector', choices={'dfm', 'odin'})
parser.add_argument('--detector_params', help='Dictionary of parameters for novelty detector')


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
    if args.only_novelty==True:
        args.skip_train=True
        save_name = 'Novelty_'+datetime.now().strftime(r'%d%m%y_%H%M%S')
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
file_acc_avg = '%s/%s.txt'%(dir_save, args.file_acc_avg)
file_acc_pertask = '%s/%s.txt'%(dir_save, args.file_acc_pertask)



# 1) ----- Dataset 
train_datasets, test_datasets, list_tasks, dset_prep = dset.call_dataset(args.dset_name, args.data_dir, args.experiment_dir, 
                                                experiment_filepath=args.experiment_filepath, experiment_name=args.experiment_name, 
                                                scenario=args.scenario, scenario_classif=args.scenario_classif, exp_type=args.exp_type, num_per_task=args.num_per_task,
                                                shuffle=args.shuffle_order, online=args.online, preload=args.preload)

# Flag for stopping if just running experiments 
if args.only_experiment_gen:
    sys.exit()


num_tasks = len(train_datasets)
target_ind = dset_prep['scenario_classif'] 
homog_ind = -2
# print('target_ind', target_ind)

if args.online:
    batchsize = 1
else:
    batchsize = args.batchsize
batchsize_new_types = {0:args.batchsize}

for t in range(1,num_tasks):
    batchsize_new_types[t]= int(0.5*batchsize)

base_init_train_loader = torch.utils.data.DataLoader(train_datasets[0], batch_size=args.batchsize_test,
                                            shuffle=True, num_workers=args.num_workers)
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
# TODO add later option of more latent layers
if args.latent_layer_name is not None:
    list_layers_extract = [args.input_layer_name, args.latent_layer_name]
    pooling_factors_layers ={args.input_layer_name:-1, args.latent_layer_name: args.latent_pooling_factor}
    coreset_latent=[args.latent_layer_name]
else:
    list_layers_extract = [args.input_layer_name]
    pooling_factors_layers ={args.input_layer_name:-1}
    coreset_latent=[]

network_inner = NetworkLatents(network, list_layers_extract, pool_factors=pooling_factors_layers)
print('prepare classifier/feat_extractor', time.time()-start)



# 3) ----- Main Memory  
# TODO remove hardcoded params that refer to resnet18 current feature extractor parameters
num_channels_feats = 512
spatial_feat_dim = -1
if args.quantize_buffer:
    new_data = futils.extract_features(network_inner, base_init_train_loader, target_ind=target_ind, homog_ind=homog_ind, device=args.device)
    feats_base_init = new_data[0][args.input_layer_name]
    print('pq input', feats_base_init.shape)
    quantizer = quant.fit_pq(feats_base_init, num_channels_feats, args.num_codebooks,
            args.codebook_size, spatial_feat_dim=spatial_feat_dim)
    print('Extracted features for quantizer and finished training PQ')
    quantizer_dict = {'pq':quantizer, 'num_codebooks':args.num_codebooks, 'codebook_size':args.codebook_size,  'spatial_feat_dim':spatial_feat_dim, 'num_channels_init':num_channels_feats}
else:
    quantizer_dict=None

if args.coreset_size_MB:
    coreset_size = utils.memory_equivalence(args.coreset_size_MB, num_channels_feats, quantizer_dict=quantizer_dict)
else:
    coreset_size = args.coreset_size
print('coreset_size', coreset_size)
# coreset = mem.CoresetDynamic(coreset_size, feature_extractor, target_ind=target_ind, homog_ind=-2, quantizer_dict=quantizer_dict)
coreset = mem.CoresetDynamic(coreset_size, input_layer_name=args.input_layer_name, latent_layers=coreset_latent,\
    quantizer_dict=quantizer_dict, target_ind=target_ind, homog_ind=homog_ind, device=args.device)


# 3.1) Latents 
if args.latent_layer_name is not None:
    LatentRep = latent.latent_Nodes(exp_type=args.exp_type_latent, p=args.percent_latent)
    criterion_latent = latent.DistanceLossLatents(distance_type='cosine', eps=1e-6)
else:
    LatentRep=None
    criterion_latent=None

# 4) Novelty
novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector, params=args.detector_params)
noveltyResults = novelval.save_novelty_results(num_tasks, args.detector_params['score_type'], dir_save)

# ---------------------------------------------------------------
# ---------------------------- Train ----------------------------
# ---------------------------------------------------------------
start = time.time()
accuracy_main = []
patience_lr = int(np.ceil(args.schedule_patience_perepoch*args.num_epochs))

for t in range(num_tasks):
    print('###############################')
    print('######### task %d ############'%(t))
    print('###############################')

    current_task = list_tasks[t]
    print(current_task)
    # continue

    if t>0:
        batchsize_new =  batchsize_new_types[t]
        batchsize_old = batchsize - batchsize_new
        # --- wrap old data features (latent)
        dataset_old = dset.DSET_wrapper_Replay(coreset.coreset_im, coreset.coreset_t, latents=coreset.coreset_latents)
        loader_old = torch.utils.data.DataLoader(dataset_old, batch_size=batchsize_old,
                                                shuffle=True, num_workers=args.num_workers)
    else:
        batchsize_new = batchsize_new_types[t]
        batchsize_old = 0
        loader_old = None

    accuracy_main.append([])
    if t>0:
        print('Memory for Task', t, Counter(coreset.coreset_t.numpy()))
        print('Memory for Task - instances', t, Counter(coreset.coreset_homog.numpy()))


    # Evaluate Novelty Detector 
    if t>0:
        print('Get scores for novelty detector')
        new_train_loader = train_loaders[t]
        new_gt, new_scores = novelty_detector.score(network_inner, args.input_layer_name, new_train_loader)
        for  task_idx, old_test_loader in enumerate(test_loaders[:t]):
            gt, scores = novelty_detector.score(network_inner, args.input_layer_name, old_test_loader)
            if task_idx == 0:
                old_gt = gt
                old_scores = scores
            else:
                old_gt = np.concatenate((old_gt, gt))
                old_scores = np.concatenate((old_scores, scores))
        print('Task {}, old scores length {}'.format(t, len(old_scores)))
        scores_concat = np.concatenate((new_scores, old_scores))
        gt_concat = np.ones(len(scores_concat))
        gt_concat[len(new_scores):] = 0
        noveltyResults.compute(t, gt_concat, scores_concat)

    
    if args.skip_train == True:
        pass
    else:
        # ---- Optimizer for training 
        optimizer_main = optim.Adam(filter(lambda p: p.requires_grad, network_inner.model.parameters()), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, 'min', patience=patience_lr, factor=args.schedule_decay, min_lr=0.00001)


        # ----- relative batchsizes between old and new 
        for epoch in range(args.num_epochs):
            print('##########epoch %d###########'%epoch)
            train_main_epoch(epoch, train_loaders[t], loader_old, network_inner, optimizer_main,\
                        latent_criterion=criterion_latent, weight_latent=args.weight_latent, feature_name=args.input_layer_name, latent_name=args.latent_layer_name, weight_old=args.weight_old, cut_epoch_short=args.cut_epoch_short,\
                        quantizer=quantizer_dict, target_ind=target_ind, cuda=True, display_freq=args.log_interval, base_freeze=not args.retrain_base)

            accuracy_test = test_main(epoch, t, test_loaders, network_inner, accuracy_main, file_acc_avg, file_acc_pertask, feature_name=args.input_layer_name,target_ind=target_ind, quantizer=quantizer_dict, cuda=True)


    print('Generate features for novelty evaluation and coreset')
    current_features = futils.extract_features(network_inner, train_loaders[t], target_ind=target_ind, homog_ind=homog_ind, device=args.device)

   
    # ----append memory
    if t<num_tasks-1:
        print('Append data to memory')
        # Update for latent replay 
        only_input_data = current_features[0].pop(args.input_layer_name) # remove input data 
        if args.latent_layer_name is not None:
            processed_data = LatentRep.process_data(current_features, network_inner)
            criterion_latent.__update__(LatentRep.positions_classes)
        else:
            processed_data = current_features
        processed_data[0][args.input_layer_name]=only_input_data
        coreset.append_memory(processed_data, current_task)
        print('coreset_im', coreset.coreset_im.shape)
        # Update Novelty detector 
        novelty_detector.fit(only_input_data.T, current_features[1])
        


    # sys.exit()
    # if t==3:
    #     break


print('Runtime: ', time.time()-start)
