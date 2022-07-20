'''
Main Script for Simple Supervised Instance per instance training core50 or cifar10. 
    Instance level x class level classification
    1. Simple dynamic-buffer replay storing embeddings
    2. quantizing the embeddings and storing the codebooks 

*Now incorporates generalized ODIN and DFM as novelty detectors 

*Introduces the option to finetune backbone for first task or for all tasks 

--------- To run (currently with flag only_novelty which skips training, change in yaml file)---------
===for cifar100:
---fine grained
CUDA_VISIBLE_DEVICES=4 python3 -i ./src/main_supervised.py --config_file ./src/configs/ODD_cifar100_classinc_config.yaml --test_num 0


===for cifar10:
===class incremental (task = all instances for 1 class)
CUDA_VISIBLE_DEVICES=2 python3 -i ./src/main_supervised.py --config_file ./src/configs/ODD_cifar10_classinc_config.yaml --test_num 15


===for svhn:
===class incremental (task = all instances for 1 class)
CUDA_VISIBLE_DEVICES=2 python3 -i ./src/main_supervised.py --config_file ./src/configs/ODD_svhn_classinc_config.yaml --test_num 0








====for core50:
====class incremental (task = all instances for 1 class)
CUDA_VISIBLE_DEVICES=7 python3 -i ./src/main_supervised.py --config_file ./src/configs/ODD_core50_classinc_config.yaml --test_num 15



====core50 instance incremental (task = 1 instance)
CUDA_VISIBLE_DEVICES=4 python3 -i ./src/main_supervised.py --config_file ./src/configs/ODD_core50_instinc_config.yaml --test_num 12

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
import copy
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

import novelty_ODD.novelty_detector as novel 
import novelty_ODD.novelty_eval as novelval 


import feature_extraction.quantization as quant
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents
import feature_extraction.latent_layer_functions as latent


parser = argparse.ArgumentParser(description='Main Script')

#---- general 
parser.add_argument('--scenario', type=str, default='nc', help='If incremental (nc) or joint (joint_nc)')
parser.add_argument('--keep_all_data', action='store_true', default=False, help='keep all data incrementally - baseline')
parser.add_argument('--finetune_backbone', type=str, default='off', help='options=off,first,all')

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
parser.add_argument("--config_file", help="YAML config file. Use --save_config to generate sample configs")
parser.add_argument('--cut_epoch_short', action='store_true', default=False, help='cut epoch short with break in train function')
parser.add_argument('--weight_old', type=float, default=0.5, help='weight for old samples versus new')
parser.add_argument('--num_threads', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--only_novelty', action='store_true', default=False, help='Only evaluate novelty')
parser.add_argument('--only_experiment_gen', action='store_true', default=False, help='Only run to generate experiment setup and save it')
parser.add_argument('--input_layer_name', type=str, default='base.8', help='input from frozen backbone')


parser.add_argument('--weights_pretrained', type=str, default=None)


# ----data 
parser.add_argument('--dset_name', type=str, default='core50')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--preload', action='store_true', default=False, help='store the data during dataset init')
parser.add_argument('--num_per_task', type=int, default=1, help='used right now only for for cifar')
parser.add_argument('--shuffle_order', action='store_true', default=False, help='shuffle order of classes in inc learning')
parser.add_argument('--num_tasks_cifar', type=int, default=20, help='used right now only for for cifar')
parser.add_argument('--type_l_cifar', type=str, default='fine')
parser.add_argument('--num_classes_first', type=int, default=None, help='used right now only for for cifar')



# ----memory for replay
parser.add_argument('--coreset_size', type=int, default=1000, help='Number of embedding images to store. Later switch to GB measure')
parser.add_argument('--coreset_size_MB', type=float, default=None, help='coreset size in Megabytes - optional to coreset_size')
parser.add_argument('--exp_type_latent', type=str, default='random', help='how to pick latent nodes to keep in memory for replay')
parser.add_argument('--percent_latent', type=float, default=0.5, help='percentage of original latent layer output to keep in memory for replay')
parser.add_argument('--latent_layer_name', type=str, default=None, help='latent layer name, ex FC.3')
parser.add_argument('--use_image_as_input', action='store_true', default=False, help='use images instead of fixed embeddings as input to CL classifier')
parser.add_argument('--latent_pooling_factor', type=int, default=None, help='how much to pool latent by. None does no pooling. -1 does global avg pooling')
parser.add_argument('--weight_latent', type=float, default=0.5, help='loss weight for latent')


# ---- Main training 
parser.add_argument('--net_type', type=str, default='resnet18')
parser.add_argument('--fc_sizes', type=str, default='7680,4096')
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
parser.add_argument('--experiment_name_plot', type=str)





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
    if args.only_novelty==True:
        args.skip_train=True
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
datasets_use = dset.call_dataset(args.dset_name, args.data_dir, args.experiment_dir, 
                                                experiment_filepath=args.experiment_filepath, experiment_name=args.experiment_name, num_classes_first=args.num_classes_first,
                                                type_l_cifar=args.type_l_cifar, num_tasks_cifar=args.num_tasks_cifar, 
                                                scenario=args.scenario, keep_all_data=args.keep_all_data, scenario_classif=args.scenario_classif, 
                                                exp_type=args.exp_type, num_per_task=args.num_per_task,
                                                shuffle=args.shuffle_order, online=args.online, preload=args.preload)
if args.keep_all_data==True:
    train_datasets, train_datasets_new_only, test_datasets, list_tasks, dset_prep = datasets_use
else:
    train_datasets, test_datasets, list_tasks, dset_prep = datasets_use

# Flag for stopping if just running experiments 
if args.only_experiment_gen:
    sys.exit()


num_tasks = len(train_datasets)
target_ind = dset_prep['scenario_classif'] 
homog_ind = dset_prep['homog_ind']
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



if args.keep_all_data==True:
    train_loaders_new_only = [torch.utils.data.DataLoader(train_datasets_new_only[t], batch_size=batchsize_new_types[t],
                                            shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]
else:
    train_loaders_new_only = train_loaders

# 2) ------ Main network (feat extractor + classifier)

if len(args.fc_sizes)>0:
    fc_sizes = args.fc_sizes.split(",")
    fc_sizes = [int(x) for x in fc_sizes]
else:
    fc_sizes = []
print('total_classes', dset_prep['total_classes'])
base_freeze=True
# if args.finetune_backbone=='off':
#     base_freeze = True
# else:
#     base_freeze = False
# sys.exit()

network = clf.Resnet(dset_prep['total_classes'], resnet_arch=args.net_type, FC_layers=fc_sizes,  
            resnet_base=args.resnet_base, multihead_type=args.head_type, base_freeze=base_freeze, pretrained_weights=args.weights_pretrained)



# if base_freeze is True:
#     network.base.train(False)
#     print('freeze backbone')


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
    quantizer_dict = {'pq':quantizer, 'num_codebooks':args.num_codebooks, 'codebook_size':args.codebook_size,  \
                            'spatial_feat_dim':spatial_feat_dim, 'num_channels_init':num_channels_feats}
else:
    quantizer_dict=None


# if keeping raw images in coreset, this has to be changed 
if args.coreset_size_MB:
    coreset_size = utils.memory_equivalence(args.coreset_size_MB, num_channels_feats, quantizer_dict=quantizer_dict)
else:
    coreset_size = args.coreset_size
print('coreset_size', coreset_size)
# coreset = mem.CoresetDynamic(coreset_size, feature_extractor, target_ind=target_ind, homog_ind=-2, quantizer_dict=quantizer_dict)

if coreset_size>0 and args.keep_all_data==False:
    if args.use_image_as_input:
        coreset_embedding_name = 'image'
    else:
        coreset_embedding_name = args.input_layer_name

    coreset = mem.CoresetDynamic(coreset_size, input_layer_name=coreset_embedding_name, latent_layers=coreset_latent,\
        quantizer_dict=quantizer_dict, target_ind=target_ind, homog_ind=homog_ind, device=args.device)
else:
    coreset=None

# 3.1) Latents 
if args.latent_layer_name is not None:
    LatentRep = latent.latent_Nodes(exp_type=args.exp_type_latent, p=args.percent_latent)
    criterion_latent = latent.DistanceLossLatents(distance_type='cosine', eps=1e-6)
else:
    LatentRep=None
    criterion_latent=None

# 4) Novelty
detector_params=args.detector_params
detector_params['target_ind']=target_ind
if args.novelty_detector=='odin':
    detector_params['base_network']=network #simple network (not wrapper)
    detector_params['num_classes']=dset_prep['total_classes']
    detector_params['criterion']= nn.CrossEntropyLoss()
    detector_params['num_epochs']=args.num_epochs
    

detector_params['device']=args.device


noveltyResults = novelval.save_novelty_results(num_tasks, args.experiment_name_plot, dir_save)
novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector, params=detector_params)

# ---------------------------------------------------------------
# ---------------------------- Train ----------------------------
# ---------------------------------------------------------------
start = time.time()
patience_lr = int(np.ceil(args.schedule_patience_perepoch*args.num_epochs))

for t in range(num_tasks):
    print('###############################')
    print('######### task %d ############'%(t))
    print('###############################')

    current_task = list_tasks[t]
    print(current_task)
    # continue

    if t>0 and (coreset is not None):
        batchsize_new =  batchsize_new_types[t]
        batchsize_old = batchsize - batchsize_new
        # --- wrap old data features (latent)
        dataset_old = dset.DSET_wrapper_Replay(coreset.coreset_im, coreset.coreset_t, latents=coreset.coreset_latents, transform=tf_coreset)
        loader_old = torch.utils.data.DataLoader(dataset_old, batch_size=batchsize_old,
                                                shuffle=True, num_workers=args.num_workers)
    else:
        batchsize_new = batchsize_new_types[t]
        batchsize_old = 0
        loader_old = None


    if t>0 and (coreset is not None):
        print('Memory for Task', t, Counter(coreset.coreset_t.numpy()))
        print('Memory for Task - instances', t, Counter(coreset.coreset_homog.numpy()))

    # finetuning parameters 
    if (args.finetune_backbone=='first' and t==0) or args.finetune_backbone=='all':
        finetune_backbone_task = True
    else:
        finetune_backbone_task = False


    # Evaluate Novelty Detector 
    if t>0:
        print('Get scores for novelty detector')
        # train_loaders_new_only evaluates only current new data (unseen by classifier/main model). 
        params_score = {'layer':args.input_layer_name, 'feature_extractor':network_inner, 'base_apply_score':True, 'target_ind':target_ind}
        # print('new', next(iter(train_loaders_new_only[t])))
        if args.novelty_detector=='odin':
            novelval.evaluate_odin(t, novelty_detector, noveltyResults, params_score, train_loaders_new_only[t], test_loaders)
        elif args.novelty_detector=='dfm':
            novelval.evaluate_dfm(t, novelty_detector, noveltyResults, params_score, train_loaders_new_only[t], test_loaders)




    if args.skip_train == True:
        pass
    else:
        print('train classifier and (maybe) backbone')
        # ---- Optimizer for training
        if args.novelty_detector=='dfm':  
            optimizer_main = optim.Adam(filter(lambda p: p.requires_grad, network_inner.model.parameters()), lr=args.lr)
            scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, 'min', patience=patience_lr, factor=args.schedule_decay, min_lr=0.00001)
        else:
            optimizer_main = novelty_detector.optimizer
            scheduler_main = novelty_detector.scheduler

        # ----- relative batchsizes between old and new 
        for epoch in range(args.num_epochs):
            print('##########epoch %d###########'%epoch)
            train_main_epoch(epoch, train_loaders[t], loader_old, network_inner, optimizer_main, scheduler_main, OOD_class=novelty_detector,\
                        latent_criterion=criterion_latent, weight_latent=args.weight_latent, feature_name=args.input_layer_name, latent_name=args.latent_layer_name, weight_old=args.weight_old, cut_epoch_short=args.cut_epoch_short,\
                        quantizer=quantizer_dict, target_ind=target_ind, cuda=True, display_freq=args.log_interval, finetune_backbone=finetune_backbone_task)

            test_main(epoch, t, test_loaders, network_inner, file_acc_avg, file_acc_pertask,  \
                OOD_class=novelty_detector, feature_name=args.input_layer_name,target_ind=target_ind, quantizer=quantizer_dict, cuda=True)


    # print(train_loaders[t].dataset.transform)

    # modify the transform of the dataset in case of storing raw images in coreset
    if args.use_image_as_input and (coreset is not None):
        print('modify transform for raw images in coreset')
        temp_train_loader = copy.deepcopy(train_loaders[t])
        # temp_train_loader.dataset.transform = tforms.tf_simple()
        temp_train_loader.dataset.transform = None
        tf_coreset = tforms.tf_additional(args.dset_name)
        add_tf = tforms.TFBatch(tf_coreset)
        generate_raw = True
    else:
        temp_train_loader = train_loaders[t]
        tf_coreset = None
        add_tf = None
        generate_raw = False


    print('Generate features for novelty evaluation and coreset')
    current_features = futils.extract_features(network_inner, temp_train_loader, \
        target_ind=target_ind, homog_ind=homog_ind, device=args.device, use_raw_images=generate_raw, raw_image_transform=add_tf)


    # ----append memory
    if t<num_tasks-1:
        # Update for latent replay 
        if args.latent_layer_name is not None:
            only_input_data = current_features[0].pop(args.input_layer_name) # remove input data 
            if generate_raw:
                raw_input_data = current_features[0].pop(coreset_embedding_name) # remove input data 
            processed_data = LatentRep.process_data(current_features, network_inner)
            criterion_latent.__update__(LatentRep.positions_classes)
            if generate_raw:
                processed_data[0][coreset_embedding_name]=raw_input_data
            else:
                processed_data[0][args.input_layer_name]=only_input_data
        else:
            processed_data = current_features


        if args.novelty_detector=='dfm':
            dfm_x = processed_data[0][args.input_layer_name]
            dfm_y = current_features[1]
            if args.finetune_backbone=='all':
                # call init method again
                novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector, params=detector_params)
                if args.keep_all_data==False and (coreset is not None) and (coreset.coreset_im.shape[0]>0):
                    # means you have a coreset which needs to be used to recompute the DFM fit. 
                    if generate_raw==False:
                        # coreset contains embedding features
                        dfm_x = torch.cat((processed_data[0][args.input_layer_name], coreset.coreset_im), dim=0)
                        dfm_y = torch.cat((current_features[1], coreset.coreset_t), dim=0)
                    else:
                        print('fuse coreset with new data for recomputing DFM')
                        # means you have to extract features from coreset data
                        d_temp = dset.DSET_wrapper_Replay(coreset.coreset_im, coreset.coreset_t, transform=tf_coreset)
                        l_temp = torch.utils.data.DataLoader(d_temp, batch_size=50,
                                                                shuffle=False, num_workers=args.num_workers)
                        coreset_feats_temp = futils.extract_features(network_inner, l_temp, \
                                    target_ind=target_ind, homog_ind=homog_ind, device=args.device, \
                                    use_raw_images=False, raw_image_transform=None)
                        del d_temp, l_temp
                        dfm_x = torch.cat((processed_data[0][args.input_layer_name], coreset_feats_temp[0][args.input_layer_name]), dim=0)
                        dfm_y = torch.cat((current_features[1], coreset_feats_temp[1]), dim=0)
                        

            print('dfm data in complete finetune', dfm_x.shape, dfm_y)
            # fit DFM 
            novelty_detector.fit_total(dfm_x.T, dfm_y)


        # update coreset if applicable
        if coreset is not None:
            print('Append data to memory')
            coreset.append_memory(processed_data, current_task)
            print('coreset_im', coreset.coreset_im.shape)
        

        # sys.exit()


    # save network weights 
    if finetune_backbone_task:
        torch.save(network.state_dict(), '%s/model_saved.pt'%dir_save)
    # sys.exit()
    # if t==2:
    #     sys.exit()


print('Runtime: ', time.time()-start)
