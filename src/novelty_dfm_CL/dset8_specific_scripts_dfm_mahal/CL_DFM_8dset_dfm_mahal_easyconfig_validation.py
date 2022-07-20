'''
## Adds validation set to set threshold for novelty

# incDFM 1-class Incremental Loop with Novelty prediction and study of error propagation

All incomming data (new task) will be mixture of unseen old (iid) + new 

Points to note:
1. Measure error propagation with Novelty prediction and pseudolabeling as novel/old 
2. Use predicted novel samples to update DFM with unseen label (n+1) but do not touch old parameters (for n or less)
3. incremental continual novelty detection 

BASELINES: ODIN, Mahalanobis
DSETS: cifar10

** For now, other improvements will probably rely on having better embedding representations, most likely through some degree of finetuning while preventing overfitting (due to few data)

'''
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib import colors as mcolors
import torch
import random
import time
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA, FastICA
import itertools
import copy
import yaml
import pickle

from argparse import Namespace

sys.path.append('../')
sys.path.append('../../')

import tforms
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents
# import classifier as clf



import novelty_dfm_CL.novelty_utils as novelu
import novelty_dfm_CL.novelty_eval as novelval
import novelty_dfm_CL.incDFM_w_validation as novelinc
import novelty_dfm_CL.novelty_detector as novel


import memory as mem
import utils
import datasets_utils as dsetutils
import novelty_dfm_CL.classifier as clf

import novelty_dfm_CL.scoring_multi_threshold as ThScores
import novelty_dfm_CL.novelty_dataset_wrappers as dwrap

import novelty_dfm_CL.dset8_specific_scripts_dfm_mahal.datasets_holdout_validation_8dset_dfm_mahal as dseth
import argparse

## For optional Arguments 
parser = argparse.ArgumentParser(description='incDFM and baselines')
parser.add_argument('--general_config_path', type=str, default='../configs_new/incDFM_8dset_dfm_mahal_CL_modifiable_config.yaml')
args_command = parser.parse_args()


with open(args_command.general_config_path) as fid:
    args = Namespace(**yaml.load(fid, Loader=yaml.SafeLoader))
    
args = utils.results_saving_setup(args)
if args.num_tasks<0:
    args.num_tasks=8
num_tasks = args.num_tasks

args = utils.save_ood_config_simple(args)

# setup 
utils.seed_torch(args.seed)
torch.set_num_threads(args.num_threads)
pin_memory=False


# ------------------------------------------------------------------------------------------


start=time.time()
# 1) ----- Dataset 
if not hasattr(args, 'experiment_filepath'):
    args.experiment_filepath = None # have dictionary for defaults???
if not hasattr(args, 'experiment_name'):
    args.experiment_name = None # have dictionary for defaults??




datasets_use = dseth.call_8dset_holdout_w_validation(args.data_dir, args.experiment_dir, num_tasks=args.num_tasks,\
                                        experiment_filepath=args.experiment_filepath, experiment_name=args.experiment_name, 
                                        holdout_percent=args.holdout_percent,  val_holdout=args.val_percent,
                                        )

train_datasets, train_holdout_datasets, val_datasets, test_datasets, list_tasks, dset_prep = datasets_use
list_tasks_targets = list_tasks

   
old_dsets = train_holdout_datasets
old_dsets_test = test_datasets
use_old_dsets = copy.deepcopy(train_holdout_datasets)
use_old_dsets_test = copy.deepcopy(test_datasets)
 

test_loaders = [torch.utils.data.DataLoader(test_datasets[t], batch_size=args.batchsize_test,
                                                shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]



args.dset_prep = dset_prep

# ------------------------------------------------------------------------------------------
print('list_tasks_targets', list_tasks_targets)
# sys.exit()

num_old_per_task = {}
num_old_per_task_test = {}
ratio_per_task={}
for t_w in range(1,num_tasks):
    print('compute ratios data')
    # how much to mix from old and new in one task
    num_old, num_new, num_old_per_task_pt = novelu.num_mix_old_novelty(args.percent_old_mix, train_datasets[t_w], old_dsets[:t_w], t_w, list_tasks_targets[:t_w])
    print('Task', t_w, 'num_old', num_old,  'num_new', num_new, 'num_old_per_task', num_old_per_task_pt)
    num_old_per_task[t_w] = num_old_per_task_pt
    ratio_per_task[t_w] = num_new/(num_old+num_new)

for t_w in range(1,num_tasks):
    print('compute ratios data test')
    # how much to mix from old and new in one task
    num_old_test, num_new_test, num_old_per_task_pt_test = novelu.num_mix_old_novelty(args.percent_old_mix, test_datasets[t_w], old_dsets_test[:t_w], t_w, list_tasks_targets[:t_w])
    print('Task', t_w, 'num_old', num_old_test,  'num_new', num_new_test, 'num_old_per_task', num_old_per_task_pt_test)
    num_old_per_task_test[t_w] = num_old_per_task_pt_test
print('Data set up', time.time()-start)

# sys.exit()

print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())


# 2) ------ Network 
if len(args.fc_sizes)>0:
    fc_sizes = args.fc_sizes.split(",")
    fc_sizes = [int(x) for x in fc_sizes]
else:
    fc_sizes = []

network = clf.Resnet(dset_prep['total_classes'], resnet_arch=args.net_type, FC_layers=fc_sizes, base_freeze=True)
network = network.to(args.device)
dfm_inputs = args.dfm_layers_input.split(",")
dfm_layers_factors = str(args.dfm_layers_factors)
dfm_layers_factors = dfm_layers_factors.split(',')
dfm_inputs_factors = {}
for n in range(len(dfm_inputs)):
    dfm_inputs_factors[dfm_inputs[n]]=int(dfm_layers_factors[n]) #adaptive pooling 
network_inner = NetworkLatents(network, dfm_inputs, pool_factors=dfm_inputs_factors)
print('Network set up', time.time()-start)


# x= torch.rand((50,3,224,224)).to(args.device)
# print(network.base(x).shape)
# # print(futils.layer_names(network))


# 3) ---- Novelty

total_classes = dset_prep['total_classes']
args.detector_params['target_ind']=dset_prep['scenario_classif']
args.detector_params['device']=args.device

noveltyResults = novelval.save_novelty_results(num_tasks, args.experiment_name_plot, args.dir_save)
noveltyResults_test = novelval.save_novelty_results_test(num_tasks, args.experiment_name_plot, args.dir_save)
novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector_name, params=args.detector_params)
print('Novelty Detector set up', time.time()-start, args.novelty_detector_name)





# 4) ---- Memory (if applicable for recomputing DFM per task)
# Use raw images in memory
coreset = None
if args.novelty_detector_name=='mahal' and args.coreset_size>0:
    if args.use_image_as_input:
        raw_sizes = {'svhn':32, 'cifar10':32, 'cifar100':32, 'emnist':32, 'inaturalist19':224, 'inaturalist21':224}
        input_size = 3*(raw_sizes[args.dset_name]**2)
    else:
        input_size = network.feat_size  
    if args.coreset_size>0:
        coreset_size = args.coreset_size
        coreset = mem.CoresetDynamic(coreset_size, target_ind= dset_prep['scenario_classif'], \
            homog_ind=dset_prep['homog_ind'], device=args.device)
print('Memory set up', time.time()-start)


loader_old = None

# sys.exit()



# ----------------------------------------------------------------------------------------
start = time.time()

per_task_results = {'scores':{}, 'scores_dist':{}, 'gt_novelty':{}, 'gt':{}, 'inds':{}, 'threshold':{}, 'preds':{}}

per_task_results = Namespace(**per_task_results)

num_classes = 0
for t in range(num_tasks):
    print('###############################')
    print('######### task %d ############'%(t))
    print('###############################')
    current_task = list_tasks_targets[t]

    batchsize_new = args.batchsize

    if t>0:
        
        # 1) get appropriate "current data"

        new_data = train_datasets[t]

        # mix old (holdout) with new data for "current task"    
        print('Get scores for novelty detector')
        for t_old in range(t):
            use_old_dsets[t_old].select_random_subset(num_old_per_task[t][t_old])
            
        current_dset = dwrap.CurrentTask(new_data, use_old_dsets[:t], use_coarse=True)
        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)
        

        for t_old_ in range(t):
            use_old_dsets_test[t_old_].select_random_subset(num_old_per_task_test[t][t_old_])
            
        # print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())

        test_old_plus_novel_dset = dwrap.CurrentTask(test_datasets[t], use_old_dsets_test[:t], use_coarse=True)
        test_old_plus_novel_loader = torch.utils.data.DataLoader(test_old_plus_novel_dset, batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)
        
        # print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())

        
        params_score = {'layer':args.dfm_layers_input,'feature_extractor':network_inner, \
            'base_apply_score':True, 'target_ind':dset_prep['scenario_classif'], 'device': args.device, 'current_task':t}
        
        
        # make validation wrapper 
        val_loader = torch.utils.data.DataLoader(dwrap.ValidationDset(val_datasets[:t], use_coarse=True), \
                                                batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)


        if args.threshold_type=='simple':
            # 2) Threshold Novel/Old - Binary prediction 

            results_novelty = novelval.evaluate_simple_CL(t, novelty_detector, noveltyResults, params_score, \
                current_loader, args.novelty_detector_name)

                        
            # have function here to estimate the threshold ..... 
            threshold_estimated_novel = novelinc.estimate_threshold_from_val_simple(val_loader, args.percentile_val_threshold, \
                novelty_detector, params_score, args.novelty_detector_name)   
            

            # threshold_estimated_novel TODO
            inds_pred_novel,_ = novelinc.Threshold_n_Select_Simple(t, results_novelty, threshold_estimated_novel, \
                                th_percentile_confidence=args.th_percentile_confidence,  metric_confidence=(args.metric_confidence, args.metric_confidence_direction_novel))
            
            acc_overall, acc_novel, prec_novel, recall_novel = noveltyResults.compute_accuracies(t, 0, inds_pred_novel, results_novelty.gt_novelty)

            print('Binary Pseudolabeling - acc_overall: %.4f, acc_novel: %.4f, prec_novel: %.4f, recall_novel: %.4f' %(acc_overall, acc_novel, prec_novel, recall_novel))
            
            novelval.evaluate_simple_CL_test(t, novelty_detector, noveltyResults_test, params_score, test_old_plus_novel_loader, args.novelty_detector_name)

        else:
            args.params_score = Namespace(**params_score)
            args.num_samples = current_dset.__len__()
            args.w = args.w_score
            args.current_old_new_ratio = ratio_per_task[t]
            args.validation_iid_loader = val_loader
            # Threshold_n_Select_Iters(params)
            if args.alg_dfm == 'simple':
                SelectTh = novelinc.Threshold_n_Select_Iters(args)
            # elif args.alg_dfm == 'tug':
            #     SelectTh = novelinc.Threshold_Tug_incDFM(args)
                
            inds_pred_novel,_ = SelectTh.select_novel(t, current_dset, novelty_detector, noveltyResults)
            results_novelty = SelectTh.evaluate_CL(t, current_dset, novelty_detector, noveltyResults)
            
            results_novelty_test = SelectTh.evaluate_CL(t, test_old_plus_novel_dset, novelty_detector, noveltyResults_test)

            
        # to evaluate final auroc/aupr use test set with the same mixing ratio
        # if t==1:
        #     inds_pred_novel=np.array([]).astype(int)

        # store results per task 
        per_task_results.scores_dist[t] = results_novelty.scores_dist
        per_task_results.scores[t] = results_novelty.scores
        per_task_results.gt_novelty[t] = results_novelty.gt_novelty
        per_task_results.gt[t] = results_novelty.gt
        per_task_results.inds[t] = results_novelty.dset_inds
        per_task_results.threshold[t] = args.percentile_val_threshold

        preds_array = np.zeros(results_novelty.scores.shape)
        preds_array[inds_pred_novel]=1
        per_task_results.preds[t] = preds_array
        
        
        print('inds_pred_novel', inds_pred_novel.shape)



        # 3) Pseudolabel "Novel" Samples
        if args.prediction_propagation:
            if inds_pred_novel.shape[0]>0:
                print('Use Predicted Labels for Propagation/DFM Fit')
                loader_new = torch.utils.data.DataLoader(dwrap.NovelTask(t, num_classes, current_dset, pred_novel_inds=inds_pred_novel, \
                    use_coarse=True), batch_size=batchsize_new,
                                                        shuffle=True, num_workers=args.num_workers)
            else:
                print('No indices were predicted as novel --> skip to next task')
                loader_new=None
        else:
            inds_gt_novel = np.where(current_dset.novelty_y==1)[0]
            print('Use Ground Truth Labels for Propagation/DFM Fit', inds_gt_novel.shape[0])
            loader_new = torch.utils.data.DataLoader(dwrap.NovelTask(t, num_classes, current_dset, pred_novel_inds=inds_gt_novel, \
                use_coarse=True), batch_size=batchsize_new,
                                                    shuffle=True, num_workers=args.num_workers)
            
    else:
        loader_new = torch.utils.data.DataLoader(dwrap.NovelTask(t, num_classes, train_datasets[t], use_coarse=True), \
            batch_size=batchsize_new, shuffle=True, num_workers=args.num_workers)
        


    if loader_new is not None:


        # ----Update memory and Novelty detector
        if t<num_tasks-1:
                #  call training loop with one liner 
            args, temp_train_loader = novelu.temporary_loader_novelty(args, loader_new, coreset)


            print('Generate features for novelty evaluation and coreset')
            # target_ind, homog_ind=1 because they are wrapped by NovelTask 
            processed_data = futils.extract_features(network_inner, temp_train_loader, \
                target_ind=1, homog_ind=1, 
                device=args.device, use_raw_images=args.use_image_as_input, raw_image_transform=args.add_tf)
        
            
            
            # ------ update coreset (if applicable)
            if t<num_tasks-1:
                if coreset is not None:
                    print('Append data to memory')
                    coreset.append_memory(processed_data, current_task)
                    print('coreset_im', coreset.coreset_im.shape)


            if 'dfm' in args.novelty_detector_name:
                dfm_x = processed_data[0][args.dfm_layers_input]
                dfm_y = processed_data[1]
                if args.finetune_backbone=='all':
                    novelty_detector, dfm_x, dfm_y = novelu.reprocess_data_novelty(args, dfm_x, dfm_y, network_inner, coreset)
                novelty_detector.fit_total(dfm_x.T, dfm_y)

            elif args.novelty_detector_name=='mahal':
                if t>0:
                    # join processed_data with coreset_data
                    mahal_x = torch.cat((processed_data[0][args.dfm_layers_input],coreset.coreset_im), dim=0)
                    mahal_y = torch.cat((processed_data[1],coreset.coreset_t))
                else:
                    mahal_x = processed_data[0][args.dfm_layers_input]
                    mahal_y = processed_data[1]
                novelty_detector.fit_total(mahal_x.T, mahal_y)

        
            print('finish fit - dfm/mahal')
    # sys.exit()

    num_classes = num_classes + len(current_task)

    # print('PCA mats', novelty_detector.pca_mats)
    # if t==2:
    #     print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())
    #     sys.exit()

# ------------------------------------------------------------------------------------------------


with open('%s/results_tasks.pickle'%(args.dir_save), 'wb') as handle:
    pickle.dump(per_task_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
