'''
Should be identical code to CL_DFM_1class.ipynb
Not meant as a finalized script
\--------------------------------------------
DFM 1-class Incremental Loop with Novelty prediction and study of error propagation

All incomming data (new task) will be mixture of unseen old (iid) + new 

Points to note:
1. Measure error propagation with Novelty prediction and pseudolabeling as novel/old 
2. Use predicted novel samples to update DFM with unseen label (n+1) but do not touch old parameters (for n or less)
3. Experiment with doing prediction all at once versus in rounds (batches) guided by thresholded novelty scores or confidence values 
    3.1. Option also to discard low confidence or very low-score samples. 
4. Have option to use test set for unseen old data or to use hold-out train set 
    4.1. In the latter case, performance will also be affected because less data will be available for fitting each class originally

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
import tforms
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents
# import classifier as clf


import novelty_dfm_CL.novelty_detector as novel
import novelty_dfm_CL.novelty_eval as novelval 
import novelty_dfm_CL.classifier as clf
import novelty_dfm_CL.novelty_utils as novelu


import memory as mem
import utils
import datasets as dset
import novelty_dfm_CL.datasets_holdout as dseth
import datasets_utils as dsetutils



from novelty_dfm_CL.novelty_eval import acuracy_report, scores_metrics_results



# ---------------------- Read params 


utils.seed_torch(0)
general_config_path = '/home/amandari/CodeDev/ProjIntel/src/configs/DFM_CL_1class_loop_script.yaml' # path to file 
with open(general_config_path) as fid:
        args = Namespace(**yaml.load(fid, Loader=yaml.SafeLoader))

torch.set_num_threads(args.num_threads)
pin_memory=False
# set up save
args = utils.config_saving(args)


# ---------------------- Data 

start=time.time()
# 1) ----- Dataset 
if not hasattr(args, 'experiment_filepath'):
    args.experiment_filepath = None # have dictionary for defaults???
if not hasattr(args, 'experiment_name'):
    args.experiment_name = None # have dictionary for defaults??


if args.holdout:
    # load holdout dset
    datasets_use = dseth.call_dataset_holdout(args.dset_name, args.data_dir, args.experiment_dir, experiment_filepath=args.experiment_filepath, experiment_name=args.experiment_name, 
                                            holdout_percent=args.holdout_percent,  max_holdout=args.max_holdout, scenario=args.scenario, 
                                            scenario_classif=args.scenario_classif, exp_type=args.exp_type, num_per_task=args.num_per_task, num_classes_first=args.num_classes_first, 
                                            shuffle=args.shuffle_order, preload=False, keep_all_data=args.keep_all_data)

    if args.keep_all_data==True:
        train_datasets, train_holdout_datasets, train_datasets_new_only, test_datasets, list_tasks, dset_prep = datasets_use
    else:
        train_datasets, train_holdout_datasets, test_datasets, list_tasks, dset_prep = datasets_use

else:
    datasets_use = dset.call_dataset(args.dset_name, args.data_dir, args.experiment_dir, experiment_filepath=args.experiment_filepath, experiment_name=args.experiment_name, 
                                                    num_classes_first=args.num_classes_first, keep_all_data=args.keep_all_data, 
                                                    scenario=args.scenario, scenario_classif=args.scenario_classif, 
                                                    exp_type=args.exp_type, num_per_task=args.num_per_task,
                                                    type_l_cifar=args.type_l_cifar, num_tasks_cifar=args.num_tasks_cifar, 
                                                    shuffle=args.shuffle_order, preload=False)
    if args.keep_all_data==True:
        train_datasets, train_datasets_new_only, test_datasets, list_tasks, dset_prep = datasets_use
    else:
        train_datasets, test_datasets, list_tasks, dset_prep = datasets_use


args.dset_prep = dset_prep
if args.num_tasks>0:
    num_tasks = args.num_tasks
else:
    num_tasks = len(train_datasets)

test_loaders = [torch.utils.data.DataLoader(test_datasets[t], batch_size=args.batchsize_test,
                                                shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]


# ---------------------------------------Networks setup, etc

if args.holdout:
    old_dsets = train_holdout_datasets
    use_old_dsets = copy.copy(train_holdout_datasets)
else:
    old_dsets = test_datasets
    use_old_dsets = copy.copy(test_datasets)


num_old_per_task = {}
for t_w in range(1,num_tasks):
    print('Get scores for novelty detector')
    # how much to mix from old and new in one task
    num_old, num_new, num_old_per_task_pt = novelu.num_mix_old_novelty(args.percent_old_mix, train_datasets[t_w], old_dsets[:t_w], t_w, list_tasks)
    print('Task', t_w, 'num_old', num_old,  'num_new', num_new, 'num_old_per_task', num_old_per_task_pt)
    num_old_per_task[t_w] = num_old_per_task_pt

print('Data set up', time.time()-start)


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




# 3) ---- Novelty
args.detector_params['target_ind']=dset_prep['scenario_classif']
args.detector_params['device']=args.device
noveltyResults = novelval.save_novelty_results(num_tasks, args.experiment_name_plot, args.dir_save)
novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector_name, params=args.detector_params)
print('Novelty Detector set up', time.time()-start)




# 4) ---- Memory (if applicable for recomputing DFM per task)
# Use raw images in memory
coreset = None
if args.coreset_size>0:
    if args.use_image_as_input:
        raw_sizes = {'svhn':32, 'cifar10':32, 'cifar100':32}
        input_size = 3*(raw_sizes[args.dset_name]**2)
    else:
        input_size = network.feat_size  
    if args.coreset_size>0 and args.keep_all_data==False:
        if args.coreset_size_MB>0:
            coreset_size = utils.memory_equivalence(args.coreset_size_MB, input_size, quantizer_dict=None)
        else:
            coreset_size = args.coreset_size
        coreset = mem.CoresetDynamic(coreset_size, target_ind= dset_prep['scenario_classif'], homog_ind=dset_prep['homog_ind'], device=args.device)
print('Memory set up', time.time()-start)






# -------------------------------------------------------CL training + evaluation 



start = time.time()
args.patience_lr = int(np.ceil(args.schedule_patience_perepoch*args.num_epochs))

per_task_results = {'scores':{}, 'scores_dist':{}, 'gt_novelty':{}, 'gt':{}, 'inds':{}, 'threshold':{}, 'preds':{}}

per_task_results = Namespace(**per_task_results)

num_classes = 0
for t in range(num_tasks):
    print('###############################')
    print('######### task %d ############'%(t))
    print('###############################')
    current_task = list_tasks[t]


    # ------- Set up data 
    if t>0 and (coreset is not None):
        batchsize_new =  int(args.batchsize*0.5)
        batchsize_old = args.batchsize - batchsize_new
        # --- wrap old data features (latent)
        dataset_old = dsetutils.DSET_wrapper_Replay(coreset.coreset_im, coreset.coreset_t, latents=coreset.coreset_latents, transform=args.tf_coreset)
        loader_old = torch.utils.data.DataLoader(dataset_old, batch_size=batchsize_old,
                                                shuffle=True, num_workers=args.num_workers)
    else:
        batchsize_new = args.batchsize
        batchsize_old = 0
        loader_old = None
        

    if t>0:
        # 1) get appropriate "current data"
        if args.keep_all_data==True:
            new_data = train_datasets_new_only[t]
        else:
            new_data = train_datasets[t]

        print('Get scores for novelty detector')
        for t_old in range(t):
            use_old_dsets[t_old].select_random_subset(num_old_per_task[t][t_old])

        current_dset = novelu.CurrentTask(new_data, use_old_dsets[:t])

        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)

        params_score = {'layer':args.dfm_layers_input, 'feature_extractor':network_inner, \
            'base_apply_score':True, 'target_ind':dset_prep['scenario_classif']}


        if args.threshold_type=='simple':

            # 2) Threshold Novel/Old - Binary prediction 
            results_novelty = novelval.evaluate_dfm_CL(t, novelty_detector, noveltyResults, params_score, current_loader)
            inds_pred_novel,_ = novelu.Threshold_n_Select_Simple(t, results_novelty, th_percentile_score=args.threshold_percentile, 
                                th_percentile_confidence=args.th_percentile_confidence,  metric_confidence=(args.metric_confidence, args.metric_confidence_direction_novel))
            threshold_used = args.threshold_percentile

        else:
            args.params_score = Namespace(**params_score)
            args.num_samples = current_dset.__len__()
            args.th_percentile = args.threshold_percentile
            args.w = args.w_score
            # Threshold_n_Select_Iters(params)
            SelectTh = novelu.Threshold_n_Select_Iters(args)
            inds_pred_novel = SelectTh.select_novel(current_dset, novelty_detector)
            results_novelty = SelectTh.evaluate_CL(t, current_dset, novelty_detector, noveltyResults)
            threshold_used = args.max_threshold_total

                    
        # store results per task 
        per_task_results.scores_dist[t] = results_novelty.scores_dist
        per_task_results.scores[t] = results_novelty.scores
        per_task_results.gt_novelty[t] = results_novelty.gt_novelty
        per_task_results.gt[t] = results_novelty.gt
        per_task_results.inds[t] = results_novelty.dset_inds
        per_task_results.threshold[t] = threshold_used

        preds_array = np.zeros(results_novelty.scores.shape)
        preds_array[inds_pred_novel]=1
        per_task_results.preds[t] = preds_array

        # TODO when classification is introduced, log per_task_results.preds as well 
        # For Multi-Iter Thresholding this should all happen here 
        # One issue with the threshold being a percentile is that it pre-fixes the number of samples of "novelty"

        # 3) Pseudolabel "Novel" Samples
        if args.prediction_propagation:
            print('Use Predicted Labels for Propagation/DFM Fit')
            loader_new = torch.utils.data.DataLoader(novelu.NovelTask(t, num_classes, current_dset, pred_novel_inds=inds_pred_novel), batch_size=batchsize_new,
                                                    shuffle=True, num_workers=args.num_workers)
        else:
            inds_gt_novel = np.where(current_dset.novelty_y==1)[0]
            print('Use Ground Truth Labels for Propagation/DFM Fit')
            loader_new = torch.utils.data.DataLoader(novelu.NovelTask(t, num_classes, current_dset, pred_novel_inds=inds_gt_novel), batch_size=batchsize_new,
                                                    shuffle=True, num_workers=args.num_workers)

    else:
        loader_new = torch.utils.data.DataLoader(novelu.NovelTask(t, num_classes, train_datasets[t]), batch_size=batchsize_new, shuffle=True, num_workers=args.num_workers)


    # if training classifier or doing finetuning of backbone 
    if args.train_clf:
        clf.train(t, args, novelty_detector, network_inner, loader_old, loader_new, test_loaders, train_technique=args.train_technique)

    #  call training loop with one liner 
    args, temp_train_loader = novelu.temporary_loader_novelty(args, loader_new, coreset)

    print('Generate features for novelty evaluation and coreset')
    # Right now, homog_labels = pseudo-class_labels 
    processed_data = futils.extract_features(network_inner, temp_train_loader, \
        target_ind=dset_prep['scenario_classif'], homog_ind=1, 
        device=args.device, use_raw_images=args.use_image_as_input, raw_image_transform=args.add_tf)



    # ----Update memory and Novelty detector
    if t<num_tasks-1:

        if args.novelty_detector_name=='dfm':
            dfm_x = processed_data[0][args.dfm_layers_input]
            dfm_y = processed_data[1]
            if args.finetune_backbone=='all':
                novelty_detector, dfm_x, dfm_y = novelu.reprocess_data_novelty(args, dfm_x, dfm_y, network_inner, coreset)


            novelty_detector.fit_total(dfm_x.T, dfm_y)


        # ------ update coreset (if applicable)
        if coreset is not None:
            print('Append data to memory')
            coreset.append_memory(processed_data, current_task)
            print('coreset_im', coreset.coreset_im.shape)


    num_classes = num_classes + len(current_task)



#----------------------------------------------------save outputs 


with open('%s/results_tasks.pickle'%(args.dir_save), 'wb') as handle:
    pickle.dump(per_task_results, handle, protocol=pickle.HIGHEST_PROTOCOL)