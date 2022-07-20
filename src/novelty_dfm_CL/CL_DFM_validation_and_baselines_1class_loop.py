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
import tforms
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents
# import classifier as clf


import novelty_dfm_CL.novelty_detector as novel
import novelty_dfm_CL.novelty_eval as novelval 
import novelty_dfm_CL.classifier as clf
import novelty_dfm_CL.novelty_utils as novelu
import novelty_dfm_CL.incDFM_w_validation as novelinc


import memory as mem
import utils
import datasets as dset
import novelty_dfm_CL.datasets_holdout_validation as dseth
import datasets_utils as dsetutils

import novelty_dfm_CL.scoring_multi_threshold as ThScores


from novelty_dfm_CL.novelty_eval import acuracy_report, scores_metrics_results




# TODO add config file as system bash entry (using argparse)

utils.seed_torch(0)

general_config_path = '../configs/DFM_CL_validation_n_baselines_1class_loop_script.yaml' # path to file 
# general_config_path = '../configs/DFM_CL_validation_n_baselines_ratios_1class_loop_script.yaml' # path to file 

# general_config_path = '../configs/DFM_CL_n_baselines_multitask_baseline_script.yaml' # path to file 


with open(general_config_path) as fid:
        args = Namespace(**yaml.load(fid, Loader=yaml.SafeLoader))



torch.set_num_threads(args.num_threads)
pin_memory=False

# set up save
args = utils.config_saving(args)

# ------------------------------------------------------------------------------------------


start=time.time()
# 1) ----- Dataset 
if not hasattr(args, 'experiment_filepath'):
    args.experiment_filepath = None # have dictionary for defaults???
if not hasattr(args, 'experiment_name'):
    args.experiment_name = None # have dictionary for defaults??



# load holdout dset
datasets_use = dseth.call_dataset_holdout_w_validation(args.dset_name, args.data_dir, args.experiment_dir, 
                                        experiment_filepath=args.experiment_filepath, experiment_name=args.experiment_name, 
                                        holdout_percent=args.holdout_percent,  val_holdout=args.val_percent, scenario=args.scenario, 
                                        num_per_task=args.num_per_task, num_classes_first=args.num_classes_first, 
                                        type_l_cifar=args.type_l_cifar, num_tasks_cifar=args.num_tasks_cifar, num_tasks=args.num_tasks, 
                                        shuffle=args.shuffle_order, preload=args.preload, keep_all_data=args.keep_all_data, equalize_labels=args.equalize_labels)

if args.keep_all_data==True:
    train_datasets, train_holdout_datasets, train_datasets_new_only, val_datasets, test_datasets, list_tasks, list_tasks_targets, dset_prep = datasets_use
else:
    train_datasets, train_holdout_datasets, val_datasets, test_datasets, list_tasks, list_tasks_targets, dset_prep = datasets_use




args.dset_prep = dset_prep
if args.num_tasks>0:
    num_tasks = args.num_tasks
else:
    num_tasks = len(train_datasets)
    
   
old_dsets = train_holdout_datasets
old_dsets_test = test_datasets
use_old_dsets = copy.deepcopy(train_holdout_datasets)
use_old_dsets_test = copy.deepcopy(test_datasets)
 

test_loaders = [torch.utils.data.DataLoader(test_datasets[t], batch_size=args.batchsize_test,
                                                shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]


# val_loaders = [torch.utils.data.DataLoader(val_datasets[t], batch_size=args.batchsize_test,
#                                                 shuffle=True, num_workers=args.num_workers) for t in range(num_tasks)]

# print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())

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


args.patience_lr = int(np.ceil(args.schedule_patience_perepoch*args.num_epochs))

# x= torch.rand((50,3,224,224)).to(args.device)
# print(network.base(x).shape)
# # print(futils.layer_names(network))


# 3) ---- Novelty
args.detector_params['target_ind']=dset_prep['scenario_classif']
args.detector_params['device']=args.device
if args.novelty_detector_name=='odin':
    args.detector_params['base_network']=network #simple network (not wrapper) - Is this problematic? TODO
    args.detector_params['num_classes']=dset_prep['total_classes']
    args.detector_params['criterion']= nn.CrossEntropyLoss()
    args.detector_params['num_epochs']=args.num_epochs
    args.detector_params['train_technique']=args.train_technique
    args.detector_params['lr']=args.lr
    args.detector_params['patience_lr'] = args.patience_lr
    args.detector_params['schedule_decay'] = args.schedule_decay
    args.detector_params['step_size_epoch_lr']= args.step_size_epoch_lr
    args.detector_params['gamma_step_lr']= args.gamma_step_lr
elif args.novelty_detector_name=='softmax':
    args.detector_params['base_network']=network #simple network (not wrapper) - Is this problematic? TODO
    args.detector_params['num_classes']=dset_prep['total_classes']


noveltyResults = novelval.save_novelty_results(num_tasks, args.experiment_name_plot, args.dir_save)
noveltyResults_test = novelval.save_novelty_results_test(num_tasks, args.experiment_name_plot, args.dir_save)
novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector_name, params=args.detector_params)
print('Novelty Detector set up', time.time()-start, args.novelty_detector_name)


# sys.exit()


# 4) ---- Memory (if applicable for recomputing DFM per task)
# Use raw images in memory
coreset = None
if args.coreset_size>0:
    if args.use_image_as_input:
        raw_sizes = {'svhn':32, 'cifar10':32, 'cifar100':32, 'emnist':32, 'inaturalist':224}
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


    batchsize_old, batchsize_new = clf.divide_batches_old_new(t, args.batchsize, list_tasks_targets, coreset,\
        max_batch_ratio=args.max_batch_ratio, type_batch_divide=args.type_batch_divide)
    
    print('batch_old/batch_new', batchsize_old, batchsize_new)
        
    # continue


    # ------- Set up data + coreset 
    if t>0 and (coreset is not None):
        # --- wrap old data features (latent)
        dataset_old = dsetutils.DSET_wrapper_Replay(coreset.coreset_im, coreset.coreset_t, latents=coreset.coreset_latents, transform=args.tf_coreset)
        loader_old = torch.utils.data.DataLoader(dataset_old, batch_size=batchsize_old,
                                                shuffle=True, num_workers=args.num_workers)
    else:
        loader_old = None
        


    if t>0:
        # 1) get appropriate "current data"
        if args.keep_all_data==True:
            new_data = train_datasets_new_only[t]
        else:
            new_data = train_datasets[t]

        # mix old (holdout) with new data for "current task"    
        print('Get scores for novelty detector')
        for t_old in range(t):
            use_old_dsets[t_old].select_random_subset(num_old_per_task[t][t_old])
            
        current_dset = novelu.CurrentTask(new_data, use_old_dsets[:t], use_coarse=dset_prep['use_coarse'])
        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)
        
        
        # Get test set but keep same ratio of old/new 
        # num_new_test = test_datasets[t].__len__()
        # num_old_test = int(args.percent_old_mix*num_new_test)
        # num_per_old_test = utils.divide_integer_K(num_old_test,len(use_old_dsets_test[:t]), shuff=True)
        
        
        for t_old_ in range(t):
            use_old_dsets_test[t_old_].select_random_subset(num_old_per_task_test[t][t_old_])
            
        # print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())

        test_old_plus_novel_dset = novelu.CurrentTask(test_datasets[t], use_old_dsets_test[:t], use_coarse=dset_prep['use_coarse'])
        test_old_plus_novel_loader = torch.utils.data.DataLoader(test_old_plus_novel_dset, batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)
        
        # print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())

        


        params_score = {'layer':args.dfm_layers_input, 'feature_extractor':network_inner, \
            'base_apply_score':True, 'target_ind':dset_prep['scenario_classif'], 'device': args.device}
        
        
        # make validation wrapper 
        val_loader = torch.utils.data.DataLoader(novelu.ValidationDset(val_datasets[:t], use_coarse=dset_prep['use_coarse']), batch_size=args.batchsize_test,
                                                shuffle=False, num_workers=args.num_workers)


        if args.threshold_type=='simple':
            
            # 2) Threshold Novel/Old - Binary prediction 
            if args.novelty_detector_name=='odin':
                results_novelty = novelval.evaluate_odin_CL(t, novelty_detector, noveltyResults, params_score, current_loader)
                params_score['score_func'] = results_novelty.best_score_func
                params_score['noise_magnitude'] = results_novelty.best_noise_magnitude
                # invert scores for ODIN so that high scores are novelty (1)
            else:
                results_novelty = novelval.evaluate_simple_CL(t, novelty_detector, noveltyResults, params_score, current_loader, args.novelty_detector_name)

            # have function here to estimate the threshold ..... 
            threshold_estimated_novel = novelinc.estimate_threshold_from_val_simple(val_loader, args.percentile_val_threshold, \
                novelty_detector, params_score, args.novelty_detector_name)   
            
            
            # threshold_estimated_novel TODO
            inds_pred_novel,_ = novelinc.Threshold_n_Select_Simple(t, results_novelty, threshold_estimated_novel, \
                                th_percentile_confidence=args.th_percentile_confidence,  metric_confidence=(args.metric_confidence, args.metric_confidence_direction_novel))
            threshold_used = args.threshold_percentile
            acc_overall, acc_novel, prec_novel, recall_novel = noveltyResults.compute_accuracies(t, 0, inds_pred_novel, results_novelty.gt_novelty)

            print('Binary Pseudolabeling - acc_overall: %.4f, acc_novel: %.4f, prec_novel: %.4f, recall_novel: %.4f' %(acc_overall, acc_novel, prec_novel, recall_novel))

            novelval.evaluate_simple_CL_test(t, novelty_detector, noveltyResults_test, params_score, test_old_plus_novel_loader)

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
        
        
        print('inds_pred_novel', inds_pred_novel)



        # 3) Pseudolabel "Novel" Samples
        if args.prediction_propagation:
            if inds_pred_novel.shape[0]>0:
                print('Use Predicted Labels for Propagation/DFM Fit')
                loader_new = torch.utils.data.DataLoader(novelu.NovelTask(t, num_classes, current_dset, pred_novel_inds=inds_pred_novel), batch_size=batchsize_new,
                                                        shuffle=True, num_workers=args.num_workers)
            else:
                print('No indices were predicted as novel --> skip to next task')
                loader_new=None
        else:
            inds_gt_novel = np.where(current_dset.novelty_y==1)[0]
            print('Use Ground Truth Labels for Propagation/DFM Fit')
            loader_new = torch.utils.data.DataLoader(novelu.NovelTask(t, num_classes, current_dset, pred_novel_inds=inds_gt_novel), batch_size=batchsize_new,
                                                    shuffle=True, num_workers=args.num_workers)

    else:
        loader_new = torch.utils.data.DataLoader(novelu.NovelTask(t, num_classes, train_datasets[t], use_coarse=dset_prep['use_coarse']), \
            batch_size=batchsize_new, shuffle=True, num_workers=args.num_workers)


    # if training classifier or doing finetuning of backbone 
    # print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())

    if args.train_clf:
        clf.train(t, args, novelty_detector, network_inner, loader_old, loader_new, test_loaders, train_technique=args.train_technique)


    if loader_new is not None:

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



        # ----Update memory and Novelty detector
        if t<num_tasks-1:
            if args.novelty_detector_name=='dfm':
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


    # sys.exit()

    num_classes = num_classes + len(current_task)

    # print('PCA mats', novelty_detector.pca_mats)
    # if t==1:
    #     print('test_loaders[0].dataset.__len__()', test_loaders[0].dataset.__len__())
    #     sys.exit()

# ------------------------------------------------------------------------------------------------


with open('%s/results_tasks.pickle'%(args.dir_save), 'wb') as handle:
    pickle.dump(per_task_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
