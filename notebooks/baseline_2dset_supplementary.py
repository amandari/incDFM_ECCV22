import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib import colors as mcolors
import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import datasets, models, transforms
from sklearn.decomposition import PCA, FastICA
import itertools
import yaml
from argparse import Namespace


sys.path.append('../src')
import tforms
import feature_extraction.feature_extraction_utils as futils
from feature_extraction.Network_Latents_Wrapper import NetworkLatents

import datasets_utils as dsetutils
import utils

sys.path.append('../src/novelty_dfm_CL')

import novelty_dfm_CL.datasets_holdout_validation as dseth
import novelty_dfm_CL.novelty_detector as novel
import novelty_dfm_CL.novelty_eval as novelval 
import novelty_dfm_CL.classifier as clf
import novelty_dfm_CL.novelty_utils as novelu
import novelty_dfm_CL.incDFM_w_validation as novelinc
import novelty_dfm_CL.novelty_dataset_wrappers as dwrap


np.random.seed(42)

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# parameters

device = 1
ood_method='softmax'
dset_id_name = 'cifar10'
dset_od_name = 'svhn'
extractor_name = 'resnet50'
finetune_backbone=False
num_epochs=10
batch_size_train=50

cut_epoch_short=False

load_pretrained_backbone=None
# load_pretrained_backbone = '/lab/arios/ProjIntel/incDFM/sandbox/results_ood_sup_2d/Network_finetune_resnet50_cifar10_epochs_10/'
# load_pretrained_backbone = '/lab/arios/ProjIntel/incDFM/sandbox/results_ood_sup_2d/Network_finetune_cifar10_epochs_10/'




experiment_name='MT_Supplementary'
data_dir = '/lab/arios/ProjIntel/incDFM/data/'
experiment_dir='/lab/arios/ProjIntel/incDFM/src/novelty_dfm_CL/Experiments_DFM_CL/'

holdout_percent=0.001
val_percent=0.1



name_experiment='OOD_only2_ID_%s_OD_%s'%(dset_id_name,dset_od_name)
dir_save='/lab/arios/ProjIntel/incDFM/sandbox/results_ood_sup_2d/OOD_%s_ID_%s_OD_%s_finetuning_resnetsimple/'%(ood_method, dset_id_name, dset_od_name)
utils.makedirectory(dir_save)


ood_config_paths = {'dfm':'/lab/arios/ProjIntel/incDFM/src/configs_new/ood_method/dfm_ood_config.yaml',
                    'incdfm':'/lab/arios/ProjIntel/incDFM/src/configs_new/ood_method/incdfm_ood_config.yaml',
                    'mahal':'/lab/arios/ProjIntel/incDFM/src/configs_new/ood_method/mahal_ood_config.yaml',
                    'odin':'/lab/arios/ProjIntel/incDFM/src/configs_new/ood_method/odin_ood_config.yaml',
                    'softmax':'/lab/arios/ProjIntel/incDFM/src/configs_new/ood_method/softmax_ood_config.yaml',
}

num_outputs_map = {'cifar10':10, 'svhn':10, 'cifar100':100}
num_classes_id = num_outputs_map[dset_id_name]
num_classes_od = num_outputs_map[dset_od_name]


# ----------------------Set up network -------------------
def get_features(dataset, network_inner, device=0):

    loader = torch.utils.data.DataLoader(dataset, batch_size=100,
                                            shuffle=True, num_workers=4)
    start = time.time()
    print('feat extraction begin')
    current_features = futils.extract_features(network_inner, loader, \
            target_ind=1, homog_ind=-2, device=device)

    print('feat extraction done', time.time()-start)
    feat_name = 'base.8'

    X = current_features[0][feat_name]
    Y = current_features[-2]

    return X, Y, current_features


## Include validation set 
network = clf.Resnet(num_classes_id, resnet_arch=extractor_name, FC_layers=[4096],  
            resnet_base=-1, multihead_type='single', base_freeze = not finetune_backbone, pretrained_weights=None)
network = network.to(device)
# run extractor 
network_inner = NetworkLatents(network, ['base.8'], pool_factors={'base.8':-1})


# ----------- Prepare data -----------------------------


data_ID = dseth.call_dataset_holdout_w_validation(dset_id_name, data_dir, experiment_dir, 
                                        experiment_filepath=None, experiment_name=experiment_name, 
                                        holdout_percent=holdout_percent,  val_holdout=val_percent, scenario='nc', 
                                        num_per_task=num_classes_id, num_classes_first=num_classes_id, 
                                        num_tasks=1, 
                                        shuffle=False, preload=False, keep_all_data=False, \
                                            equalize_labels=False, clip_labels=False, clip_max=15000)


train_dataset_ID, train_holdout_dataset_ID, val_dataset_ID, test_dataset_ID, list_tasks_ID, list_tasks_targets_ID, dset_prep_ID = data_ID
print(train_holdout_dataset_ID[0].__len__(), train_dataset_ID[0].__len__(), val_dataset_ID[0].__len__(), test_dataset_ID[0].__len__())
train_dataset_ID, train_holdout_dataset_ID, val_dataset_ID, test_dataset_ID = train_dataset_ID[0], train_holdout_dataset_ID[0], val_dataset_ID[0], test_dataset_ID[0]




data_OD = dseth.call_dataset_holdout_w_validation(dset_od_name, data_dir, experiment_dir, 
                                        experiment_filepath=None, experiment_name=experiment_name, 
                                        holdout_percent=holdout_percent,  val_holdout=val_percent, scenario='nc', 
                                        num_per_task=num_classes_od, num_classes_first=num_classes_od, 
                                        num_tasks=1, 
                                        shuffle=False, preload=False, keep_all_data=False, \
                                            equalize_labels=False, clip_labels=False, clip_max=15000)

train_dataset_OD, train_holdout_dataset_OD, val_dataset_OD, test_dataset_OD, list_tasks_OD, list_tasks_targets_OD, dset_prep_OD = data_OD

print(train_holdout_dataset_OD[0].__len__(), train_dataset_OD[0].__len__(), val_dataset_OD[0].__len__(), test_dataset_OD[0].__len__())

train_dataset_OD, train_holdout_dataset_OD, val_dataset_OD, test_dataset_OD = train_dataset_OD[0], train_holdout_dataset_OD[0], val_dataset_OD[0], test_dataset_OD[0]



# -------------OOD Methodd ----------------------

ood_config = ood_config_paths[ood_method]



with open(ood_config) as fid:
    params = Namespace(**yaml.load(fid, Loader=yaml.SafeLoader))
        

# Set up detector params 
params.detector_params['target_ind']=1
params.detector_params['device']=device
params.detector_params['num_classes']=num_classes_id
if ood_method=='odin':
    params.detector_params['base_network']=network #simple network (not wrapper) - Is this problematic? TODO
    params.detector_params['num_classes_fine_tasks']=num_classes_id
    params.detector_params['criterion']= nn.CrossEntropyLoss()
    params.detector_params['num_epochs']=num_epochs
    params.detector_params['train_technique']=1
    params.detector_params['lr']=0.001
    params.detector_params['patience_lr'] = 0.25
    params.detector_params['schedule_decay'] = 0.5
    params.detector_params['step_size_epoch_lr']= 7
    params.detector_params['gamma_step_lr']= 0.1
elif ood_method=='softmax':
    params.detector_params['base_network']=network #simple network (not wrapper) - Is this problematic? TODO
    params.detector_params['num_classes_fine_tasks']=num_classes_id
elif ood_method=='mahal':
    params.detector_params['num_components']=0.995
    params.detector_params['balance_classes']=False


    

print('Get scores for novelty detector')
noveltyResults = novelval.save_novelty_results(1, name_experiment, dir_save)
# train_loaders_new_only evaluates only current new data (unseen by classifier/main model). 
# print('new', next(iter(train_loaders_new_only[t])))

novelty_detector = novel.NoveltyDetector().create_detector(type=ood_method, params=params.detector_params)


# --------------------- Fit ---------------------



#Test loader
val_loader_ID = torch.utils.data.DataLoader(val_dataset_ID,\
            batch_size=100, shuffle=True, num_workers=4)

print('Fitting %s on %s'%(ood_method, dset_id_name))

if ood_method=='dfm' or ood_method=='mahal' or ood_method=='incdfm':
    print('get features ID - %s'%dset_id_name)
    
    if load_pretrained_backbone is not None:
        print('load pretrained featt extractor')
        network.load_state_dict(torch.load(load_pretrained_backbone+'/model_best'))
        network_inner.model.load_state_dict(torch.load(load_pretrained_backbone+'/model_best'))

    
    # Training - Only ID -------------
    Feats_ID_train = get_features(train_dataset_ID, network_inner, device=device)
    novelty_detector.fit_total(Feats_ID_train[0].T, Feats_ID_train[1])
else:
    
    from novelty_dfm_CL.train_main import train_main_epoch
    from novelty_dfm_CL.test_main import test_main

    if load_pretrained_backbone is not None:
        print('load pretrained featt extractor')
        network.load_state_dict(torch.load(load_pretrained_backbone+'/model_best'))
        network_inner.model.load_state_dict(torch.load(load_pretrained_backbone+'/model_best'))
    else:
        # train Loader
        train_loader_ID = torch.utils.data.DataLoader(dwrap.NovelTask(0, num_classes_id, train_dataset_ID, use_coarse=False), \
                    batch_size=batch_size_train, shuffle=True, num_workers=4)
        
        # train classifier jointly 
        if finetune_backbone==True:
            network_inner.model.base.train(True)
            network_inner.base_freeze = False
            for param in network_inner.model.base.parameters():
                param.requires_grad = True
        else:
            network_inner.model.base.train(False)
            network_inner.base_freeze = True
            for param in network_inner.model.base.parameters():
                param.requires_grad = False

        if ood_method=='odin':
            clf_parameters = []
            for name, parameter in network_inner.model.named_parameters():
                if name == 'h.h.weight' or name == 'h.h.bias':
                    pass
                else:
                    clf_parameters.append(parameter)
        else:
            clf_parameters = network_inner.model.parameters()

            
        optimizer_main = optim.Adam(filter(lambda p: p.requires_grad, clf_parameters), lr=0.001)
        scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, 'min', patience=0.25, factor=0.5, min_lr=0.00001)
        
        # ----- relative batchsizes between old and new 
        for epoch in range(num_epochs):
            print('##########epoch %d###########'%epoch)
            network_inner.model.train()
            if finetune_backbone==True:
                network_inner.model.base.train(True)
                network_inner.base_freeze = False
                for param in network_inner.model.base.parameters():
                    param.requires_grad = True
            else:
                network_inner.model.base.train(False)
                network_inner.base_freeze = True
                for param in network_inner.model.base.parameters():
                    param.requires_grad = False
                    
            train_main_epoch(epoch, train_loader_ID, None, network_inner, optimizer_main,  device=device, OOD_class=novelty_detector,\
                        feature_name='base.8', weight_old=0.5, cut_epoch_short=cut_epoch_short,\
                        quantizer=None, target_ind=1, cuda=True, display_freq=10)

            val_loss = test_main(epoch, 0, [val_loader_ID], network_inner, novelty_detector, dir_save, \
                feature_name='base.8', target_ind=1, \
                    quantizer=None, cuda=True, device=device)

            # Update schedulers 
            scheduler_main.step(val_loss)


            if novelty_detector.name == 'odin':
                novelty_detector.h_scheduler.step(val_loss)




# ----------- test ------------------

current_dset = dwrap.CurrentTask(test_dataset_OD, [test_dataset_ID])
current_loader = torch.utils.data.DataLoader(current_dset, batch_size=100,
                                                shuffle=False, num_workers=4)

print('current_dset.__len__()', current_dset.__len__())

params_score = {'layer':'base.8', 'feature_extractor':network_inner, 'base_apply_score':True, 'target_ind':1}

args=Namespace()
params_score['device']=device
print('Computing Scores for ID %s using %s'%(dset_od_name, ood_method))
if ood_method=='dfm' or ood_method=='odin' or ood_method=='softmax' or ood_method=='mahal':
    
    # 2) Threshold Novel/Old - Binary prediction 
    if ood_method=='odin':
        results_novelty = novelval.evaluate_odin_CL(1, novelty_detector, noveltyResults, params_score, current_loader)
        # invert scores for ODIN so that high scores are novelty (1)
    else:
        results_novelty = novelval.evaluate_simple_CL(1, novelty_detector, noveltyResults, params_score, current_loader, ood_method)

else:
    args.params_score = Namespace(**params_score)
    args.num_samples = current_dset.__len__()
    args.w = 0.5
    args.alg_dfm='simple'
    args.max_iter_pseudolabel=10
    args.threshold_percentile=85
    current_old_new_ratio = 1.0
    args.validation_iid_loader = val_loader_ID
    args.percentile_val_threshold=95
    args.batchsize_test=100
    args.num_workers=4
    args.novelty_detector_name=ood_method
    args.detector_params = params.detector_params
    args.dset_prep = dset_prep_ID
    args.device=device
    args.use_image_as_input = False
    args.add_tf=None
    args.dfm_layers_input = 'base.8'
    # Threshold_n_Select_Iters(params)
    if args.alg_dfm == 'simple':
        SelectTh = novelinc.Threshold_n_Select_Iters(args)
    # elif args.alg_dfm == 'tug':
    #     SelectTh = novelinc.Threshold_Tug_incDFM(args)
        
    inds_pred_novel,_ = SelectTh.select_novel(1, current_dset, novelty_detector, noveltyResults)
    results_novelty = SelectTh.evaluate_CL(1, current_dset, novelty_detector, noveltyResults)
    