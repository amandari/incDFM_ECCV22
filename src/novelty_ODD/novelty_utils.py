import sys
import numpy as np
from sklearn.decomposition import PCA, FastICA
import copy
import os
import torch

import novelty_ODD.novelty_detector as novel

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import datasets as dset
import feature_extraction.feature_extraction_utils as futils
import tforms 



def reduce_dim_features(data_subset, data_subset_labels, num_components=320):
    for j, lb in enumerate(np.unique(data_subset_labels)):
        #get subset of data corresponding to that label
        inds = np.where(data_subset_labels==lb)[0]
        lbls = data_subset_labels[inds]

        pcaobj = PCA(n_components=num_components)
        pca_result = pcaobj.fit_transform(data_subset[lbls, ...])
        print('pca_result', pca_result.shape)
        if j==0:
            labels_ordered = lbls
            data_reduced = pca_result
        else:
            labels_ordered = np.concatenate((labels_ordered, lbls))
            data_reduced = np.concatenate((data_reduced, pca_result), axis=0)
            
    return data_reduced, labels_ordered


def reduce_dim_features_fixed(data_subset, data_subset_labels, num_components=320):
    pcaobj = PCA(n_components=num_components)
    for j, lb in enumerate(np.unique(data_subset_labels)):
        #get subset of data corresponding to that label
        inds = np.where(data_subset_labels==lb)[0]
        lbls = data_subset_labels[inds]
        
        if j==2:
            pcaobj.fit(data_subset[lbls,...])
        
        pca_result = pcaobj.transform(data_subset[lbls, ...])

        print('pca_result', pca_result.shape)
        if j==0:
            labels_ordered = lbls
            data_reduced = pca_result
        else:
            labels_ordered = np.concatenate((labels_ordered, lbls))
            data_reduced = np.concatenate((data_reduced, pca_result), axis=0)
            
    return data_reduced, labels_ordered



    
def temporary_loader_novelty(args, loader_new, coreset):
    # modify the transform of the dataset in case of storing raw images in coreset
    if args.use_image_as_input and (coreset is not None):
        print('modify transform for raw images in coreset')
        temp_train_loader = copy.deepcopy(loader_new)
        # temp_train_loader.dataset.transform = tforms.tf_simple()
        temp_train_loader.dataset.transform = None
        args.tf_coreset = tforms.tf_additional(args.dset_name)
        args.add_tf = tforms.TFBatch(args.tf_coreset)
        args.generate_raw = True
    else:
        temp_train_loader = loader_new 
        args.tf_coreset = None
        args.add_tf = None
        args.generate_raw = False

    return args, temp_train_loader



def reprocess_data_novelty(args, dfm_x, dfm_y, network_inner, coreset):

    # call init method again
    novelty_detector = novel.NoveltyDetector().create_detector(type=args.novelty_detector_name, params=args.detector_params)
    if args.keep_all_data==False and (coreset is not None) and (coreset.coreset_im.shape[0]>0):
        # means you have a coreset which needs to be used to recompute the DFM fit
        if args.generate_raw==False:
            # coreset contains embedding features
            dfm_x = torch.cat((dfm_x, coreset.coreset_im), dim=0)
            dfm_y = torch.cat((dfm_y, coreset.coreset_t), dim=0)
        else:
            print('fuse coreset with new data for recomputing DFM')
            # means you have to extract features from coreset data
            d_temp = dset.DSET_wrapper_Replay(coreset.coreset_im, coreset.coreset_t, transform=args.tf_coreset)
            l_temp = torch.utils.data.DataLoader(d_temp, batch_size=50,
                                                    shuffle=False, num_workers=args.num_workers)
            coreset_feats_temp = futils.extract_features(network_inner, l_temp, \
                        target_ind=args.dset_prep['scenario_classif'], homog_ind=args.dset_prep['homog_ind'], device=args.device, \
                        use_raw_images=args.use_image_as_input, raw_image_transform=None)
            del d_temp, l_temp
            dfm_x = torch.cat((dfm_x, coreset_feats_temp[0][args.input_layer_name]), dim=0)
            dfm_y = torch.cat((dfm_y, coreset_feats_temp[1]), dim=0)

    return novelty_detector, dfm_x, dfm_y
                
