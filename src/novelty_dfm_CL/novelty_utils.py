import sys
import numpy as np
from sklearn.decomposition import PCA, FastICA
import copy
import os
import torch
from scipy.stats import entropy

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import novelty_dfm_CL.novelty_detector as novel

import datasets as dset
import feature_extraction.feature_extraction_utils as futils
import tforms 
import utils


# -----------------------------------------------Dataset utils 



class PseudolabelDset():
    def __init__(self, current_dset, pred_novel_inds=None, transform=None):
        """ 
        pseudolabeled dset for use when pseudolabeling iteratively
        """
        self.x = current_dset.x
        self.novelty_gt = current_dset.novelty_y
        self.novelty_y =np.ones(pred_novel_inds.shape)

        if pred_novel_inds is not None:
            self.x = self.x[pred_novel_inds,...]
            self.novelty_gt = self.novelty_gt[pred_novel_inds]


        if transform is None:
            self.transform = current_dset.transform
        else:
            self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        im = self.x[idx,...]
        
        if self.transform is not None: ##input the desired tranform 
            im = self.transform(im)
            
        return im, self.novelty_y[idx], self.novelty_y[idx]





class NovelTask():
    def __init__(self, task, num_current_classes, current_dset, pred_novel_inds=None, use_coarse=False, transform=None, returnGT=False):
        """ 
        When Each novel task only contains one class/label (except for first task)
        Filter and keep only predicted_novel_inds
        Return Ground Truths Optionally
        """

        self.x = current_dset.x

        if use_coarse:
            labels = current_dset.y_coarse
        else:
            labels = current_dset.y


        if task>0:
            # labels to be used are just a +1 
            self.y = (np.ones(self.x.shape[0])*(num_current_classes)).astype(int)
        else:
            # First task (task 0)
            if torch.is_tensor(labels):
                self.y = labels.numpy().astype(int) # for first task
            else:
                self.y = labels.astype(int)


        self.gt_novelty_y = np.ones(self.x.shape[0]).astype(int)

        if torch.is_tensor(labels):
            self.gt_y = labels.numpy().astype(int)
        else:
            self.gt_y = labels.astype(int)


        if pred_novel_inds is not None:
            self.x = self.x[pred_novel_inds,...]
            self.y = self.y[pred_novel_inds]
            self.gt_novelty_y = self.gt_novelty_y[pred_novel_inds]
            self.gt_y = self.gt_y[pred_novel_inds]

        if transform is None:
            self.transform = current_dset.transform
        else:
            self.transform = transform

        self.indices_task = np.arange(self.y.shape[0])

        self.returnGT = returnGT

    def __len__(self):
        return self.y.shape[0]

        
    def __getitem__(self, idx):

        im = self.x[idx,...]
                
        if self.transform is not None: ##input the desired tranform 
            im = self.transform(im)
            im = im.to(dtype=torch.float32)

        class_lbl = self.y[idx]

        if self.returnGT:
            novelty_gt = self.gt_novelty_y[idx]
            class_gt = self.gt_y[idx]
            return im, class_lbl, class_gt, novelty_gt
            
        return im, class_lbl




class CurrentTask():
    def __init__(self, new_dset, old_dsets, use_coarse=False, transform=None, returnIDX=True):
        """ 
        combine old IID task samples and New task samples
        Load x_old and y_old data here (already subsampled if applicable)
            --> Consider bypassing dsetTask and loading the experiment_indices directly here
        x = torch
        y = numpy
        novelty_y = numpy
        """

        self.x = new_dset.x[new_dset.indices_task,...]
        if use_coarse:
            self.y = new_dset.y_coarse[new_dset.indices_task].numpy().astype(int)
        else:
            self.y = new_dset.y[new_dset.indices_task].numpy().astype(int)
        self.novelty_y = np.ones(self.y.shape[0]).astype(int)


        for i, dset in enumerate(old_dsets):
            self.x = torch.cat((self.x, dset.x[dset.indices_task,...]), dim=0)
            if use_coarse:
                self.y = np.concatenate((self.y, dset.y_coarse[dset.indices_task].numpy().astype(int)))
            else:
                self.y = np.concatenate((self.y, dset.y[dset.indices_task].numpy().astype(int)))
            self.novelty_y = np.concatenate((self.novelty_y, np.zeros(dset.indices_task.shape).astype(int)))

        if transform is None:
            self.transform = new_dset.transform
        else:
            self.transform = transform
        self.returnIDX = returnIDX

        self.indices_task = np.arange(self.y.shape[0])
        

    def __len__(self):
        return self.y.shape[0]

        
    def __getitem__(self, idx):

        im = self.x[idx,...]
        
        if self.transform is not None: ##input the desired tranform 

            im = self.transform(im)

        class_lbl = self.y[idx]
        novelty_lbl = self.novelty_y[idx]

        if self.returnIDX:
            return im, class_lbl, novelty_lbl, idx
            
        return im, class_lbl, novelty_lbl




class ValidationDset():
    def __init__(self, val_dsets, use_coarse=False, transform=None, returnIDX=True):
        """ 
        combine all validation dsets into one 
        """

        for i, dset in enumerate(val_dsets):
            if i==0:
                self.x = dset.x[dset.indices_task,...]
                if use_coarse:
                    self.y = dset.y_coarse[dset.indices_task]
                else:
                    self.y = dset.y[dset.indices_task]
            else:
                self.x = torch.cat((self.x, dset.x[dset.indices_task,...]), dim=0)
                if use_coarse:
                    self.y = np.concatenate((self.y, dset.y_coarse[dset.indices_task].numpy().astype(int)))
                else:
                    self.y = np.concatenate((self.y, dset.y[dset.indices_task].numpy().astype(int)))


        if transform is None:
            self.transform = dset.transform
        else:
            self.transform = transform
            
        self.returnIDX = returnIDX
        self.indices_task = np.arange(self.y.shape[0])
        

    def __len__(self):
        return self.y.shape[0]

        
    def __getitem__(self, idx):

        im = self.x[idx,...]
        
        if self.transform is not None: ##input the desired tranform 

            im = self.transform(im)

        class_lbl = self.y[idx]

        if self.returnIDX:
            return im, class_lbl, class_lbl, idx
            
        return im, class_lbl, class_lbl



# ------------------------------ other utils 



def flatten(t):
    return [item for sublist in t for item in sublist]


def num_mix_old_novelty_test(percent_old, train_dataset, train_holdout_datasets, current_task, list_tasks):
    '''
    num_old_per_task (array): num_per_old for each old class, summing num_old. 
    '''

    print('list_tasks', list_tasks)

    num_new = train_dataset.__len__()
    num_old_total = sum([train_holdout_datasets[i].__len__() for i in range(current_task)])

    # print('num_old_total', num_old_total)

    num_old = min(int(percent_old*num_new), num_old_total)
    # real_percent_old = num_old/num_new
    # print('real_percent_old', real_percent_old)


    # num_old_per_task = utils.divide_integer_K(num_old, len(train_holdout_datasets))

    num_old_per_task_ = utils.divide_integer_K(num_old, len(flatten(list_tasks)))


    # account for first task which may have more labels 
    num_first = len(list_tasks[0])
    num_old_per_task = [sum(num_old_per_task_[:num_first])]
    if len(num_old_per_task_[num_first:])>0:
        num_old_per_task.extend(num_old_per_task_[num_first:])
    

    # print('num_old_per_task', num_old_per_task)

    # uniform sampling accross old tasks, can be changed after
    # num_old_per_task = int(num_old/len(flatten(list_tasks[:current_task])))

    return num_old, num_new, num_old_per_task






def num_mix_old_novelty(percent_old, train_dataset, train_holdout_datasets, current_task, list_tasks):
    '''
    num_old_per_task (array): num_per_old for each old class, summing num_old. 
    '''

    print('list_tasks', list_tasks)

    num_new = train_dataset.__len__()
    num_old_total = sum([train_holdout_datasets[i].__len__() for i in range(current_task)])

    # print('num_old_total', num_old_total)

    num_old = min(int(percent_old*num_new), num_old_total)
    # real_percent_old = num_old/num_new
    # print('real_percent_old', real_percent_old)


    # num_old_per_task = utils.divide_integer_K(num_old, len(train_holdout_datasets))

    num_old_per_task_ = utils.divide_integer_K(num_old, len(flatten(list_tasks)))


    # account for first task which may have more labels 
    num_first = len(list_tasks[0])
    num_old_per_task = [sum(num_old_per_task_[:num_first])]
    if len(num_old_per_task_[num_first:])>0:
        num_old_per_task.extend(num_old_per_task_[num_first:])
    

    # print('num_old_per_task', num_old_per_task)

    # uniform sampling accross old tasks, can be changed after
    # num_old_per_task = int(num_old/len(flatten(list_tasks[:current_task])))

    return num_old, num_new, num_old_per_task




def compute_max_nested_dict(dict_):
    max_val_plot =0
    for key1 in dict_.keys():
        for key2 in dict_[key1].keys():
            if dict_[key1][key2].shape[0]:
                max_val_plot=max(max_val_plot,dict_[key1][key2].max())
    return max_val_plot



def confidence_scores(dist_array, metric='entropy'):
    '''
    Compute metric over num_class dimension of PCA/NLL score 
    dist_array shape: (num_Classes, num_samples)
    '''
    if metric=='entropy':
        confidences = entropy(dist_array, axis=0)
        eval_type='min'
    elif metric=='variance':
        confidences = np.var(dist_array, axis=0)
        eval_type='min'

    return confidences, eval_type


    
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



#---------------------------------------------------------------------------------



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
                
