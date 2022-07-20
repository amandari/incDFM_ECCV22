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




class CurrentTask():
    def __init__(self, new_dset, old_dsets, use_coarse=False, returnIDX=False):
        """ 
        combine old IID task samples and New task samples
        Load x_old and y_old data here (already subsampled if applicable)
            --> Consider bypassing dsetTask and loading the experiment_indices directly here
        x = torch
        y = numpy
        novelty_y = numpy
        """
        self.use_coarse = use_coarse
        
        self.dsets = [new_dset] + old_dsets
        
        self.novelty_y = np.ones(new_dset.__len__())
        
        indices_task_new = list(zip([0 for i in range(new_dset.__len__())], [i for i in range(new_dset.__len__())]))
        indices_task_old = []
        for j, dset in enumerate(old_dsets):
            indices_task_old.extend(list(zip([j+1 for i in range(dset.__len__())], [i for i in range(dset.__len__())])))
            self.novelty_y = np.concatenate((self.novelty_y, np.zeros((dset.__len__(),))))
            

        self.indices_task_init = indices_task_new + indices_task_old
        
        self.indices_task = copy.deepcopy(self.indices_task_init)
                    
        self.returnIDX = returnIDX
        

    def __len__(self):
        return len(self.indices_task)
    
    
    def select_specific_subset(self, indices_select):
        
        self.indices_task = [self.indices_task_init[i] for i in indices_select]
        
    def __getitem__(self, idx):
        
        dset_id, idx_dset = self.indices_task[idx]
        
        # print('dset_id', dset_id)
        
        im, y_fine, y_coarse = self.dsets[dset_id].__getitem__(idx_dset)
        
        if self.use_coarse:
            class_lbl = y_coarse
        else:
            class_lbl = y_fine

        novelty_lbl = self.novelty_y[idx]

        if self.returnIDX:
            return im, class_lbl, novelty_lbl, idx
            
        return im, class_lbl, novelty_lbl




class PseudolabelDset():
    def __init__(self, current_dset, pred_novel_inds=None):
        """ 
        pseudolabeled dset for use when pseudolabeling iteratively
        """
        ## need to get indices_task 
        self.dset = copy.deepcopy(current_dset)
        self.novelty_y =np.ones(pred_novel_inds.shape)

        if pred_novel_inds is not None:
            self.dset.select_specific_subset(pred_novel_inds)
            
        self.indices_task = self.dset.indices_task

    def __len__(self):
        return len(self.indices_task)

    def __getitem__(self, idx):                

        im, _, _ = self.dset.__getitem__(idx) 
        
        return im, self.novelty_y[idx], self.novelty_y[idx]



class NovelTask():
    def __init__(self, task, num_current_classes, current_dset, \
        pred_novel_inds=None, returnGT=False, use_coarse=False):
        """ 
        When Each novel task only contains one class/label (except for first task)
        Filter and keep only predicted_novel_inds
        Return Ground Truths Optionally
        """
        
        if hasattr(current_dset, 'use_coarse'):
            self.use_coarse = current_dset.use_coarse
        else:
            self.use_coarse = use_coarse

        self.dset = current_dset
        self.returnGT = returnGT
        self.task = task
        self.num_current_classes = num_current_classes
        
        if pred_novel_inds is not None:
            self.dset.select_specific_subset(pred_novel_inds)
        self.indices_task = self.dset.indices_task


    def __len__(self):
        return len(self.indices_task)

        
    def __getitem__(self, idx):
        
        
        if self.task==0:
            im, y_fine, y_coarse = self.dset.__getitem__(idx)
            novelty_gt = 1
            if self.use_coarse:
                class_lbl = y_coarse
            else:
                class_lbl = y_fine
        else:
            im, class_gt, novelty_gt = self.dset.__getitem__(idx)
            class_lbl = self.num_current_classes


        if self.returnGT:
            return im, class_lbl, class_gt, novelty_gt
            
        return im, class_lbl




class ValidationDset():
    def __init__(self, val_dsets, use_coarse=False, returnIDX=True):
        """ 
        combine all validation dsets into one 
        """
        
        self.dsets = val_dsets # list of dset objects 
        
        size_dsets = sum([d.__len__() for d in self.dsets])
        
        self.novelty_y = np.zeros((size_dsets,)) # only In-distribution datasets 
        
        self.indices_task = []
        for j, dset in enumerate(self.dsets):
            self.indices_task.extend(list(zip([j for i in range(dset.__len__())], [i for i in range(dset.__len__())])))
            
        self.returnIDX = returnIDX
        
        self.use_coarse = use_coarse
        

    def __len__(self):
        return len(self.indices_task)

    
    def __getitem__(self, idx):
        
        dset_id, idx_dset = self.indices_task[idx]
        
        # print('dset_id', dset_id)
        
        im, y_fine, y_coarse = self.dsets[dset_id].__getitem__(idx_dset)

        if self.use_coarse:
            class_lbl = y_coarse
        else:
            class_lbl = y_fine

        if self.returnIDX:
            return im, class_lbl, class_lbl, idx
            
        return im, class_lbl, class_lbl
