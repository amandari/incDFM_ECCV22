import sys
import numpy as np
import os
import torch
from argparse import Namespace
import json


sys.path.append('../')

# sys.path.insert(1, os.path.join(sys.path[0], '..'))

import OOD.novelty_detector as novel
from OOD.novelty_eval import acuracy_report, scores_metrics_results
import OOD.scoring_multi_threshold as ThScores
from OOD.novelty_utils import *

import dataset_scripts.novelty_dataset_wrappers as dwrap
import feature_extraction.feature_extraction_utils as futils





# threshold_estimated_novel = novelinc.estimate_threshold_from_val_simple(val_loader, args.percentile_val_threshold, \
#                 novelty_detector, params_score)   
            
# ------------------------ incDFM original Iterative Thresholding --------------------------------------------

class Score_Simple():
    def __init__(self):
        '''
        Get top X percentile 
        '''
    def compute(self, scores_old, scores_new):
        scores = scores_old/scores_new
        return scores
    
    
def weighted_average(weights, values):
    avg = sum(weight * value for weight, value in zip(weights, values)) / sum(weights)
    return int(avg)

# compute threshold percentile value
class Thresholding_Iter_per_Val():
    def __init__(self, params):
        '''
        Use validation set to establish threshold for novelty
        Do not predetermine number of novel samples. 
        Just go selecting according to percentile and validation threshold. 
        '''
        
        self.max_iter = params.max_iter_pseudolabel
        
        self.threshold_percentile_iter = params.threshold_percentile
        self.percentile_val_threshold = params.percentile_val_threshold
        
        
        self.num_samples = params.num_samples
        self.num_samples_init = self.num_samples
        
        
        self.detector_params = params.params_score
        

        self.force_end = False
        self.num_iters = 0
        self.history_select=[]
        
        self.num_new_max_history = []
        
        
        self.selection_rate = []
        

    def update_n_select(self, th_percentile, scores, scores_val_i):
        '''
        Assumption that at each thresholding round, 
        all pesudolabeled new actually were True New 
        '''
        # percentile_val_threshold = 100 - self.percentile_val_threshold
        
        # set validation threshold 
        th_i_val = np.percentile(scores_val_i, self.percentile_val_threshold)
        # dynamic quantity, maximum quantity of new left over 
        num_new_max = np.where(scores > th_i_val)[0].shape[0]
        
        
        # estimated_select = int(num_new_max*th_percentile)
        # smoothen a bit percentile threshold 
        th_percentile_current = weighted_average(self.selection_rate+[th_percentile], self.selection_rate+[th_percentile])     
        
        th_percentile_current = (100-th_percentile_current)/100
        self.selection_rate.append(th_percentile_current)

        
        # This number will go decreasing 
        self.num_new_max_history.append(num_new_max)
        print('num_new_max_history', self.num_new_max_history)
        print('Estimated New left', num_new_max)
        print('selection_rate', self.selection_rate)

        if num_new_max==0:
            self.force_end=True
            select_s=0
        else:
            # but have to introduce a graded selection here
            select_s = int(num_new_max*th_percentile_current)
            # select_s = current_select
            if select_s>num_new_max:
                select_s = int(num_new_max*self.percentile_val_threshold/100)
                self.force_end = True
            
        # update history
        self.num_iters += 1
        if self.num_iters>self.max_iter:
            select_s = int(num_new_max*self.percentile_val_threshold/100) #subtract error from validation
            # select_s = max(num_new_max - sum(self.history_select),0)
            self.force_end = True
            
            
        self.num_samples -= select_s
        self.history_select.append(select_s)
        
        print('select_s', select_s, th_i_val)
        print('history_select', self.history_select)

        
        if select_s>0:
            inds_novel = self.select(select_s, scores)
        else:
            inds_novel = np.array([])
            self.force_end=True
            

        return inds_novel, self.force_end, th_i_val
    
    
    def select(self, select_s, scores):
        
        inds_sort = np.argsort(scores) # ascending order         
        inds_novel = inds_sort[-int(select_s):]
        
        return inds_novel
         



class Threshold_n_Select_Iters():
    def __init__(self, params):
        '''
        Select high-initial-threshold
        Use those samples to further iterate 
        Use validation set to establish threshold for novelty, incrementally
        '''
        self.ScoreNew = Score_Simple()
        
        self.TH_iter = Thresholding_Iter_per_Val(params)

        self.params = params

        self.pseudo_novel_inds = np.array([]) # increasing indices of pseudolabeld "novel" with reference to current_dset indices 
        
        self.validation_loader = params.validation_iid_loader


    def fit_new_PCA(self, pseudolabeled_new_loader):

        # create
        self.novelty_detector_new = novel.NoveltyDetector().create_detector(type=self.params.novelty_detector_name, params=self.params.detector_params)

        # fit with increasing novel_populated data
        processed_data = futils.extract_features(self.params.params_score.feature_extractor, pseudolabeled_new_loader, \
        target_ind=self.params.dset_prep['scenario_classif'], homog_ind=1, 
        device=self.params.device, use_raw_images=self.params.use_image_as_input, raw_image_transform=self.params.add_tf)

        # ----Update memory and Novelty detector (just has one label in this current setting)
        dfm_x = processed_data[0][self.params.dfm_layers_input]
        dfm_y = processed_data[1]
        self.novelty_detector_new.fit_total(dfm_x.T, dfm_y)


    def select_novel(self, t, current_dset, novelty_detector, noveltyResults):
        
        inds_current_dset = np.arange(current_dset.__len__())


        # compute past PCA scores (for already fitted classes)
        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=self.params.batchsize_test,
                                                shuffle=False, num_workers=self.params.num_workers)
        _, gt_novelty, scores_past, _, _, dset_inds = novelty_detector.score(current_loader, vars(self.params.params_score))
        
        # compute PCA scores (past) for validation set
        _, _, scores_val_past, _, _, _ = novelty_detector.score(self.validation_loader, vars(self.params.params_score))


        print('scores_past_old', scores_past[current_loader.dataset.novelty_y==0], np.array(scores_past[current_loader.dataset.novelty_y==0]).mean())
        print('scores_past_new', scores_past[current_loader.dataset.novelty_y==1], np.array(scores_past[current_loader.dataset.novelty_y==1]).mean())


        # ======1) Iteration 1 --> only use old_scores 
        inds_pred_novel, force_end,_  = self.TH_iter.update_n_select(self.params.threshold_percentile, scores_past, scores_val_past)
        # append selected indices 
        self.pseudo_novel_inds = np.concatenate((self.pseudo_novel_inds, inds_pred_novel)).astype(int)


        # Get scores (accuracies etc) per iteration
        accs_iter = noveltyResults.compute_accuracies(t, 1, self.pseudo_novel_inds, gt_novelty)
        

        
        print('Iter %d - Acc  All/Precision %.3f/%.3f: inds_pred_novel %d/%d/%d - Force_End %s'%(\
        self.TH_iter.num_iters, accs_iter[0], accs_iter[2], \
        inds_pred_novel.shape[0], sum(self.TH_iter.history_select), self.TH_iter.num_new_max_history[-1], str(self.TH_iter.force_end)))
            
        # sys.exit()
            

        scores_current = scores_past
        # ======2) Iterations >1
        force_end = False
        while force_end==False:
            
            # remove the already selected labels 
            inds_nonlabeled = np.setdiff1d(inds_current_dset, self.pseudo_novel_inds)
            print('inds_nonlabeled', inds_nonlabeled.shape, self.pseudo_novel_inds.shape)


            # ===== Fit PCA on pseudolabeled samples from past iteration =====
            pseudolabeled_new_loader = torch.utils.data.DataLoader(dwrap.PseudolabelDset(current_dset, pred_novel_inds=self.pseudo_novel_inds), \
                                batch_size=self.params.batchsize_test, shuffle=False, num_workers=self.params.num_workers)
            self.fit_new_PCA(pseudolabeled_new_loader)


            # ===== Compute scores for novel iteration
            _, _, scores_new_current, _, _, _ = self.novelty_detector_new.score(current_loader, vars(self.params.params_score))
            _, _, scores_new_current_val, _, _, _ = self.novelty_detector_new.score(self.validation_loader, vars(self.params.params_score))
            scores_current = self.ScoreNew.compute(scores_past, scores_new_current)
            scores_current_val = self.ScoreNew.compute(scores_val_past, scores_new_current_val)


            print('scores_current_old', scores_current[current_loader.dataset.novelty_y==0], np.array(scores_current[current_loader.dataset.novelty_y==0]).mean())
            print('scores_current_new', scores_current[current_loader.dataset.novelty_y==1], np.array(scores_current[current_loader.dataset.novelty_y==1]).mean())


            # Select Indices for novel Iteration
            inds_pred_novel, force_end, th_val = self.TH_iter.update_n_select(self.params.threshold_percentile, scores_current[inds_nonlabeled], scores_current_val)
            if inds_pred_novel.shape[0]>0:
                inds_pred_novel = inds_nonlabeled[inds_pred_novel]
                self.pseudo_novel_inds = np.concatenate((self.pseudo_novel_inds, inds_pred_novel.astype(int))).astype(int)


                accs_iter = noveltyResults.compute_accuracies(t, self.TH_iter.num_iters, self.pseudo_novel_inds, gt_novelty)
                num_estimated_new_i = np.where(scores_current > th_val)[0].shape[0]


                print('Iter %d - Acc  All/Precision %.3f/%.3f: inds_pred_novel %d/%d/%d - Force_End %s'%(\
                self.TH_iter.num_iters, accs_iter[0], accs_iter[2], \
                inds_pred_novel.shape[0], sum(self.TH_iter.history_select), num_estimated_new_i, str(self.TH_iter.force_end)))
            else:
                print('Iter %d - Force End:%s - None were selected - iteration interrupted and loop ends'%(self.TH_iter.num_iters, str(force_end)))
        # sys.exit()
            

        return self.pseudo_novel_inds, None



    def evaluate_CL(self, t, current_dset, novelty_detector, noveltyResults):
        '''
        Compute AUROC/AUPR + Accuracies/precision given actual selection
        '''

        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=self.params.batchsize_test,
                                            shuffle=False, num_workers=self.params.num_workers)

        gt, gt_novelty, scores_past, scores_dist_past, preds, dset_inds = novelty_detector.score(current_loader, vars(self.params.params_score))

        _, _, scores_current, scores_dist_current, _, _ = self.novelty_detector_new.score(current_loader, vars(self.params.params_score))
        
        # Select Indices for novel Iteration
        scores = self.ScoreNew.compute(scores_past, scores_current)

        # print('scores_dist_past', scores_dist_past.shape)
        # print('scores_dist_current', scores_dist_current.shape)

        scores_dist = np.concatenate((scores_dist_past, scores_dist_current), axis=0)


        # separate into True Old IID and New 
        inds_old = np.where(gt_novelty==0)[0]
        inds_new = np.where(gt_novelty==1)[0]

        # print('true novel scores', scores[inds_new], scores[inds_new].mean())
        # print('true old scores', scores[inds_old], scores[inds_old].mean())

        rep_old = acuracy_report(gt[inds_old], preds[inds_old], scores[inds_old])
        
        results = {}
        results['new_scores'] = scores[inds_new]
        results['old_scores'] = scores[inds_old]
        results['scores_dist'] = scores_dist
        results['gt']= gt
        results['gt_novelty']= gt_novelty
        results['accuracy'] = rep_old['accuracy']
        results['report'] = rep_old
        results['dset_inds'] = dset_inds
        results = Namespace(**results)


        results.scores = np.concatenate((results.new_scores, results.old_scores))
        auroc, aupr, aupr_norm, _,_,_,_ = scores_metrics_results(results.scores, results.gt_novelty)

        # plot results for best epsilon 
        noveltyResults.compute(t, results.gt_novelty, results.scores)

        results.auroc = auroc
        results.aupr = aupr
        results.aupr_norm = aupr_norm

        # save per class results 
        with open('%s/per_class_report_old.txt'%noveltyResults.dir_save, 'w') as outfile:
            json.dump(results.report, outfile, sort_keys = True, indent = 4)

        print('DFM Results -  Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(auroc, aupr, aupr_norm))
        print('Average Accuracy per class old %.4f'% results.accuracy)

        return results



# ----------------- Tug-incDFM ---------------------------------------------------------

# compute threshold percentile value
class Counter_Iters_Tug():
    def __init__(self, params):
        '''
        Keeps track of number of samples that can be selected through iterations 
        
        Use validation set to establish threshold for novelty, incrementally
        '''

        self.max_iter = params.max_iter_pseudolabel
        self.num_samples_init= params.num_samples

        self.mix_percent_new = params.current_old_new_ratio # ratio of new samples to all unknown
        
        # keep track of number chosen old and new for the current task 
        self.max_selected_new = np.floor(self.num_samples_init*self.mix_percent_new)
        self.max_selected_old = self.num_samples_init - self.max_selected_new
        
        
        # initialize counters 
        self.num_samples = self.num_samples_init
        self.num_rem_new = self.max_selected_new
        self.num_rem_old = self.max_selected_old
        self.history_select_new=[]
        self.history_select_old=[]
        self.num_iters = 0
        
        
        self.force_end_new = False
        self.force_end_old = False
    



    def update_new(self, th_percentile):
        '''
        Assumption that at each thresholding round, 
        all pesudolabeled new actually were True New 
        '''

        # top of distribution 
        th_percentile = 100-th_percentile
        
        # number to select as new (attempt)
        select_new = int((th_percentile/100)*self.num_samples_init)
        
        
        # check if we have already not selected the maximum amount of new (correction)
        if ((sum(self.history_select_new)+select_new)>self.max_selected_new):
            # if we have, decrease select_new to the allowed remaining amount
            select_new = self.max_selected_new - sum(self.history_select_new)
            th_percentile = select_new/self.num_samples_init
            self.force_end_new = True
            
            
        # check if we have selected all "old" (correction)
        if sum(self.history_select_old)>=self.max_selected_old:
            select_new = self.max_selected_new - sum(self.history_select_new)
            self.force_end_new=True


        # update counters 
        self.num_rem_new = self.num_rem_new - select_new
        self.num_samples = self.num_rem_new + self.num_rem_old
        self.history_select_new.append(select_new)


        return select_new, self.force_end_new 
    
    
    def update_old(self, th_percentile):
        
        #bottom of distribution
        # number to select as old
        # top of distribution 
        th_percentile = 100-th_percentile # ex: 100-85 = 15 
         
        select_old = int((th_percentile/100)*self.num_samples_init)
        
        # check if we have already not selected the maximum amount of old (correction)
        if ((sum(self.history_select_old)+select_old)>self.max_selected_old):
            # if we have, decrease select_new to the allowed remaining amount
            select_old = self.max_selected_old - sum(self.history_select_old)
            th_percentile = select_old/self.num_samples_init
            self.force_end_old = True
        
        # check if we have selected all "new" (correction)
        if sum(self.history_select_new)>=self.max_selected_new:
            select_old = self.max_selected_old - sum(self.history_select_old)
            self.force_end_old=True
            

        # update counters 
        self.num_rem_old = self.num_rem_old - select_old
        self.num_samples = self.num_rem_new + self.num_rem_old
        self.history_select_old.append(select_old)


        return select_old, self.force_end_old 
    
    
    def update(self, th_percentile):
        
        self.num_iters +=1
        
        if not self.force_end_new:
            select_new, self.force_end_new = self.update_new(th_percentile)
        else:
            select_new = 0
        
        # if self.force_end:
        #     return select_new, 0, force_end

        if not self.force_end_old:
            select_old, self.force_end_old = self.update_old(th_percentile)
        else:
            select_old = 0
            
        # check if we have overdone the amount of iters 
        if self.num_iters>self.max_iter:
            force_end = True
        else:
            force_end = self.force_end_new and self.force_end_old
        
        return select_new, select_old, force_end
        
        
    
        




class Threshold_Tug_incDFM():
    def __init__(self, params):
        '''
        Select high-initial-threshold
        Use those samples to further iterate 
        Do this for both "old" and "new"
        Use validation set to establish threshold for novelty, incrementally

        '''
        self.ScoreMethod = ThScores.Score_Tug_Simple(params)
        
        self.TH_iter = Counter_Iters_Tug(params)

        self.params = params

        self.pseudo_novel_inds = np.array([]) # increasing indices of pseudolabeld "novel" with reference to current_dset indices 
        self.pseudo_old_inds = np.array([]) # increasing indices of pseudolabeld "old" with reference to current_dset indices 


    def fit_new_PCA(self, pseudolabeled_new_loader):

        # create
        DFM = novel.NoveltyDetector().create_detector(type=self.params.novelty_detector_name, params=self.params.detector_params)

        # fit with increasing novel_populated data
        processed_data = futils.extract_features(self.params.params_score.feature_extractor, pseudolabeled_new_loader, \
        target_ind=self.params.dset_prep['scenario_classif'], homog_ind=1, 
        device=self.params.device, use_raw_images=self.params.use_image_as_input, raw_image_transform=self.params.add_tf)

        # ----Update memory and Novelty detector (just has one label in this current setting)
        dfm_x = processed_data[0][self.params.dfm_layers_input]
        dfm_y = processed_data[1]
        DFM.fit_total(dfm_x.T, dfm_y)
        
        return DFM



    def select_novel(self, t, current_dset, novelty_detector, noveltyResults):

        inds_current_dset = np.arange(current_dset.__len__())
        print('inds_current_dset', inds_current_dset.shape)


        # ===== Compute past PCA scores (for already fitted classes)
        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=self.params.batchsize_test,
                                                shuffle=False, num_workers=self.params.num_workers)
        _, gt_novelty, scores_old_past, _, _, dset_inds = novelty_detector.score(current_loader, vars(self.params.params_score))

        # print('scores_old_past for old', scores_old_past[current_loader.dataset.novelty_y==0], np.array(scores_old_past[current_loader.dataset.novelty_y==0]).mean())
        # print('scores_old_past for new', scores_old_past[current_loader.dataset.novelty_y==1], np.array(scores_old_past[current_loader.dataset.novelty_y==1]).mean())
        # print('self.params.th_percentile', self.params.th_percentile, self.TH_iter.num_samples, self.TH_iter.max_selected_new, self.TH_iter.max_selected_old)


        # ======1) Iteration 0 --> only use scores_old_past
        select_new, select_old, force_end = self.TH_iter.update(self.params.threshold_percentile)
        inds_pred_novel, inds_pred_old = self.ScoreMethod.select(0, scores_old_past, None, select_new, select_old)
        
        
        self.pseudo_novel_inds = np.concatenate((self.pseudo_novel_inds, inds_pred_novel)).astype(int)
        self.pseudo_old_inds = np.concatenate((self.pseudo_old_inds, inds_pred_old)).astype(int)

        # Get scores (accuracies etc) per iteration
        accs_iter = noveltyResults.compute_accuracies(t, 1, self.pseudo_novel_inds, gt_novelty)
        print('Iter %d - Acc  All/Precision %.3f/%.3f: inds_pred_novel %d/%d/%d inds_pred_old %d/%d/%d force_end new/old %s/%s'%(1, accs_iter[0], accs_iter[2], \
                inds_pred_novel.shape[0], sum(self.TH_iter.history_select_new), self.TH_iter.max_selected_new, \
                inds_pred_old.shape[0], sum(self.TH_iter.history_select_old), self.TH_iter.max_selected_old, self.TH_iter.force_end_new, self.TH_iter.force_end_old))


        # ======2) Iterations >0
        force_end = False
        while force_end==False:
            
            # Indices still unlabeled/unknown
            inds_nonlabeled = np.setdiff1d(inds_current_dset, np.concatenate((self.pseudo_novel_inds, self.pseudo_old_inds)))
            print('inds_nonlabeled', inds_nonlabeled.shape, self.pseudo_novel_inds.shape)


            # Fit PCA on newly pseudolabeled samples 
            pseudolabeled_new_loader = torch.utils.data.DataLoader(dwrap.PseudolabelDset(current_dset, pred_novel_inds=self.pseudo_novel_inds), \
                                batch_size=self.params.batchsize_test, shuffle=False, num_workers=self.params.num_workers)
            pseudolabeled_old_loader = torch.utils.data.DataLoader(dwrap.PseudolabelDset(current_dset, pred_novel_inds=self.pseudo_old_inds), \
                                batch_size=self.params.batchsize_test, shuffle=False, num_workers=self.params.num_workers)
            self.DFM_new = self.fit_new_PCA(pseudolabeled_new_loader)
            self.DFM_old = self.fit_new_PCA(pseudolabeled_old_loader)


            # compute scores for novel iteration            
            select_new, select_old, force_end = self.TH_iter.update(self.params.threshold_percentile)
            _, _, scores_new_current, _, _, _ = self.DFM_new.score(current_loader, vars(self.params.params_score))
            _, _, scores_old_current, _, _, _ = self.DFM_old.score(current_loader, vars(self.params.params_score))
            scores_old_i = self.ScoreMethod.combine_score_old(scores_old_current, scores_old_past)


            # eval - measure 
            # print('scores_new_current for new', scores_new_current[current_loader.dataset.novelty_y==1], np.array(scores_new_current[current_loader.dataset.novelty_y==1]).mean())
            # print('scores_new_current for old', scores_new_current[current_loader.dataset.novelty_y==0], np.array(scores_new_current[current_loader.dataset.novelty_y==0]).mean())
            # print('scores_old_current for new', scores_old_current[current_loader.dataset.novelty_y==1], np.array(scores_old_current[current_loader.dataset.novelty_y==1]).mean())
            # print('scores_old_current for old', scores_old_current[current_loader.dataset.novelty_y==0], np.array(scores_old_current[current_loader.dataset.novelty_y==0]).mean())


            # Select Indices for novel Iteration - new_preds and old_preds indices
            inds_pred_novel, inds_pred_old = self.ScoreMethod.select(self.TH_iter.num_iters, scores_old_i[inds_nonlabeled], scores_new_current[inds_nonlabeled], \
                select_new, select_old)
            inds_pred_novel = inds_nonlabeled[inds_pred_novel]
            inds_pred_old = inds_nonlabeled[inds_pred_old]
            self.pseudo_novel_inds = np.concatenate((self.pseudo_novel_inds, inds_pred_novel)).astype(int)
            self.pseudo_old_inds = np.concatenate((self.pseudo_old_inds, inds_pred_old)).astype(int)

            accs_iter = noveltyResults.compute_accuracies(t, self.TH_iter.num_iters, self.pseudo_novel_inds, gt_novelty)
            
            print('Iter %d - Acc  All/Precision %.3f/%.3f: inds_pred_novel %d/%d/%d inds_pred_old %d/%d/%d, force_end new/old %s/%s'%(\
                self.TH_iter.num_iters, accs_iter[0], accs_iter[2], \
                inds_pred_novel.shape[0], sum(self.TH_iter.history_select_new), self.TH_iter.max_selected_new, \
                inds_pred_old.shape[0], sum(self.TH_iter.history_select_old), self.TH_iter.max_selected_old, self.TH_iter.force_end_new, self.TH_iter.force_end_old))
            
            
        return self.pseudo_novel_inds, self.pseudo_old_inds



    def evaluate_CL(self, t, current_dset, novelty_detector, noveltyResults):

        current_loader = torch.utils.data.DataLoader(current_dset, batch_size=self.params.batchsize_test,
                                            shuffle=False, num_workers=self.params.num_workers)


        # Get scores for past 
        gt, gt_novelty, scores_past, scores_dist_past, preds, dset_inds = novelty_detector.score(current_loader, vars(self.params.params_score))



        # get scores for current 
        _, _, scores_new_current, scores_new_dist_current, _, _ = self.DFM_new.score(current_loader, vars(self.params.params_score))
        _, _, scores_old_current, scores_old_dist_current, _, _ = self.DFM_old.score(current_loader, vars(self.params.params_score))

        
        scores_old = self.ScoreMethod.combine_score_old(scores_old_current, scores_past)
        scores_dist_old = self.ScoreMethod.combine_score_old(scores_old_current, scores_dist_past)
        
        # combine scores 
        scores = self.ScoreMethod.compute(scores_old, scores_new_current)
        scores_dist = np.concatenate((scores_dist_old, scores_new_dist_current), axis=0)


        # separate into True Old IID and New 
        inds_old = np.where(gt_novelty==0)[0]
        inds_new = np.where(gt_novelty==1)[0]


        rep_old = acuracy_report(gt[inds_old], preds[inds_old], scores[inds_old])
        results = {}
        results['new_scores'] = scores[inds_new]
        results['old_scores'] = scores[inds_old]
        results['scores_dist'] = scores_dist
        results['gt']= gt
        results['gt_novelty']= gt_novelty
        results['accuracy'] = rep_old['accuracy']
        results['report'] = rep_old
        results['dset_inds'] = dset_inds
        results = Namespace(**results)
        results.scores = np.concatenate((results.new_scores, results.old_scores))
        
        
        # Get aupr, etc 
        auroc, aupr, aupr_norm, _,_,_,_ = scores_metrics_results(results.scores, results.gt_novelty)


        # plot results 
        noveltyResults.compute(t, results.gt_novelty, results.scores)

        results.auroc = auroc
        results.aupr = aupr
        results.aupr_norm = aupr_norm

        # save per class results 
        with open('%s/per_class_report_old.txt'%noveltyResults.dir_save, 'w') as outfile:
            json.dump(results.report, outfile, sort_keys = True, indent = 4)

        print('DFM Results -  Auroc %.3f, Aupr %.3f, Aupr_norm %.3f'%(auroc, aupr, aupr_norm))
        print('Average Accuracy per class old %.4f'% results.accuracy)

        return results







# ------------------- Simple Threshold ---------------------------------------------------------------------

def Threshold_n_Select_Simple(task, results, estimated_threshold_novel, th_percentile_confidence=-1,  metric_confidence=('variance', 'min')):
    '''
    Use validation set to establish threshold for novelty, incrementally
    '''
    scores = results.scores # scores or scores_dist???
    
    # print('scores', scores[:100])
    # print('estimated_threshold_novel', estimated_threshold_novel)

    inds_novel = np.where(scores>estimated_threshold_novel)[0] # binary classification (above th is considered to be Novelty (Positive class))

    print('num samples in', scores.shape, 'num pred as novel', inds_novel.shape)
    
    if th_percentile_confidence>0:
        # Prefer low variance? 
        confs,_ = confidence_scores(results.scores_dist[...,inds_novel], metric=metric_confidence[0])
        threshold_conf=np.percentile(confs, th_percentile_confidence) 
        if metric_confidence[1]=='min':
            inds_conf= np.where(confs<threshold_conf)[0] # get X lowest confidences
        else:
            inds_conf= np.where(confs>threshold_conf)[0] # get X lowest confidences
        inds_novel = inds_novel[inds_conf]

        return inds_novel, scores[inds_novel], confs[inds_conf]

    # if inds_novel.shape[0]>0:
    return inds_novel, scores[inds_novel]




def estimate_threshold_from_val_simple(validation_loader_iid, percentile_val_threshold, novelty_detector, \
    params_detector, novelty_detector_name):
    
    # params_detector includes feature extractor 
    _, _, scores_val_past, _, _, dset_inds = novelty_detector.score(validation_loader_iid, params_detector)
    
    if novelty_detector_name=='odin' or novelty_detector_name=='softmax':
        scores_val_past = -scores_val_past

    estimated_threshold_novel = np.percentile(scores_val_past, percentile_val_threshold)

    print('scores_val_past', scores_val_past[:100])
    print('Threshold', estimated_threshold_novel)
    
    return estimated_threshold_novel


