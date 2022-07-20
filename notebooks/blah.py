
# compute threshold percentile value
class Counter_Iters_Tug():
    def __init__(self, params):
        '''
        Keeps track of number of samples that can be selected through iterations 
        
        Should we keep the percentile as from the initial number of samples? 
        '''

        self.max_iter = params.max_iter_pseudolabel
        self.num_samples_init= params.num_samples

        self.mix_percent_new = params.current_old_new_ratio # ratio of new samples to all unknown

        self.force_end = False

        # keep track of number chosen old and new for the current task 
        self.max_selected_new = np.floor(self.num_samples*self.mix_percent_new)
        self.max_selected_old = self.num_samples - self.max_selected_new
        
        
        # initialize counters 
        self.num_samples = self.num_samples_init
        self.num_rem_new = self.max_selected_new
        self.num_rem_old = self.max_selected_old
        self.history_select_new=[]
        self.history_select_old=[]
        self.num_iters_new = 0
        self.num_iters_old = 0



    def update_new(self, th_percentile):
        '''
        Assumption that at each thresholding round, 
        all pesudolabeled new actually were True New 
        '''

        # top of distribution 
        th_percentile = 100-th_percentile
        
        # number to select as new (attempt)
        select_new = int((th_percentile/100)*self.num_samples)
        
        
        # check if we have already not selected the maximum amount of new (correction)
        if ((sum(self.history_select_new)+select_new)>self.max_selected_new):
            # if we have, decrease select_new to the allowed remaining amount
            select_new = self.max_selected_new - sum(self.history_select_new)
            th_percentile = select_new/self.num_samples
            self.force_end = True
            
            
        # check if we have selected all "old" (correction)
        if sum(self.history_select_old)>=self.max_selected_old:
            select_new = self.max_selected_new - sum(self.history_select_new)
            self.force_end=True


        # update history
        self.num_iters_new += 1

        # check if we have overdone the amount of iters 
        if self.num_iters_new>self.max_iter:
            self.force_end = True


        # update counters 
        self.num_rem_new = self.num_rem_new - select_new
        self.num_samples = self.num_rem_new + self.num_rem_old
        self.history_select_new.append(select_new)


        return 100-th_percentile, self.force_end 
    
    
    def update_old(self, th_percentile):
        
        #bottom of distribution
        # number to select as old 
        select_old = int((th_percentile/100)*self.num_samples)
        
        # check if we have already not selected the maximum amount of old (correction)
        if ((sum(self.history_select_old)+select_old)>self.max_selected_old):
            # if we have, decrease select_new to the allowed remaining amount
            select_old = self.max_selected_old - sum(self.history_select_old)
            th_percentile = select_old/self.num_samples
            self.force_end = True
        
        # check if we have selected all "new" (correction)
        if sum(self.history_select_new)>=self.max_selected_new:
            select_old = self.max_selected_old - sum(self.history_select_old)
            self.force_end=True
            
            
        # update history
        self.num_iters_old += 1

        # check if we have overdone the amount of iters 
        if self.num_iters_old>self.max_iter:
            self.force_end = True


        # update counters 
        self.num_rem_old = self.num_rem_old - select_old
        self.num_samples = self.num_rem_new + self.num_rem_old
        self.history_select_old.append(select_old)


        return th_percentile, self.force_end 



class Score_Tug_Simple():
    def __init__(self, params):
        '''
        Get top X and bottom X percentile for new and old respectively
        '''
        self.w_old_i = params.w_old_i
        
    def combine_score_old(self, scores_old_i, scores_old_fixed):
        
        scores_old = self.w_old_i*scores_old_i + (1-self.w_old_i)*scores_old_fixed
        
        return scores_old
        
    def compute(self, scores_old, scores_new):
        scores = scores_old/scores_new
        return scores

    def select(self, i, scores_old, scores_new, th_percentile_score_new, th_percentile_score_old):
        
        if i==0:
            assert scores_new==None
            scores = scores_old            
        else:
            scores = self.compute(scores_old, scores_new)
            
        # compute thresholds 
        threshold_new=np.percentile(scores, th_percentile_score_new)
        threshold_old=np.percentile(scores, th_percentile_score_old)
    
        # get max values 
        inds_novel = np.where(scores>threshold_new)[0] # binary classification (above th is considered to be Novelty (Positive class))
        inds_old = np.where(scores<threshold_old)[0] # binary classification (above th is considered to be Novelty (Positive class))

        return inds_novel, inds_old
    




class Threshold_Tug_incDFM():
    def __init__(self, params):
        '''
        Select high-initial-threshold
        Use those samples to further iterate 
        Do this for both "old" and "new"
        '''
        self.ScoreMethod = Score_Tug_Simple(params)
        
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

        print('scores_old_past for old', scores_old_past[current_loader.dataset.novelty_y==0], np.array(scores_old_past[current_loader.dataset.novelty_y==0]).mean())
        print('scores_old_past for new', scores_old_past[current_loader.dataset.novelty_y==1], np.array(scores_old_past[current_loader.dataset.novelty_y==1]).mean())
        print('self.params.th_percentile', self.params.th_percentile, self.TH_iter.num_samples, self.TH_iter.max_selected_new, self.TH_iter.max_selected_old)


        # ======1) Iteration 0 --> only use scores_old_past
        th_percentile_new, force_end_new  = self.TH_iter.update_new(self.params.th_percentile)
        th_percentile_old, force_end_old  = self.TH_iter.update_old(self.params.th_percentile)
        force_end = force_end_new or force_end_old
        inds_pred_novel, inds_pred_old = self.ScoreMethod.select(scores_old_past, None, th_percentile_new, th_percentile_old)
        self.pseudo_novel_inds = np.concatenate((self.pseudo_novel_inds, inds_pred_novel)).astype(int)
        self.pseudo_old_inds = np.concatenate((self.pseudo_old_inds, inds_pred_old)).astype(int)

        # Get scores (accuracies etc) per iteration
        accs_iter = noveltyResults.compute_accuracies(t, 0, self.pseudo_novel_inds, gt_novelty)
        print('Iter %d - Acc  All/Precision %.3f/%.3f: inds_pred_novel %d/%d'%(0, accs_iter[0], accs_iter[2], \
            inds_pred_novel.shape[0], scores_old_past.shape[0]), 'percentile', th_percentile_new)


        scores_current = scores_old_past
        # ======2) Iterations >0
        force_end = False
        while force_end==False:
            
            # Indices still unlabeled/unknown
            inds_nonlabeled = np.setdiff1d(inds_current_dset, self.pseudo_novel_inds)
            print('inds_nonlabeled', inds_nonlabeled.shape, self.pseudo_novel_inds.shape)


            # Fit PCA on newly pseudolabeled samples 
            pseudolabeled_new_loader = torch.utils.data.DataLoader(PseudolabelDset(current_dset, pred_novel_inds=self.pseudo_novel_inds), \
                                batch_size=self.params.batchsize_test, shuffle=False, num_workers=self.params.num_workers)
            pseudolabeled_old_loader = torch.utils.data.DataLoader(PseudolabelDset(current_dset, pred_novel_inds=self.pseudo_old_inds), \
                                batch_size=self.params.batchsize_test, shuffle=False, num_workers=self.params.num_workers)
            self.DFM_new = self.fit_new_PCA(pseudolabeled_new_loader)
            self.DFM_old = self.fit_new_PCA(pseudolabeled_old_loader)


            # compute scores for novel iteration            
            th_percentile_new, force_end_new  = self.TH_iter.update_new(self.params.th_percentile)
            th_percentile_old, force_end_old  = self.TH_iter.update_old(self.params.th_percentile)
            force_end = force_end_new or force_end_old
            _, _, scores_new_current, _, _, _ = self.DFM_new.score(current_loader, vars(self.params.params_score))
            _, _, scores_old_current, _, _, _ = self.DFM_old.score(current_loader, vars(self.params.params_score))


            # eval - measure 
            print('scores_new_current for new', scores_new_current[current_loader.dataset.novelty_y==1], np.array(scores_new_current[current_loader.dataset.novelty_y==1]).mean())
            print('scores_new_current for old', scores_new_current[current_loader.dataset.novelty_y==0], np.array(scores_new_current[current_loader.dataset.novelty_y==0]).mean())
            print('scores_old_current for new', scores_old_current[current_loader.dataset.novelty_y==1], np.array(scores_old_current[current_loader.dataset.novelty_y==1]).mean())
            print('scores_old_current for old', scores_old_current[current_loader.dataset.novelty_y==0], np.array(scores_old_current[current_loader.dataset.novelty_y==0]).mean())


            # Select Indices for novel Iteration - new_preds and old_preds indices            
            scores_old_i = self.ScoreMethod.combine_score_old(scores_old_current, scores_old_past)
            inds_pred_novel, inds_pred_old = self.ScoreMethod.select(scores_old_i[inds_nonlabeled], scores_new_current[inds_nonlabeled], th_percentile_new, th_percentile_old)
            inds_pred_novel = inds_nonlabeled[inds_pred_novel]
            inds_pred_old = inds_nonlabeled[inds_pred_old]
            self.pseudo_novel_inds = np.concatenate((self.pseudo_novel_inds, inds_pred_novel)).astype(int)
            self.pseudo_old_inds = np.concatenate((self.pseudo_old_inds, inds_pred_old)).astype(int)

            print('Iter %d - inds_pred_novel %d/%d'%(self.TH_iter.num_iters_new, inds_pred_novel.shape[0], \
            scores_current.shape[0]), 'percentile', th_percentile_new)

            accs_iter = noveltyResults.compute_accuracies(t, self.TH_iter.num_iters, self.pseudo_novel_inds, gt_novelty)
            
            
            # Get scores (accuracies etc) per iteration
            accs_iter = noveltyResults.compute_accuracies(t, 0, self.pseudo_novel_inds, gt_novelty)
            print('Iter %d - Acc  All/Precision %.3f/%.3f: inds_pred_novel %d/%d - percentile %.2f'%(0, accs_iter[0], accs_iter[2], \
                inds_pred_novel.shape[0], scores_old_past.shape[0], th_percentile_new))

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

