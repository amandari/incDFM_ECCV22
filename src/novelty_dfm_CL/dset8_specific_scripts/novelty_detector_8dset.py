import os
from tkinter import image_names
from sklearn.decomposition import PCA, FastICA
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import abc
import sys
import time


import matplotlib

from tqdm import tqdm 

matplotlib.use('Agg')

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from novelty_ODD.DFM_utils import singleclass_gaussian
import novelty_dfm_CL.dset8_specific_scripts.ODIN_utils_8dset as odutils

from novelty_dfm_CL.novelty_utils import *


class NoveltyDetector():
    def create_detector(self, type: str, params):
        if type.lower() == 'dfm':
            return DFM(params)
        if type.lower() == 'incdfm':
            return DFM(params)
        elif type.lower() == 'odin':
            return GenerlizedODIN(params)
        elif type.lower() == 'softmax':
            return SoftmaxOOD(params)
        elif type.lower() == 'mahal':
            return Mahalanobis(params)


class NoveltyDetectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit_total(self, features, ground_truth):
        # completely fit ODD on novel data
        pass

    def get_loss(self, inputs, targets):
        # only do one batch iteration of ODD
        pass

    @abc.abstractmethod
    def score(self, dataloader, params):
        pass


class DFM(NoveltyDetectorInterface):
    def __init__(self, params) -> None:
        super().__init__()
        '''Continual Version'''

        self.name = 'dfm'
        self.score_type = params['score_type'] if params['score_type'] is not None else 'pca'
        self.pca_level = params['pca_level'] if params['pca_level'] is not None else 0.995
        self.embed_type = params['score_type'] if (params['score_type'] != 'nll') else 'pca'
        self.feature_extractor = params['feature_extractor']

        self.n_components = params['n_components']
        self.n_percent_comp = params['n_percent_comp']

        
        self.pca_mats = dict()
        self.prob_models = dict()

        self.stored_labels = []


        self.device = params['device']
        # TODO make compatible with core50, cifar100 which are multilabel datasets 
        # self.target_ind = params['target_ind'] #if core50, cifar100 which are multilabel datasets 


    def fit_total(self, features, ground_truth):
        # for l in args.layers:
        ground_truth = ground_truth.numpy()
        features = features.cpu().numpy()

        if self.n_components == 'None' and self.score_type=='ica':
            # print('blah')
            self.n_components = int(self.n_percent_comp*features.shape[0])
            # print('n_comp', self.n_components)
        else:
            pass
            # print('n_comp var', self.pca_level)

        # sys.exit()
        start = time.time()
        # features_reduced = dict()
        for j in np.unique(ground_truth):
            self.stored_labels.append(j)
            idx = ground_truth == j
            data_layer = features[:, idx]
            if self.embed_type=='pca':
                self.pca_mats[j] = PCA(self.pca_level)
            elif self.embed_type=='ica':
                self.pca_mats[j]= FastICA(n_components=self.n_components) # set 

            self.pca_mats[j].fit(data_layer.T)


        if self.score_type == 'nll':
            for j in np.unique(ground_truth):
                idx = ground_truth == j
                data_layer = features[:, idx]
                features_reduced = self.pca_mats[j].transform(data_layer.T)
                self.prob_models[j] = singleclass_gaussian()
                self.prob_models[j].fit(features_reduced.T)

        # print('end fit', time.time()-start)
        

    def get_loss(self, inputs, targets):
        return None


    def score(self, dataloader, params, v2=True):
        '''
        dataloader returns (sample, class_lbl, novelty_lbl, index) in each batch index
        '''

        # feature_extractor = params['feature_extractor']
        self.feature_extractor.model.eval()
        # feature_extractor.model.base.train(False)

        layer = params['layer']

        num_categories = len(self.pca_mats)
        len_dataset = len(dataloader.dataset)

        # Output arrays 
        gt = np.zeros((len_dataset,)) # class labes 
        scores = np.zeros((num_categories, len_dataset))

        if v2:
            gt_novelty = np.zeros((len_dataset,)) # binary labels 
            indexes = np.zeros((len_dataset,))


        with torch.no_grad():
            count = 0

            for k, batch in enumerate(dataloader):
                data = batch[0]
                labels = batch[1]

                if v2:
                    labels_novelty = batch[3]

                inputs = data.to(self.device)
                
                _, features = self.feature_extractor(inputs)

                oi = features[layer].cpu().numpy()
                
                num_im = oi.shape[0]

                
                # print(inputs.shape, num_im, oi.shape, features[layer].shape, num_categories, len_dataset)

                # sys.exit()

                scores_pca = np.zeros((oi.shape[0], num_categories)) # Num_samples x categories
                scores_nll = np.zeros((oi.shape[0], num_categories))

                for j, l in enumerate(self.pca_mats):
                    pca_mat = self.pca_mats[l]
                    oi_or = oi
                    oi_j = pca_mat.transform(oi)
                    oi_reconstructed = pca_mat.inverse_transform(oi_j)
                    scores_pca[:, j] = np.sum(np.square(oi_or - oi_reconstructed), axis=1)
                    if self.score_type == 'nll':
                        scores_nll[:, j] = self.prob_models[j].score_samples(oi_j)
                
                if self.score_type == 'nll':
                    scores[:, count: count + num_im] = -scores_nll.T
                else:
                    scores[:, count: count + num_im] = scores_pca.T

                gt[count:count + num_im] = labels
                
                if v2:
                    gt_novelty[count:count + num_im] = labels_novelty
                    indexes[count:count + num_im] = batch[-1]

                count += num_im

            # get min score (ex, min reconstruction distance)
            preds = np.argmin(scores, axis=0)
            scores_min = scores.min(axis=0)

            gt = gt.astype(int)
            if v2:
                gt_novelty = gt_novelty.astype(int)
                indexes = indexes.astype(int)

        if v2:
            return gt, gt_novelty, scores_min, scores, preds, indexes
        else:
            return gt,None,scores_min,scores,preds,None






class Mahalanobis(NoveltyDetectorInterface):
    def __init__(self, params) -> None:
        super().__init__()
        '''Continual Version'''

        self.name = 'mahal'
        # self.pca_level = params['pca_level'] if params['pca_level'] is not None else 0.995

        self.num_components = params['num_components']
        self.balance = params['balance_classes']
        self.feature_extractor = params['feature_extractor']

        
        #re-computed every task
        self.pca_model = None
        self.u_data = None
        self.sigma_data = None

        # appended every task
        self.mean_vec = dict()

        self.stored_labels = []


        self.device = params['device']


    def fit_total(self, features, ground_truth):
        '''
        features will include all classes seen up until this point
        Junction of coreset + new_data
        '''
        # for l in args.layers:
        features = features.cpu().numpy()
        ground_truth = ground_truth.numpy()

        unique_labels, unique_counts = np.unique(ground_truth, return_counts=True)
        if self.balance:
            # balance classes (downsample novel)
            min_count = np.min(unique_counts)
            inds_lb_keep = np.array([])
            for i, lb in enumerate(unique_labels):
                inds_lb = np.where(ground_truth==lb)[0]
                if unique_counts[i]>min_count:
                    # reduce number of counts (downsample)
                    inds_lb = np.random.permutation(inds_lb)
                    inds_lb = inds_lb[:min_count]                    
                inds_lb_keep = np.concatenate((inds_lb_keep, inds_lb))
            inds_lb_keep = inds_lb_keep.astype(int)
            # print('features', features.shape, inds_lb_keep)
            features_keep = features[..., inds_lb_keep]
            ground_truth_keep = ground_truth[inds_lb_keep]

        else:
            features_keep = features
            ground_truth_keep = ground_truth


        self.stored_labels = []
        start = time.time()
        self.pca_model = PCA(self.num_components)
        print('features before PCA fit mahalanobis', features.shape, features_keep.shape)
        self.pca_model.fit(features_keep.T) # need input as N x D 
        # print('features', features.shape)
        # features_reduced_keep = self.pca_model.transform(features_keep.T)
        features_reduced = self.pca_model.transform(features_keep.T)
        # print('features_reduced', features_reduced.shape)
        for j in unique_labels:
            # get means per class 
            self.stored_labels.append(j)
            idx = ground_truth_keep == j
            # print('idx', idx.shape)
            data_layer = features_reduced[idx,...]
            # print('data_layer', data_layer.shape)
            self.mean_vec[j] = np.mean(data_layer, axis=0) # subtract mean per class from reduced features 
            features_reduced[idx,...] -= np.expand_dims(self.mean_vec[j], 0)
        # print('features_reduced.T', features_reduced.T.shape)
        self.u_data, self.sigma_data, _ = np.linalg.svd(features_reduced.T, full_matrices=False)
        # print('self.u_data, self.sigma_data', self.u_data.shape, self.sigma_data.shape)
        print('end fit', time.time()-start)


    def get_loss(self, inputs, targets):
        return None


    def score(self, dataloader, params, v2=True):
        '''
        dataloader returns (sample, class_lbl, novelty_lbl, index) in each batch index
        '''

        self.feature_extractor.model.eval()
        # feature_extractor.model.base.train(False)

        layer = params['layer']

        num_categories = len(self.mean_vec)
        len_dataset = len(dataloader.dataset)

        # Output arrays 
        gt = np.zeros((len_dataset,)) # class labes 
        scores = np.zeros((num_categories, len_dataset))

        if v2:
            gt_novelty = np.zeros((len_dataset,)) # binary labels 
            indexes = np.zeros((len_dataset,))


        with torch.no_grad():
            count = 0

            for k, batch in enumerate(dataloader):
                data = batch[0]
                labels = batch[1]

                if v2:
                    labels_novelty = batch[3]

                inputs = data.to(self.device)
                num_im = inputs.shape[0]
                
                _, embeddings = self.feature_extractor(inputs)
                features = embeddings[layer].cpu().numpy()
                features_reduced = self.pca_model.transform(features)
                scores_nll = np.zeros((features.shape[0], num_categories))

                for j in self.mean_vec:
                    v = np.dot(features_reduced - np.expand_dims(self.mean_vec[j], axis=0), self.u_data / self.sigma_data)
                    scores_nll[:, j] = np.sum(v * v, axis=1)
                
                scores[:, count: count + num_im] = scores_nll.T

                gt[count:count + num_im] = labels
                
                if v2:
                    gt_novelty[count:count + num_im] = labels_novelty
                    indexes[count:count + num_im] = batch[-1]

                count += num_im

            # get min score (ex, min reconstruction distance)
            preds = np.argmin(scores, axis=0)
            scores_min = scores.min(axis=0)

            gt = gt.astype(int)
            if v2:
                gt_novelty = gt_novelty.astype(int)
                indexes = indexes.astype(int)

        if v2:
            return gt, gt_novelty, scores_min, scores, preds, indexes
        else:
            return gt,None,scores_min,scores,preds,None




class GenerlizedODIN(NoveltyDetectorInterface):
    def __init__(self, params) -> None:
        super().__init__()
        '''Continual Version'''

        self.name='odin'

        self.h_dict = {
            'cosine':   odutils.CosineDeconf,
            'inner':    odutils.InnerDeconf,
            'baseline': odutils.InnerDeconf,
            'euclid':   odutils.EuclideanDeconf
        }
        
        self.num_tasks = params['num_tasks']
        self.device = params['device']
        self.num_epochs = params['num_epochs']
        self.baseline = (params['similarity'] == 'baseline')
        self.criterion = params['criterion']

        
        self.num_classes = max(params['num_classes_fine_tasks'])
        self.lr_h = params['lr']
        self.train_technique = params['train_technique']

        # ----- set up ODIN layers (logit quotient)
        self.h = self.h_dict[params['similarity']](params['base_network'].output_size, params['num_classes_fine_tasks'], self.num_tasks)
        self.h = self.h.to(self.device)
        self.deconf_net = odutils.DeconfNet(params['base_network'], params['base_network'].output_size, params['num_classes_fine_tasks'], \
                                                                        self.num_tasks, self.h, self.baseline)
        self.deconf_net = self.deconf_net.to(self.device)

        self.h_parameters = []
        for name, parameter in self.deconf_net.named_parameters():
            if 'h.' in name:
                self.h_parameters.append(parameter)

        if self.train_technique==1:
            self.h_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.h_parameters), lr=self.lr_h)
            self.h_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.h_optimizer, 'min', patience=params['patience_lr'], factor=params['schedule_decay'], min_lr=0.00001)
        elif self.train_technique==2:
            self.h_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.h_parameters), lr=self.lr_h, momentum=0.9)
            self.h_scheduler = optim.lr_scheduler.StepLR(self.h_optimizer, step_size=params['step_size_epoch_lr'], gamma=params['gamma_step_lr'])
        elif self.train_technique==3:
            self.h_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.h_parameters), lr = self.lr_h, momentum = 0.9, weight_decay = 0.0001)
            self.h_scheduler = optim.lr_scheduler.MultiStepLR(self.h_optimizer, milestones = [int(self.num_epochs * 0.5), int(self.num_epochs * 0.75)], gamma = 0.1)
        elif self.train_technique==4:
            self.h_optimizer = optim.RMSprop(filter(lambda p: p.requires_grad,  self.h_parameters), lr = self.lr_h)
            self.h_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.h_optimizer, 'min', patience=params['patience_lr'], factor=params['schedule_decay'], min_lr=0.00001)

            
        _,_,_,self.r_std,self.g_std,self.b_std = odutils.averages_variances()
        

    def fit_total(self, features, ground_truth):
        pass 

    def get_loss(self, inputs,targets,  task_id, base_apply=True):

        # ----- run only one epoch 
        self.h_optimizer.zero_grad()        
        logits, _ , _ = self.deconf_net(inputs, task_id, base_apply=base_apply)
        loss = self.criterion(logits, targets)
        
        return loss, logits
    
    
    def compute_score_loop_psp(self, image, num_tasks, params):
        
        '''
        single image at a time - batchsize 1
        '''
        # with torch.no_grad():
        scores=[]
        inputs_grad = []
        max_val = []
        for t in range(num_tasks):
            _, h, _, input_can_grad = self.deconf_net.forward_test(image, t, params['device'], base_apply=params['base_apply_score'])
            scores.append(h)
            inputs_grad.append(input_can_grad)
            max_val.append(scores[-1].detach().max().item())
            # print('t', t, scores[-1], max_val[-1])
        t_select = np.argmax(np.array(max_val))
        
        return scores[t_select], inputs_grad[t_select], t_select


    def score(self, dataloader, params, v2=True):

        '''
        Use batchsize=1  
        '''
        self.deconf_net.eval()

        # data characteristics CL 
        len_dataset = len(dataloader.dataset)
        num_batches = len(dataloader)
        gt = np.zeros((len_dataset,)) # class labes 
        scores_dist = np.zeros((self.num_classes, len_dataset))

        scores_max = np.zeros((len_dataset,))
        preds = np.zeros((len_dataset,))

        if v2:
            gt_novelty = np.zeros((len_dataset,)) # binary labels 
            indexes = np.zeros((len_dataset,))
            
        data_iter = tqdm(dataloader)
        count = 0
        for j, batch in enumerate(data_iter):
            images = batch[0]
            label = batch[1]
            task_id = batch[2].item()
            num_im = images.shape[0]

            if v2:
                label_novelty = batch[3]

            # data_iter.set_description(' Scores | Processing image batch %d/%d' %((j + 1), num_batches))
            # print('images', images.shape)


            # print(images)
            # print(params)
            
            # print(images, params['current_task']+1, params['score_func'])

            if params['true_task'] is not None:
                # print('task_ID', batch[2].item())
                logits, h, g, input_can_grad = self.deconf_net.forward_test(images, task_id, params['device'], base_apply=params['base_apply_score'])
                data_iter.set_description(' Scores | Processing batches %d/%d using true_task (Oracle) ID' %((j + 1), num_batches))
                scores_ = h

            else:
                scores_, input_can_grad, task_id = self.compute_score_loop_psp(images, params['current_task']+1, params)
                data_iter.set_description(' Scores | Processing batches %d/%d - unknown task ID' %((j + 1), num_batches))



            # print(input_can_grad.grad, task_id, scores_, params)
            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of the numerator w.r.t. input
            scores_ = self.add_gradient(input_can_grad, task_id, scores_, params)

            # Log results 
            max_dim_c = scores_.data.shape[1]
            scores_dist[:max_dim_c, count:count + num_im] = scores_.data.cpu().numpy().T
            scores_max[count:count + num_im] = scores_.data.cpu().numpy().max()
            preds[count:count + num_im] = np.argmax(scores_.data.cpu().numpy())
            gt[count:count + num_im] = label.numpy()
            
            if v2:
                gt_novelty[count:count + num_im] = label_novelty
                indexes[count:count + num_im] = batch[-1]

            count += num_im
        data_iter.close()


        gt = gt.astype(int)
        if v2:
            gt_novelty = gt_novelty.astype(int)
            indexes = indexes.astype(int)

        if v2:
            return gt, gt_novelty, scores_max, scores_dist, preds, indexes
        else:
            return gt,None,scores_max,scores_dist,preds,None



    def add_gradient(self, input_tensor, task_id, scores, params, comp_base=False):

        # print('input_tensor', input_tensor.shape)

        max_scores, _ = torch.max(scores, dim = 1)
        max_scores.backward(torch.ones(len(max_scores)).to(self.device))

        # Normalizing the gradient to binary in {-1, 1}
        if input_tensor.grad is not None:
            # print('Exists input tensor grad', input_tensor.grad)
            gradient = torch.ge(input_tensor.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            if len(input_tensor.shape)>2:
                gradient[::, 0] = (gradient[::, 0] )/self.r_std
                gradient[::, 1] = (gradient[::, 1] )/self.g_std
                gradient[::, 2] = (gradient[::, 2] )/self.b_std
            # Adding small perturbations to images
            tempInputs = torch.add(input_tensor, gradient, alpha=params['noise_magnitude'])
        
            # Now calculate score
            # take as input input that has grad 
            _, scores, _, _ = self.deconf_net.forward_test(tempInputs, task_id, params['device'], base_apply=comp_base)

        #     print('no input tensor grad', input_tensor.grad)
        # sys.exit()

        return scores
        


 
# def forward(self, x, base_apply=True):

#     output = self.underlying_model.forward_nologit_odin(x, base_apply=base_apply)

#     # print('output.shape', output.shape)
#     numerators = self.h(output)
    
#     if self.baseline:
#         denominators = torch.unsqueeze(self.ones.expand(len(numerators)), 1)
#     else:
#         denominators = self.g(output)
    
#     # Now, broadcast the denominators per image across the numerators by division
#     quotients = numerators / denominators

#     # logits, numerators, and denominators
#     return quotients, numerators, denominators




class SoftmaxOOD(NoveltyDetectorInterface):
    def __init__(self, params) -> None:
        super().__init__()
        '''Continual Version'''

        self.name='softmax'
        self.num_classes_fine_tasks = params['num_classes_fine_tasks']
        
        self.num_classes = max(self.num_classes_fine_tasks)

        # ----- set up ODIN layers (logit quotient)

        self.net = params['base_network']
        
        self.device = params['device']
        
        self.softmax = nn.Softmax()


    def fit_total(self, features, ground_truth):
        pass 
    
    
    def compute_score_loop_psp(self, image, num_tasks):
        
        '''
        single image at a time - batchsize 1
        '''
        with torch.no_grad():
            scores=[]
            max_val = []
            for t in range(num_tasks):
                logits = self.net(image, t, base_apply=True)
                scores.append(self.softmax(logits))
                max_val.append(scores[-1].max().item())
                # print('t', t, scores[-1], max_val[-1])
                
            t_select = np.argmax(np.array(max_val))
        
        return scores[t_select], t_select
            
    

    def score(self, dataloader, params, v2=True):
        '''
        Have to return ground_truth and scores 
        TODO modify to be PSP key loop
        '''
        self.net.eval()

        # data characteristics CL 
        len_dataset = len(dataloader.dataset)
        num_batches = len(dataloader)
        gt = np.zeros((len_dataset,)) # class labes 
        scores_dist = np.zeros((self.num_classes, len_dataset))
        
        scores_max = np.zeros((len_dataset,))
        preds = np.zeros((len_dataset,))

        if v2:
            gt_novelty = np.zeros((len_dataset,)) # binary labels 
            indexes = np.zeros((len_dataset,))


        data_iter = tqdm(dataloader)
        count = 0
        for j, batch in enumerate(data_iter):
            # Here have batchsize of 1
            image = batch[0]
            label = batch[1]
            num_im = image.shape[0]
            
            
            # print('labels', labels, num_im)
            if v2:
                labels_novelty = batch[3]

            image = image.to(self.device)
            # print('images', images.shape)
            
            if params['true_task'] is not None:
                # print('task_ID', batch[2].item())
                logits = self.net(image, batch[2].item(), base_apply=True)
                scores_=self.softmax(logits)
                data_iter.set_description(' Scores | Processing batches %d/%d using true_task ID' %((j + 1), num_batches))
                # sys.exit()

            else:
                
                scores_, task_id_estimated = self.compute_score_loop_psp(image, params['current_task']+1)
                data_iter.set_description(' Scores | Processing batches %d/%d - unknown task ID' %((j + 1), num_batches))

            

            # print('scores_', scores_)

            # sys.exit()
            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of the numerator w.r.t. input
            # Log results 
            max_dim_c = scores_.data.shape[1]
            scores_dist[:max_dim_c, count:count + num_im] = scores_.data.cpu().numpy().T
            scores_max[count:count + num_im] = scores_.data.cpu().numpy().max()
            preds[count:count + num_im] = np.argmax(scores_.data.cpu().numpy())
            gt[count:count + num_im] = label.numpy()
            if v2:
                gt_novelty[count:count + num_im] = labels_novelty
                indexes[count:count + num_im] = batch[-1]

            count += num_im
            
            
            # if j==100:
            #     break
            
        data_iter.close()
        # preds = np.argmax(scores_dist, axis=0)
        # scores_max = scores_dist.max(axis=0)

        gt = gt.astype(int)
        if v2:
            gt_novelty = gt_novelty.astype(int)
            indexes = indexes.astype(int)

        if v2:
            return gt, gt_novelty, scores_max, scores_dist, preds, indexes
        else:
            return gt,None,scores_max,scores_dist,preds,None








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


