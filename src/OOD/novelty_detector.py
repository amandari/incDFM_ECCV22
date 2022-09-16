from sklearn.decomposition import PCA, FastICA
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import abc
import sys
import time
import matplotlib
from tqdm import tqdm 
matplotlib.use('Agg')


sys.path.append('../')

# sys.path.insert(1, os.path.join(sys.path[0], '..'))

from OOD.DFM_utils import singleclass_gaussian
import OOD.ODIN_utils as odutils

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

        feature_extractor = params['feature_extractor']
        feature_extractor.model.eval()
        feature_extractor.model.base.train(False)

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
                    labels_novelty = batch[2]

                inputs = data.to(self.device)
                num_im = inputs.shape[0]
                
                _, features = feature_extractor(inputs)

                oi = features[layer].cpu().numpy()

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
        print('features_reduced', features_reduced.shape)
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

        feature_extractor = params['feature_extractor']
        feature_extractor.model.eval()
        feature_extractor.model.base.train(False)

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
                    labels_novelty = batch[2]

                inputs = data.to(self.device)
                num_im = inputs.shape[0]
                
                _, embeddings = feature_extractor(inputs)
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
        self.device = params['device']
        self.num_epochs = params['num_epochs']
        self.baseline = (params['similarity'] == 'baseline')
        self.criterion = params['criterion']

        self.num_classes = params['num_classes']
        self.lr_h = params['lr']
        self.train_technique = params['train_technique']

        # ----- set up ODIN layers (logit quotient)
        self.h = self.h_dict[params['similarity']](params['base_network'].output_size, params['num_classes'])
        self.h = self.h.to(self.device)
        self.deconf_net = odutils.DeconfNet(params['base_network'], params['base_network'].output_size, params['num_classes'], \
                                                                        self.h, self.baseline)
        self.deconf_net = self.deconf_net.to(self.device)

        self.h_parameters = []
        for name, parameter in self.deconf_net.named_parameters():
            if name == 'h.h.weight' or name == 'h.h.bias':
                self.h_parameters.append(parameter)
 

        self.h_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.h_parameters), lr = 0.1, momentum = 0.9) # No weight decay
        self.h_scheduler = optim.lr_scheduler.MultiStepLR(self.h_optimizer, milestones = [int(self.num_epochs * 0.5), int(self.num_epochs * 0.75)], gamma = 0.1)


        if self.train_technique==1:
            self.h_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.h_parameters), lr=self.lr_h)
            self.h_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.h_optimizer, 'min', patience=params['patience_lr'], factor=params['schedule_decay'], min_lr=0.00001)
        elif self.train_technique==2:
            self.h_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.h_parameters), lr=self.lr_h, momentum=0.9)
            self.h_scheduler = optim.lr_scheduler.StepLR(self.h_optimizer, step_size=params['step_size_epoch_lr'], gamma=params['gamma_step_lr'])
        elif self.train_technique==3:
            self.h_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.h_parameters), lr = self.lr_h, momentum = 0.9, weight_decay = 0.0001)
            self.h_scheduler = optim.lr_scheduler.MultiStepLR(self.h_optimizer, milestones = [int(self.num_epochs * 0.5), int(self.num_epochs * 0.75)], gamma = 0.1)
    
            
        _,_,_,self.r_std,self.g_std,self.b_std = odutils.averages_variances()
        

    def fit_total(self, features, ground_truth):
        pass 

    def get_loss(self, inputs, targets, base_apply=True):

        # ----- run only one epoch 
        self.h_optimizer.zero_grad()        
        logits, _ , _ = self.deconf_net(inputs, base_apply=base_apply)
        loss = self.criterion(logits, targets)
        
        return loss, logits


    def score(self, dataloader, params, v2=True):
        '''
        Have to return ground_truth and scores 
        '''
        self.deconf_net.eval()

        # data characteristics CL 
        len_dataset = len(dataloader.dataset)
        num_batches = len(dataloader)
        gt = np.zeros((len_dataset,)) # class labes 
        scores_dist = np.zeros((self.num_classes, len_dataset))

        if v2:
            gt_novelty = np.zeros((len_dataset,)) # binary labels 
            indexes = np.zeros((len_dataset,))


        data_iter = tqdm(dataloader)
        count = 0
        for j, batch in enumerate(data_iter):
            images = batch[0]
            labels = batch[1]
            num_im = images.shape[0]

            if v2:
                labels_novelty = batch[2]

            data_iter.set_description(' Scores | Processing image batch %d/%d' %((j + 1), num_batches))
            # print('images', images.shape)
            logits, h, g, input_can_grad = self.deconf_net.forward_test(images, params['device'], base_apply=params['base_apply_score'])

            if params['score_func'] == 'h':
                scores_ = h
            elif params['score_func'] == 'g':
                scores_ = g
            elif params['score_func'] == 'logit':
                scores_ = logits

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of the numerator w.r.t. input
            scores_ = self.add_gradient(input_can_grad, scores_, params)

            # Log results 
            scores_dist[:, count: count + num_im] = scores_.data.cpu().numpy().T
            gt[count:count + num_im] = labels
            if v2:
                gt_novelty[count:count + num_im] = labels_novelty
                indexes[count:count + num_im] = batch[-1]

            count += num_im
        data_iter.close()


        preds = np.argmax(scores_dist, axis=0)
        scores_max = scores_dist.max(axis=0)

        gt = gt.astype(int)
        if v2:
            gt_novelty = gt_novelty.astype(int)
            indexes = indexes.astype(int)

        if v2:
            return gt, gt_novelty, scores_max, scores_dist, preds, indexes
        else:
            return gt,None,scores_max,scores_dist,preds,None



    def add_gradient(self, input_tensor, scores, params, comp_base=False):

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
            logits, h, g, _ = self.deconf_net.forward_test(tempInputs, params['device'], base_apply=comp_base)

            if params['score_func'] == 'h':
                scores = h
            elif params['score_func'] == 'g':
                scores = g
            elif params['score_func'] == 'logit':
                scores = logits
        # else:
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
        self.num_classes = params['num_classes']

        # ----- set up ODIN layers (logit quotient)

        self.net = params['base_network']
        
        self.device = params['device']
        
        self.softmax = nn.Softmax()


    def fit_total(self, features, ground_truth):
        pass 

    def score(self, dataloader, params, v2=True):
        '''
        Have to return ground_truth and scores 
        '''
        self.net.eval()

        # data characteristics CL 
        len_dataset = len(dataloader.dataset)
        num_batches = len(dataloader)
        gt = np.zeros((len_dataset,)) # class labes 
        scores_dist = np.zeros((self.num_classes, len_dataset))

        if v2:
            gt_novelty = np.zeros((len_dataset,)) # binary labels 
            indexes = np.zeros((len_dataset,))


        data_iter = tqdm(dataloader)
        count = 0
        for j, batch in enumerate(data_iter):
            images = batch[0]
            labels = batch[1]
            num_im = images.shape[0]

            if v2:
                labels_novelty = batch[2]

            data_iter.set_description(' Scores | Processing image batch %d/%d' %((j + 1), num_batches))
            images = images.to(self.device)
            # print('images', images.shape)
            logits = self.net(images, base_apply=True)
            scores_ = self.softmax(logits)

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of the numerator w.r.t. input
            # Log results 
            scores_dist[:, count: count + num_im] = scores_.data.cpu().numpy().T
            gt[count:count + num_im] = labels
            if v2:
                gt_novelty[count:count + num_im] = labels_novelty
                indexes[count:count + num_im] = batch[-1]

            count += num_im
            
        data_iter.close()
        preds = np.argmax(scores_dist, axis=0)
        scores_max = scores_dist.max(axis=0)

        gt = gt.astype(int)
        if v2:
            gt_novelty = gt_novelty.astype(int)
            indexes = indexes.astype(int)

        if v2:
            return gt, gt_novelty, scores_max, scores_dist, preds, indexes
        else:
            return gt,None,scores_max,scores_dist,preds,None



