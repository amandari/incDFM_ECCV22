import os
from sklearn.decomposition import PCA, FastICA
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import abc
import sys
import time


import matplotlib

from tqdm import tqdm 

matplotlib.use('Agg')

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from novelty_ODD.DFM_utils import singleclass_gaussian, mcd_gaussian
import novelty_ODD.ODIN_utils as odutils

class NoveltyDetector():
    def create_detector(self, type: str, params):
        if type.lower() == 'dfm':
            return DFM(params)
        elif type.lower() == 'odin':
            return GenerlizedODIN(params)


class NoveltyDetectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit_total(self, features, ground_truth):
        # completely fit ODD on novel data
        pass

    def get_loss(self, batch):
        # only do one batch iteration of ODD
        pass

    @abc.abstractmethod
    def score(self, dataloader, feature_extractor, layer):
        pass


class DFM(NoveltyDetectorInterface):
    def __init__(self, params) -> None:
        super().__init__()

        self.name = 'dfm'
        self.score_type = params['score_type'] if params['score_type'] is not None else 'pca'
        self.pca_level = params['pca_level'] if params['pca_level'] is not None else 0.995
        self.embed_type = params['score_type'] if (params['score_type'] != 'nll') else 'pca'

        self.n_components = params['n_components']
        self.n_percent_comp = params['n_percent_comp']

        self.device = params['device']
        self.target_ind = params['target_ind']
        self.pca_mats = dict()
        self.prob_models = dict()


    def fit_total(self, features, ground_truth):
        ground_truth = ground_truth.cpu().numpy().astype('int64')
        num_categories = len(np.unique(ground_truth))
        # for l in args.layers:
        features = features.cpu().numpy()

        if self.n_components == 'None' and self.score_type=='ica':
            print('blah')
            self.n_components = int(self.n_percent_comp*features.shape[0])
            print('n_comp', self.n_components)
        else:
            print('n_comp var', self.pca_level)

        self.stored_labels = []
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

        self.stored_labels = np.array(self.stored_labels)

        for j in np.unique(ground_truth):
            idx = ground_truth == j
            data_layer = features[:, idx]
            features_reduced = self.pca_mats[j].transform(data_layer.T)
            if self.score_type == 'nll':
                self.prob_models[j] = singleclass_gaussian()
                self.prob_models[j].fit(features_reduced.T)
            elif self.score_type == 'robust_nll':
                print('robust')
                self.prob_models[j] = mcd_gaussian(self.support_fraction)
                self.prob_models[j].fit(features_reduced.T)



        # if self.score_type == 'nll':
        #     for j in np.unique(ground_truth):
        #         idx = ground_truth == j
        #         data_layer = features[:, idx]
        #         features_reduced = self.pca_mats[j].transform(data_layer.T)
        #         self.prob_models[j] = singleclass_gaussian()
        #         self.prob_models[j].fit(features_reduced.T)
        # elif self.score_type == 'robust_nll':
        #     for j in np.unique(ground_truth):
        #         idx = ground_truth == j
        #         data_layer = features[:, idx]
        #         features_reduced = self.pca_mats[j].transform(data_layer.T)
        #         self.prob_models[j] = singleclass_gaussian()
        #         self.prob_models[j].fit(features_reduced.T)


        print('end fit', time.time()-start)
        # sys.exit()


    def get_loss(self, batch):
        return None


    def score(self, dataloader, params):

        feature_extractor = params['feature_extractor']
        feature_extractor.model.eval()
        feature_extractor.model.base.train(False)

        layer = params['layer']

        num_categories = len(self.pca_mats)
        len_dataset = len(dataloader.dataset)
        gt = torch.empty(len_dataset)
        scores = np.zeros((num_categories, len_dataset))

        with torch.no_grad():
            count = 0

            for k, batch in enumerate(dataloader):
                data = batch[0]
                labels = batch[self.target_ind]
                inputs = data.to(self.device)
                labels = labels.to(self.device)
                num_im = inputs.shape[0]
                _, features = feature_extractor(inputs)

                oi = features[layer].cpu().numpy()
                scores_pca = np.zeros((oi.shape[0], num_categories))
                scores_nll = np.zeros((oi.shape[0], num_categories))
                for j, l in enumerate(self.pca_mats):
                    pca_mat = self.pca_mats[l]
                    # print(l, j)
                    oi_or = oi
                    oi_j = pca_mat.transform(oi)
                    oi_reconstructed = pca_mat.inverse_transform(oi_j)
                    scores_pca[:, j] = np.sum(np.square(oi_or - oi_reconstructed), axis=1)
                    if self.score_type == 'nll' or self.score_type == 'robust_nll':
                        scores_nll[:, j] = self.prob_models[j].score_samples(oi_j)
                    
                if self.score_type == 'nll' or self.score_type == 'robust_nll':
                    scores[:, count: count + num_im] = -scores_nll.T
                else:
                    scores[:, count: count + num_im] = scores_pca.T

                gt[count:count + num_im] = labels
                count += num_im

            # get min score (ex, min reconstruction distance)
            preds = np.argmin(scores, axis=0)
            scores = scores.min(axis=0)
            gt = gt.numpy().astype(int)

        return gt, scores, preds





class GenerlizedODIN(NoveltyDetectorInterface):
    def __init__(self, params) -> None:
        super().__init__()

        self.name='odin'

        self.h_dict = {
            'cosine':   odutils.CosineDeconf,
            'inner':    odutils.InnerDeconf,
            'baseline': odutils.InnerDeconf,
            'euclid':   odutils.EuclideanDeconf
        }
        self.device = params['device']
        self.num_epochs = params['num_epochs']
        self.target_ind = params['target_ind']

        self.baseline = (params['similarity'] == 'baseline')
        self.criterion = params['criterion']
        self.noise_magnitude = params['noise_magnitude']
        self.score_func = params['score_func']

        # ----- set up ODIN layers (logit quotient)
        self.h = self.h_dict[params['similarity']](params['base_network'].classifier_penultimate, params['num_classes'])
        self.h = self.h.to(self.device)
        self.deconf_net = odutils.DeconfNet(params['base_network'], params['base_network'].classifier_penultimate, params['num_classes'], \
                                                                        self.h, self.baseline)
        self.deconf_net = self.deconf_net.to(self.device)

        self.parameters = []
        self.h_parameters = []
        for name, parameter in self.deconf_net.named_parameters():
            if name == 'h.h.weight' or name == 'h.h.bias':
                self.h_parameters.append(parameter)
            else:
                self.parameters.append(parameter)

        self.optimizer = optim.SGD(self.parameters, lr = 0.1, momentum = 0.9, weight_decay=params['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = [int(self.num_epochs * 0.5), int(self.num_epochs * 0.75)], gamma = 0.1)

        self.h_optimizer = optim.SGD(self.h_parameters, lr = 0.1, momentum = 0.9) # No weight decay
        self.h_scheduler = optim.lr_scheduler.MultiStepLR(self.h_optimizer, milestones = [int(self.num_epochs * 0.5), int(self.num_epochs * 0.75)], gamma = 0.1)
            
        _,_,_,self.r_std,self.g_std,self.b_std = odutils.averages_variances()
        

    def fit_total(self, features, ground_truth):
        pass 

    def get_loss(self, inputs, targets, base_apply=True):

        # ----- run only one epoch 
        self.h_optimizer.zero_grad()
        self.optimizer.zero_grad()
        
        logits, _ , _ = self.deconf_net(inputs, base_apply=base_apply)
        loss = self.criterion(logits, targets)
        
        return loss, logits


    def score(self, dataloader, params):
        '''
        Have to return ground_truth and scores 
        '''
        self.deconf_net.eval()
        num_batches = len(dataloader)

        results = []
        gt = []
        preds = []
        data_iter = tqdm(dataloader)
        
        for j, batch in enumerate(data_iter):
            images = batch[0]
            labels = batch[self.target_ind]
            data_iter.set_description(' Scores | Processing image batch %d/%d' %((j + 1), num_batches))
            images = Variable(images.to(self.device), requires_grad = True)
            # print('images', images.shape)
            logits, h, g = self.deconf_net(images, base_apply=params['base_apply_score'])

            if self.score_func == 'h':
                scores = h
            elif self.score_func == 'g':
                scores = g
            elif self.score_func == 'logit':
                scores = logits

            # Calculating the perturbation we need to add, that is,
            # the sign of gradient of the numerator w.r.t. input
            scores = self.add_gradient(images, scores)

            # print(scores)
            # print(scores.data.cpu(), scores.shape)

            results.extend(list(scores.data.cpu().numpy().max(0)))
            preds.extend(list(np.argmax(scores.data.cpu().numpy(), axis=0)))
            gt.extend(labels.numpy()) 
        
        data_iter.close()

        return np.array(gt), np.array(results), np.array(preds)


    def add_gradient(self, input_tensor, scores):

        max_scores, _ = torch.max(scores, dim = 1)
        max_scores.backward(torch.ones(len(max_scores)).to(self.device))

        # Normalizing the gradient to binary in {-1, 1}
        if input_tensor.grad is not None:
            gradient = torch.ge(input_tensor.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[::, 0] = (gradient[::, 0] )/self.r_std
            gradient[::, 1] = (gradient[::, 1] )/self.g_std
            gradient[::, 2] = (gradient[::, 2] )/self.b_std
            # Adding small perturbations to images
            tempInputs = torch.add(input_tensor, gradient, alpha=self.noise_magnitude)
        
            # Now calculate score
            logits, h, g = self.deconf_net(tempInputs)

            if self.score_func == 'h':
                scores = h
            elif self.score_func == 'g':
                scores = g
            elif self.score_func == 'logit':
                scores = logits

        return scores
        
