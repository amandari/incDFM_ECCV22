import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

import scipy.stats
import scipy.special 

import re

                
class latent_Nodes():
    def __init__(self, exp_type='random', p=0.5):
        self.exp_type = exp_type
        self.p = p # percentage 

        self.positions_classes={} #memory 


    def process_data(self, new_data, network):

        '''
        feats_data_dict (latents)
        Get node positions of nodes for all classes in dataset 
        if compute_out==True, generate reduced size features as well
        '''
        if len(new_data)>2:
            feats_data_dict, targets, other_labels=new_data
        else:
            feats_data_dict, targets =new_data
            other_labels=None

        unique_labels = np.unique(targets.numpy())


        for i, label in enumerate(unique_labels):
            inds_class = torch.from_numpy(np.where(targets.numpy() == label)[0]).long() ##indices of that class 
            
            # get data belonging to that class 
            # 1) features 
            feats_data_dict_class={}
            for l, (layer, l_features) in enumerate(feats_data_dict.items()):
                feats_data_dict_class[layer]=l_features[inds_class,...]

            latent_nodes_class = self.select_nodes_class(feats_data_dict_class, network)
            
            # Add to memory 
            if label in self.positions_classes:
                pass
            else:
                self.positions_classes[label]=latent_nodes_class

            dict_features_class_selected = self.get_latents_class(feats_data_dict_class, latent_nodes_class)

            if i==0:
                feats_processed = dict_features_class_selected
                targets_processed = targets[inds_class]
                if other_labels is not None:
                    other_labels_processed = other_labels[inds_class]
            else:
                for l, (layer, l_features) in enumerate(dict_features_class_selected.items()):
                    feats_processed[layer] = torch.cat((feats_processed[layer],l_features), dim=0)

                targets_processed = torch.cat((targets_processed,targets[inds_class]))
                if other_labels is not None:
                    other_labels_processed = torch.cat((other_labels_processed,other_labels[inds_class]))

        if other_labels is not None:
            return feats_processed, targets_processed, other_labels_processed
        else:
            return feats_processed, targets_processed
            


    def get_nodes(self, responses, num_activities, max_=True):

        if max_:
            nodes_chosen = np.argsort(responses)[::-1]
        else:
            nodes_chosen = np.argsort(responses)

        # num_activities = network.model.layer.out_features

        nodes = int(np.round(self.p*num_activities))
        nodes_chosen = nodes_chosen[:nodes]

        return nodes_chosen


    def select_nodes_class(self, dict_features_class, network):
        '''
        TODO - normalize? If doing per class is there need to do that?
        features (dictionary): dictionary of latent features for a given class. Keys are layer names and features are tensors
        return = latent_nodes (dictionary): dictionary containing selected output nodes per layer. Items are long index tensors 
        '''
        hardcoded = {'base.8':512, 'FC.3':4096, 'FC.1':7680}
        latent_nodes_class = {} # mask-like function  
        for l, (layer, l_features) in enumerate(dict_features_class.items()):
            # num_activities = network.getLayer(layer).out_features
            num_activities = hardcoded[layer]
            if self.p==1.0:
                latent_nodes_class[layer]=torch.from_numpy(np.arange(num_activities))
            else:
                if self.exp_type=='random':
                    num_nodes = int(np.round(self.p*num_activities))
                    nodes_array = np.random.permutation(np.arange(num_activities))[:min(num_nodes,num_activities)]
                    nodes_array.sort() 
                    latent_nodes_class[layer]=torch.from_numpy(nodes_array)

                elif self.exp_type=='max':
                    responses = np.amax(l_features.numpy(), axis=0)
                    chosen = np.copy(self.get_nodes(responses, num_activities))
                    latent_nodes_class[layer]= torch.from_numpy(chosen)
                elif self.exp_type=='entropy':
                    # select most entropic nodes over a class and normalize?? 
                    responses = scipy.special.entr(l_features.numpy()).sum(axis=0)
                    chosen = np.copy(self.get_nodes(responses, num_activities))
                    latent_nodes_class[layer]= torch.from_numpy(chosen)

                elif self.exp_type=='variance':
                    responses = np.var(l_features.numpy(), axis=0)
                    chosen = self.get_nodes(responses, num_activities)
                    latent_nodes_class[layer]= torch.from_numpy(chosen)
                else:
                    print('Method does not exist. Choose random, max, entropy or variance')

                    # maybe some genetic sampling algorithmn for choosing ....

        return latent_nodes_class


    def get_latents_class(self, dict_features_class, latent_nodes_class):
        '''
        Get appropriate nodes for latents, per class
        '''
        dict_features_class_selected={}
        for l, (layer, l_features) in enumerate(dict_features_class.items()):
            print(l_features.shape)
            dict_features_class_selected[layer]=l_features[...,latent_nodes_class[layer]]

        return dict_features_class_selected
            

# unmasking operation for loss. 


class DistanceLossLatents():
    def __init__(self, distance_type='cosine', eps=1e-6):
        self.per_class_nodes={} # simple dictionary for only one type of latent
        self.distance_type = distance_type

        if self.distance_type=='cosine':
            self.distance = nn.CosineSimilarity(dim=1, eps=eps)

    def __update__(self, class_nodes):
        for (c, nodes) in class_nodes.items():
            for key in nodes.keys():
                self.per_class_nodes[c]=nodes[key]

    def __fillPerclass__(self, x_computed, x_latent_sparse, y_targets):
        
        x_latent = x_computed.detach().clone()
        # x_latent.requires_grad = True
        indexes = torch.stack([self.per_class_nodes[c] for c in y_targets.cpu().numpy()])
        # print('indexes', indexes, 'x_latent', x_latent, 'x_latent_sparse', x_latent_sparse)

        x_latent[torch.repeat_interleave(torch.arange(x_latent.shape[0]), indexes.shape[1]), indexes.flatten()]=x_latent_sparse.flatten()

        return x_latent

    def __call__(self, x_computed, x_latent_sparse, y_targets):
        '''
        x_computed (tensor)
        x_latent_sparse (tensor)
        y_targets (tensor)
        '''
        x_latent = self.__fillPerclass__(x_computed, x_latent_sparse, y_targets)
        dist = self.distance(x_computed, x_latent)

        # average mean of distance 
        dist = dist.mean()
            
        return dist


