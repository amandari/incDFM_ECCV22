import sys
from matplotlib import use
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

from tqdm import tqdm 
import re


def layer_names(network):
    num_layers = len(list(network.named_modules())) - 1
    for i, m in enumerate(network.named_modules()):
        mo = re.match('.+($|\n)', m[1].__repr__())
        print('Layer {}: {}: {}'.format(num_layers - i, m[0], mo.group(0).strip()))



def extract_features(feature_extractor, data_loader, target_ind=1, homog_ind=2, device=0, num_to_generate=None, use_raw_images=False, raw_image_transform=None):
    """
    Extract image features and put them into arrays.
    :param feature_extractor: network with hooks on layers for feature extraction
    :param data_loader: data loader of images for which we want features (images, labels, item_ixs)
    :param target_ind: target classification label position in array
    :param homog_ind: homogenizing label position in array
    :param num_to_generate: number of samples to extract features from 
    :return: torch tensors of features, labels
    """
    feature_extractor.model.eval()
    feature_extractor.model.base.train(False)
    feature_extractor.base_freeze = True
    for param in feature_extractor.model.base.parameters():
        param.requires_grad = False
    with torch.no_grad():
        # allocate space for features and labels
        features_dict={}
        if raw_image_transform is not None:
            tf = raw_image_transform
        # put features and labels into arrays
        num_gen = 0
        for batch_ix, batch in enumerate(tqdm(data_loader)):
            
            batch_x=batch[0]
            batch_y=batch[target_ind]
            batch_homog=batch[homog_ind]
            num_gen+=batch_x.shape[0]


            if raw_image_transform is not None:
                batch_feats = tf.apply(batch_x)
            else:
                batch_feats = batch_x

            _,batch_feats = feature_extractor(batch_feats.to(device))

            if use_raw_images:
                batch_feats['image'] = batch_x

            if batch_ix==0:
                for layer in feature_extractor.layer_names:
                    features_dict[layer] = batch_feats[layer].cpu().type(torch.float32)
                if use_raw_images:
                    features_dict['image'] = batch_feats['image'].cpu().type(torch.float32)
                    
                target_labels = batch_y.long()
                homog_labels = batch_homog.long()
            else:
                
                for layer in feature_extractor.layer_names:
                    features_dict[layer] = torch.cat((features_dict[layer], batch_feats[layer].cpu().type(torch.float32)), dim=0)
                if use_raw_images:
                    features_dict['image'] = torch.cat((features_dict['image'], batch_feats['image'].cpu().type(torch.float32)), dim=0)

                target_labels = torch.cat((target_labels, batch_y.long()), dim=0)
                homog_labels = torch.cat((homog_labels, batch_homog.long()), dim=0)

            if num_to_generate is not None:
                if num_gen>=num_to_generate:
                    for layer in feature_extractor.layer_names:
                        features_dict[layer] = features_dict[layer][:num_to_generate,...]
                    if use_raw_images:
                        features_dict['image']=features_dict['image'][:num_to_generate,...]
                    target_labels=target_labels[:num_to_generate]
                    homog_labels=homog_labels[:num_to_generate]
                    break
        
    return features_dict, target_labels, homog_labels


# def extract_images(data_loader, device=0,  target_ind=-1, homog_ind=-2, num_to_generate=None):

#     # allocate space for features and labels
#     features_dict={'input':None}
#     # put features and labels into arrays
#     num_gen = 0
#     for batch_ix, batch in enumerate(data_loader):
        
#         batch_x=batch[0]


#         batch_y=batch[target_ind]
#         batch_homog=batch[homog_ind]
#         num_gen+=batch_x.shape[0]


#         if batch_ix==0:
#             features_dict['input'] = batch_x.cpu().type(torch.float32)
#             target_labels = batch_y.long()
#             homog_labels = batch_homog.long()
#         else:

#             features_dict['input'] = torch.cat((features_dict[layer], batch_feats[layer].cpu().type(torch.float32)), dim=0)
#             target_labels = torch.cat((target_labels, batch_y.long()), dim=0)
#             homog_labels = torch.cat((homog_labels, batch_homog.long()), dim=0)

#         if num_to_generate is not None:
#             if num_gen>=num_to_generate:
#                 for layer in feature_extractor.layer_names:
#                     features_dict[layer] = features_dict[layer][:num_to_generate,...]
#                 target_labels=target_labels[:num_to_generate]
#                 homog_labels=homog_labels[:num_to_generate]
#                 break
        
#     return features_dict, target_labels, homog_labels
