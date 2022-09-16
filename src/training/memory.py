import torch
import numpy as np
import sys
import utils 


# import feature_extraction.quantization as quant
'''
Replay Memory storage
'''

class CoresetDynamic():
    def __init__(self, total_size, input_layer_name='base.8', latent_layers=[], quantizer_dict=None, target_ind=-1, homog_ind=-2, device=0):
        
        '''Make corset for task assigning
            Dynamic Coreset - Does not grow, rather, accommodates; Has fixed size
            Make option of storing soft labels (passing by network to store) - Distillation loss 
            features_input (tensor)
            features_latent (dict): dict of tensors
            latent_layers (list): key=name of latent layer
        '''
        
        self.coreset_im = torch.zeros(0,0) ## start with zero dimension (initializer)
        self.coreset_t = torch.zeros(0,0).type(torch.LongTensor) ## start with zero dimension (initializer)
        self.coreset_homog = torch.zeros(0,0).type(torch.LongTensor)

        self.input_layer_name = input_layer_name
        if len(latent_layers)==0:
            self.apply_latent=False
        else:
            self.apply_latent=True

        self.coreset_latents={}
        for layer in latent_layers:
            self.coreset_latents[layer]=torch.zeros(0,0)
        
        self.coreset_homog_lbls = [] # cummulative homogeneous labels

        if quantizer_dict is not None:
            self.quantizer = quantizer_dict['pq']
            self.num_codebooks = quantizer_dict['num_codebooks']
            self.spatial_feat_dim = quantizer_dict['spatial_feat_dim']
            self.num_channels_init = quantizer_dict['num_channels_init']
        else:
            self.quantizer = None

        self.size_incoreset = self.coreset_im.shape[0]
        self.total_size = int(total_size)

        self.assignments_soft = None
        
        self.target_ind = target_ind
        self.homog_ind = homog_ind

        self.device = device


    def append_memory(self, new_data, homog_lbls_new):
        
        '''
        new_data (tuple): tensors of data (features_dict, target, homog_labels), output of extract_features function 
        the labels_homog should be used as the most fine-grained label. 
        '''

        print('*******Appending to CORESET******')
        if self.size_incoreset==0:
            room_new_all = int(self.total_size)
            self.room_left = int(self.total_size)
        else:
            room_new_all = int(self.total_size/len(list(set(self.coreset_homog_lbls + homog_lbls_new))))
            room_new_all = int(room_new_all*len(list(set(homog_lbls_new))))
            self.room_left = int(self.total_size-room_new_all) # for all past classes
            # print('self.room_left', self.room_left)
        
        self.room_new = room_new_all
        print('total_size', self.total_size, 'room_new', self.room_new, 'room_left', \
            self.room_left, 'coreset now', self.coreset_im.shape[0])
            
        # ============== Make Room for New Samples ==================
        # remove homoheneous number of old samples (already in coreset)
        if self.coreset_im.shape[0]>0:
            if self.apply_latent:
                self.coreset_im, self.coreset_t, self.coreset_homog, self.coreset_latents = self.make_room(self.room_left)  
            else:
                self.coreset_im, self.coreset_t, self.coreset_homog = self.make_room(self.room_left)  


        # =========== Separate input features and latent features, if applicable ==============
        if self.apply_latent:
            features, targets, labels_homog, latents = self.pick_features_new(new_data, self.room_new, homog_lbls_new)
        else:
            features, targets, labels_homog = self.pick_features_new(new_data, self.room_new, homog_lbls_new)


        # ---- Quantize the features for coreset 
        if self.quantizer is not None:
            # print('features before pq', features.shape)
            features = quant.encode(self.quantizer, features.numpy().astype('float32'), self.num_codebooks, num_channels_init=self.num_channels_init, spatial_feat_dim=self.spatial_feat_dim)
            features = torch.from_numpy(features).to(torch.uint8)
            # print('features after pq', features.shape)
            # sys.exit()

        # ======= Append new data to the coreset======
        if self.coreset_im.shape[0]>0:
            self.coreset_im = torch.cat((self.coreset_im, features), dim=0)
            self.coreset_t = torch.cat((self.coreset_t, targets), dim=0) ## target label is task
            self.coreset_homog = torch.cat((self.coreset_homog, labels_homog), dim=0) ## target label is task
            if self.apply_latent:
                for j, (key, val) in enumerate(latents.items()):
                    self.coreset_latents[key]=torch.cat((self.coreset_latents[key], val), dim=0)
        else:
            self.coreset_im = features
            self.coreset_t = targets
            self.coreset_homog = labels_homog
            if self.apply_latent:
                self.coreset_latents = latents
        

        self.size_incoreset = self.coreset_im.shape[0] ## should be kept constant now...
        self.coreset_homog_lbls = list(set(self.coreset_homog_lbls + homog_lbls_new)) # add labels for current lab



    def pick_features_new(self, new_data, number_generate, homog_lbls_new):
        '''
        Get correct amount of new data to store in coreset
        *latents is a dictionary
        '''

        features_dict, targets, labels_homog = new_data 
        if self.apply_latent:
            features = features_dict.pop(self.input_layer_name)
            latents = features_dict
        else:
            features = features_dict[self.input_layer_name]
            latents=None
        del features_dict

        # ---- shuffle
        inds_all = torch.randperm(features.shape[0]).long()
        features = features[inds_all,...]
        targets = targets[inds_all]
        labels_homog = labels_homog[inds_all]
        if self.apply_latent:
            for j, (key, val) in enumerate(latents.items()):
                latents[key] = val[inds_all,...]

        # select  only some images for coreset
        per_label_New = utils.divide_integer_K(number_generate, len(homog_lbls_new))
        selected = self.select_homogeneous(per_label_New, features, targets, labels_h=labels_homog, latents=latents)
        if self.apply_latent:
            features, targets, labels_homog, latents = selected 
        else:
            features, targets, labels_homog = selected 
        features = features.float()
        targets = targets.long()
        labels_homog = labels_homog.long()
        if self.apply_latent:
            for j, (key, val) in enumerate(latents.items()):
                latents[key] = val.float()

    
        if self.apply_latent:
            # print('generated', features.shape, latents[0][1].shape)
            return features, targets, labels_homog, latents
        else:
            return features, targets, labels_homog


    def make_room(self, room_left):
        '''
        room_left (int): room left on coreset so re-evaluate per-label allowances of old samples (already in coreset)
        Removes select samples to make space for new 
        Enforce homogeneity of labels 
        '''
        # ----- makes room based on unique labels (homogeneous sampling from each unique label)

        per_label_NT = utils.divide_integer_K(room_left, len(self.coreset_homog_lbls), shuff=True)
                    
        data_keep = self.select_homogeneous(per_label_NT, self.coreset_im, self.coreset_t, labels_h=self.coreset_homog, latents=self.coreset_latents)

        return data_keep


    def select_homogeneous(self, per_label_NT, samples, labels_t, labels_h, latents = []):
        '''
        per_label_NT (list): list of number of samples per label_homog
        samples (tensor): images or features to select from 
        labels_t (tensor): target labels of those images for classification purposes
        labels_homog (tensor, optional): labels used to establish homogeneity
        '''

        unique_labels_homog = np.unique(labels_h.numpy())
        for i, label in enumerate(unique_labels_homog):
            
            per_NT = per_label_NT[i]
            # get coreset input of specific label
            inds_coreset = torch.from_numpy(np.where(labels_h == label)[0]).long() ##indices of that class 
            label_images = samples[inds_coreset,...]
            label_labels = labels_t[inds_coreset]
            labels_homog = labels_h[inds_coreset]
            if self.apply_latent:
                labels_latents = {}
                for j, (key, val) in enumerate(latents.items()):
                    labels_latents[key] = val[inds_coreset, ...]

            # get random per_NT samples 
            chosen_inds = torch.randperm(label_images.shape[0])[:per_NT].long() 
            if i ==0:
                images_keep = label_images[chosen_inds,...]
                labels_keep = label_labels[chosen_inds]
                labels_homog_keep = labels_homog[chosen_inds]
            else:
                images_keep = torch.cat((images_keep, label_images[chosen_inds,...]), dim=0)
                labels_keep = torch.cat((labels_keep, label_labels[chosen_inds]), dim=0)
                labels_homog_keep = torch.cat((labels_homog_keep, labels_homog[chosen_inds]), dim=0)
            del label_images, label_labels, labels_homog

            if self.apply_latent:
                if i ==0:
                    labels_latents_keep = {}
                    for j, (key, val) in enumerate(labels_latents.items()):
                        labels_latents_keep[key] = val[chosen_inds, ...]
                else:
                    for j, (key, val) in enumerate(labels_latents.items()):
                        labels_latents_keep[key] = torch.cat((labels_latents_keep[key], val[chosen_inds,...]), dim=0)
                del labels_latents

            # print('after chosen', chosen_inds.shape, images_keep.shape, labels_latents_keep[0][1].shape)
        
        if self.apply_latent:
            return images_keep, labels_keep, labels_homog_keep, labels_latents_keep
        else:
            return images_keep, labels_keep, labels_homog_keep




# class CoresetDynamic():
#     def __init__(self, total_size, feature_extractor=None, quantizer_dict=None, target_ind=-1, homog_ind=-2):
        
#         '''Make corset for task assigning
#             Dynamic Coreset - Does not grow, rather, accommodates; Has fixed size
#             Make option of storing soft labels (passing by network to store) - Distillation loss 
#             multilabel (bool): If enforcing homogeneity through another label other then target classification label
#         '''
        
#         self.coreset_im = torch.zeros(0,0) ## start with zero dimension (initializer)
#         self.coreset_t = torch.zeros(0,0).type(torch.LongTensor) ## start with zero dimension (initializer)
#         self.coreset_homog = torch.zeros(0,0).type(torch.LongTensor)

#         self.coreset_homog_lbls = [] # cummulative homogeneous labels

#         self.feature_extractor = feature_extractor
#         if quantizer_dict is not None:
#             self.quantizer = quantizer_dict['pq']
#             self.num_codebooks = quantizer_dict['num_codebooks']
#             self.spatial_feat_dim = quantizer_dict['spatial_feat_dim']
#             self.num_channels_init = quantizer_dict['num_channels_init']
#         else:
#             self.quantizer = None

#         self.size_incoreset = self.coreset_im.shape[0]
#         self.total_size = int(total_size)

#         self.assignments_soft = None
        
#         self.target_ind = target_ind
#         self.homog_ind = homog_ind


#     def append_memory(self, dataset, homog_lbls_new):
        
#         '''
#         dataset (Torch Dataset Class): Containing both features and targets 
#         labels_homog (list of integers): labels to ensure homogeneity as tasks evolve
#         the labels_homog should be used as the most fine-grained label. 
#         for core50 that would be the instance label. 
#         # TODO refresh coreset within one label (in-out type update if label is re-seen)
#         '''

#         print('*******Appending to CORESET******')
#         if self.size_incoreset==0:
#             room_new_all = int(self.total_size)
#             self.room_left = int(self.total_size)
#         else:
#             room_new_all = int(self.total_size/len(list(set(self.coreset_homog_lbls + homog_lbls_new))))
#             room_new_all = int(room_new_all*len(list(set(homog_lbls_new))))
#             self.room_left = int(self.total_size-room_new_all) # for all past classes
#             # print('self.room_left', self.room_left)
        
#         self.room_new = room_new_all
#         print('total_size', self.total_size, 'room_new', self.room_new, 'room_left', \
#             self.room_left, 'coreset now', self.coreset_im.shape[0])
            
#         # ============== Make Room for New Samples ==================
#         # remove homoheneous number of old samples (already in coreset)
#         if self.coreset_im.shape[0]>0:
#             self.coreset_im, self.coreset_t, self.coreset_homog = self.make_room(self.room_left)  


#         # =========== feature Extraction selection of new task ==============
#         #TODO make this more efficient 
#         if self.feature_extractor is not None:
#             # print('Extract features to store in Coreset')
#             ld = torch.utils.data.DataLoader(dataset, batch_size=20,
#                                                     shuffle=True, num_workers=0)

#             features=torch.zeros(0,0)
#             while features.shape[0]<(self.room_new+1000) and features.shape[0]<dataset.__len__():
#                 batch = next(iter(ld))
#                 images = batch[0]
#                 with torch.no_grad():
#                     if features.shape[0]>0:
#                         features = torch.cat((self.feature_extractor(images.cuda()).cpu(), features), dim=0)
#                         targets = torch.cat((batch[self.target_ind], targets))
#                         labels_homog = torch.cat((batch[self.homog_ind], labels_homog))
#                     else:
#                         features = self.feature_extractor(images.cuda()).cpu()
#                         targets = batch[self.target_ind]
#                         labels_homog =batch[self.homog_ind]
#         else:
#             features = dataset.x
#             labels_homog = dataset.y[self.homog_ind,...]
#             targets = dataset.y[self.target_ind,...]


#         # ---- shuffle and select only some images for coreset
#         inds_all = torch.randperm(features.shape[0]).long()
#         features = features[inds_all,...]
#         targets = targets[inds_all]
#         labels_homog = labels_homog[inds_all]
#         per_label_New = utils.divide_integer_K(self.room_new, len(homog_lbls_new))
#         features, targets, labels_homog = self.select_homogeneous(per_label_New, features, targets, labels_h=labels_homog)
#         features = features.float()
#         targets = targets.long()
#         labels_homog = labels_homog.long()
#         # features = features[:self.room_new,...].float()
#         # targets = targets[:self.room_new].long()
#         # labels_homog = labels_homog[:self.room_new].long()

#         # ---- Quantize the features for coreset 
#         if self.quantizer is not None:
#             # print('features before pq', features.shape)
#             features = quant.encode(self.quantizer, features.numpy().astype('float32'), self.num_codebooks, num_channels_init=self.num_channels_init, spatial_feat_dim=self.spatial_feat_dim)
#             features = torch.from_numpy(features).to(torch.uint8)
#             # print('features after pq', features.shape)
#             # sys.exit()

#         # ======= Append the new to the coreset======
#         if self.coreset_im.shape[0]>0:
#             self.coreset_im = torch.cat((self.coreset_im, features), dim=0)
#             self.coreset_t = torch.cat((self.coreset_t, targets), dim=0) ## target label is task
#             self.coreset_homog = torch.cat((self.coreset_homog, labels_homog), dim=0) ## target label is task
#         else:
#             self.coreset_im = features
#             self.coreset_t = targets
#             self.coreset_homog = labels_homog
        

#         self.size_incoreset = self.coreset_im.shape[0] ## should be kept constant now...
#         self.coreset_homog_lbls = list(set(self.coreset_homog_lbls + homog_lbls_new)) # add labels for current lab


#     def make_room(self, room_left):
#         '''
#         room_left (int): room left on coreset so re-evaluate per-label allowances of old samples (already in coreset)
#         Removes select samples to make space for new 
#         Enforce homogeneity of labels 
#         '''
#         # ----- makes room based on unique labels (homogeneous sampling from each unique label)

#         per_label_NT = utils.divide_integer_K(room_left, len(self.coreset_homog_lbls), shuff=True)
                    
#         data_keep = self.select_homogeneous(per_label_NT, self.coreset_im, self.coreset_t, labels_h=self.coreset_homog)

#         return data_keep


#     def select_homogeneous(self, per_label_NT, samples, labels_t, labels_h=None):
#         '''
#         per_label_NT (list): list of number of samples per label_homog
#         samples (tensor): images or features to select from 
#         labels_t (tensor): target labels of those images for classification purposes
#         labels_homog (tensor, optional): labels used to establish homogeneity
#         '''
#         if labels_h is not None:
#             labels_h = labels_h
#         else:
#             labels_h = labels_t
#         unique_labels_homog = np.unique(labels_h.numpy())

#         for i, label in enumerate(unique_labels_homog):
            
#             per_NT = per_label_NT[i]
#             # get coreset input of specific label
#             inds_coreset = torch.from_numpy(np.where(labels_h == label)[0]).long() ##indices of that class 
#             label_images = samples[inds_coreset,...]
#             label_labels = labels_t[inds_coreset]
#             labels_homog = labels_h[inds_coreset]
            
#             # get random per_NT samples 
#             chosen_inds = torch.randperm(label_images.shape[0])[:per_NT].long() 
#             if i ==0:
#                 images_keep = label_images[chosen_inds,...]
#                 labels_keep = label_labels[chosen_inds]
#             else:
#                 images_keep = torch.cat((images_keep, label_images[chosen_inds,...]), dim=0)
#                 labels_keep = torch.cat((labels_keep, label_labels[chosen_inds]), dim=0)
#             del label_images, label_labels

#             if i ==0:
#                 labels_homog_keep = labels_homog[chosen_inds]
#             else:
#                 labels_homog_keep = torch.cat((labels_homog_keep, labels_homog[chosen_inds]), dim=0)
#             del labels_homog

#         return images_keep, labels_keep, labels_homog_keep

  