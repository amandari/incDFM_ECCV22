import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

import numpy as np
import sys
# import feature_extraction.quantization as quant

'''
Train Epoch loop 
'''



def train_main_epoch(epoch, loader_new, loader_old, model_main, optimizer, OOD_class, weight_old=0.5,\
                                feature_name='base.8', latent_name=None, latent_criterion=None, weight_latent=0, quantizer=None,  \
                                target_ind=1, cuda=True, display_freq=10,  cut_epoch_short=False, device=0):

    # model_main.model.train()
    print('Number batches: ', len(loader_new))
    
    if loader_old is not None:
        num_batches = max(len(loader_new), len(loader_old))
    else:
        num_batches = len(loader_new)
        
    for batch_idx in range(num_batches):

        # for new data 
        batch = next(iter(loader_new))
        feat_new = batch[0]
        t_new = batch[target_ind]

        if cuda:
            feat_new, t_new = feat_new.to(device), t_new.to(device)
            
            


        #1)  Feature Extraction 
        _,feat_new = model_main(feat_new, base_apply=True)
        feat_new=feat_new[feature_name]
        if quantizer is not None:
            feat_new = quant.encode_n_decode(feat_new.cpu().numpy().astype(np.float32), quantizer['pq'], num_channels=quantizer['num_channels_init'], spatial_feat_dim=quantizer['spatial_feat_dim'])
            feat_new = torch.from_numpy(feat_new).to(torch.float32)
            if cuda:
                feat_new = feat_new.to(device)
        feat_new, t_new = Variable(feat_new), Variable(t_new)
        
        #2) Forward Pass new data
        optimizer.zero_grad()
        if OOD_class.name == 'odin':
            loss_new, output_new = OOD_class.get_loss(feat_new,  t_new, base_apply=False)
        else:
            output_new,_ = model_main(feat_new, base_apply=False)
            loss_new = F.cross_entropy(output_new, t_new)


        # Forward pass Old data
        if loader_old is not None:
            if latent_name is not None:
                feat_old, t_old, latent_old = next(iter(loader_old))
            else:
                feat_old, t_old = next(iter(loader_old))

            if len(feat_old.shape)>2:
                # raw images in coreset
                base_apply_old = True 
            else:
                base_apply_old=False

            if quantizer is not None:
                feat_old = quant.decode(feat_old.numpy(), quantizer['pq'], quantizer['num_codebooks'], num_channels=quantizer['num_channels_init'], spatial_feat_dim=quantizer['spatial_feat_dim'])
                feat_old = torch.from_numpy(feat_old).to(torch.float32)

            if cuda:
                feat_old, t_old = feat_old.to(device), t_old.to(device)
            feat_old, t_old = Variable(feat_old), Variable(t_old)

            if OOD_class.name == 'odin':
                loss_old, output_old = OOD_class.get_loss(feat_old, t_old, base_apply=base_apply_old)
            else:
                output_old, output_latent_old = model_main(feat_old, base_apply=base_apply_old)
                loss_old = F.cross_entropy(output_old, t_old)


            if latent_name is not None:
                output_latent_old = output_latent_old[latent_name]
                latent_old = latent_old[latent_name].to(device)
                loss_old_lat = latent_criterion(output_latent_old, latent_old, t_old)
                loss_old = loss_old_lat*weight_latent + (1-weight_latent)*loss_old

            loss = (1-weight_old)*loss_new + weight_old*loss_old
        else:
            loss = loss_new 

        loss.backward()
        optimizer.step()
        if OOD_class.name == 'odin':
            OOD_class.h_optimizer.step()

        pred_new = output_new.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct_new = pred_new.eq(t_new.data.view_as(pred_new)).cpu().sum()

        if loader_old is not None:
            pred_old = output_old.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct_old = pred_old.eq(t_old.data.view_as(pred_old)).cpu().sum()

        if batch_idx % display_freq == 0:
            if loader_old is not None:
                print('Train Epoch: %d [%d/%d (%.3f)]\tLoss: %.4f\tAcc_Train_new:%.3f\tAcc_Train_coreset:%.3f'%(
                    epoch, batch_idx, len(loader_new),
                    100. * batch_idx / len(loader_new), loss.item(), float(correct_new.item()/feat_new.shape[0]), float(correct_old.item()/feat_old.shape[0])  )
                )
                if latent_name is not None:
                    print('Latent loss: %.4f'%(loss_old_lat.item()))
            else:
                print('Train Epoch: %d [%d/%d (%.3f)]\tLoss: %.4f\tAcc_Train_new:%.3f'%(
                    epoch, batch_idx, len(loader_new),
                    100. * batch_idx / len(loader_new), loss.item(), float(correct_new.item()/feat_new.shape[0]))
                )
        
        if cut_epoch_short:
            if batch_idx == 10:
                break

