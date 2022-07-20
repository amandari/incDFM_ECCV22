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



def train_main_epoch(epoch, task, max_classes_task, loader_new, model_main, optimizer, OOD_class,\
                                feature_name='base.8', \
                                target_ind=1, cuda=True, display_freq=10,  cut_epoch_short=False, device=0):

    '''
    Deals with PSP-Resnet, task-dependent
    '''
    # model_main.model.train()
    print('Number batches: ', len(loader_new))
    
    num_batches = len(loader_new)
        
    for batch_idx in range(num_batches):

        # for new data 
        batch = next(iter(loader_new))
        feat_new = batch[0]
        t_new = batch[1]


        
        # TODO clip the target array for wrong predicted "old" samples
        # For targets above num_existing, generate random integer distribution from 0, num_class_task_current 
        max_val_class = max_classes_task-1
        t_new_inds = np.where(t_new.numpy()>max_val_class)[0]
        t_new[t_new_inds] = torch.randint(0, max_classes_task, size=(t_new[t_new_inds].shape[0],))


        if cuda:
            feat_new, t_new = feat_new.to(device), t_new.to(device)
            
        # print('feat_new', feat_new.device, task)
        # print(next(model_main.model.parameters()).device)

        #1)  Feature Extraction 
        _,feat_new = model_main(feat_new, task, base_apply=True)
        feat_new=feat_new[feature_name]
        feat_new, t_new = Variable(feat_new), Variable(t_new)
        
                
        #2) Forward Pass new data
        optimizer.zero_grad()
        if OOD_class.name == 'odin':
            loss, output_new = OOD_class.get_loss(feat_new, t_new, task, base_apply=False)
        else:
            # print('feat_new', feat_new.device, task)
            # print(next(model_main.model.parameters()).device)
            output_new,_ = model_main(feat_new, task, base_apply=False)
            loss = F.cross_entropy(output_new, t_new)


        loss.backward()
        optimizer.step()
        if OOD_class.name == 'odin':
            OOD_class.h_optimizer.step()

        pred_new = output_new.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct_new = pred_new.eq(t_new.data.view_as(pred_new)).cpu().sum()


        if batch_idx % display_freq == 0:
            print('Train Epoch: %d [%d/%d (%.3f)]\tLoss: %.4f\tAcc_Train_new:%.3f'%(
                epoch, batch_idx, len(loader_new),
                100. * batch_idx / len(loader_new), loss.item(), float(correct_new.item()/feat_new.shape[0]))
            )
        
        if cut_epoch_short:
            if batch_idx == 10:
                break

