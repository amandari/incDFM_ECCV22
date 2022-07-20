import sys
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim

from novelty_dfm_CL.dset8_specific_scripts.train_main_8dset import train_main_epoch
from novelty_dfm_CL.dset8_specific_scripts.test_main_8dset import test_main



def flatten(t):
    return [item for sublist in t for item in sublist]




def divide_batches_old_new(task, batchsize, tasks_labels, coreset, max_batch_ratio=0.85, type_batch_divide='proportional'):
    
    
    if type_batch_divide=='proportional':
        if task>0:
            old_labels=flatten(tasks_labels[:task])
        else:
            old_labels =[]
            
        max_batch_old = int(np.ceil(batchsize*max_batch_ratio))
            
        new_labels=tasks_labels[task]
        
        nb_labels = len(old_labels+new_labels)
        if task>0 and (coreset is not None):
            batch_old = min(int(np.ceil((float(len(old_labels))/nb_labels)*batchsize)),max_batch_old)
            batch_new = batchsize - batch_old
        else:
            batch_old=0
            batch_new = batchsize
    else:
        # half half 
        if task>0 and (coreset is not None):
            batch_new =  int(batchsize*0.5)
            batch_old = batchsize - batch_new
        else:
            batch_new = batchsize
            batch_old = 0            
        
            
    return batch_old, batch_new




def clf_input_size(in_name):
    '''
    TODO modify to inlcude option of multiple prior layers concatenated
    '''
    embedding_sizes = {'resnet18':512, 'resnet34':512, 'resnet50':2048, 'resnet50_contrastive':2048, 'wide_resnet50_2_contrastive': 2048}
    return embedding_sizes[in_name]



def train(t, args, novelty_detector, network_inner, loader_new, test_loaders, train_technique=1):
    
    if (args.finetune_backbone=='first' and t==0) or args.finetune_backbone=='all':
        network_inner.model.base.train(True)
        network_inner.base_freeze = False
        for param in network_inner.model.base.parameters():
            param.requires_grad = True
    else:
        network_inner.model.base.train(False)
        network_inner.base_freeze = True
        for param in network_inner.model.base.parameters():
            param.requires_grad = False


    if args.novelty_detector_name=='odin':
        clf_parameters = []
        for name, parameter in network_inner.model.named_parameters():
            if name == 'h.h.weight' or name == 'h.h.bias':
                pass
            else:
                clf_parameters.append(parameter)
    else:
        clf_parameters = network_inner.model.parameters()

        
    if train_technique==1:
        optimizer_main = optim.Adam(filter(lambda p: p.requires_grad, clf_parameters), lr=args.lr)
        scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, 'min', patience=args.patience_lr, factor=args.schedule_decay, min_lr=0.00001)
    elif train_technique==2:
        optimizer_main = optim.SGD(filter(lambda p: p.requires_grad, clf_parameters), lr=args.lr, momentum=0.9)
        scheduler_main = optim.lr_scheduler.StepLR(optimizer_main, step_size=args.step_size_epoch_lr, gamma=args.gamma_step_lr)
    elif train_technique==3:
        optimizer_main = optim.SGD(filter(lambda p: p.requires_grad, clf_parameters), lr = args.lr, momentum = 0.9, weight_decay = 0.0001)
        scheduler_main = optim.lr_scheduler.MultiStepLR(optimizer_main, milestones = [int(args.num_epochs * 0.5), int(args.num_epochs * 0.75)], gamma = 0.1)
    elif train_technique==4:
        optimizer_main = optim.RMSprop(filter(lambda p: p.requires_grad, clf_parameters), lr = args.lr)
        scheduler_main = optim.lr_scheduler.ReduceLROnPlateau(optimizer_main, 'min', patience=args.patience_lr, factor=args.schedule_decay, min_lr=0.00001)



    # ----- relative batchsizes between old and new 
    for epoch in range(args.num_epochs):
        print('##########epoch %d###########'%epoch)
        network_inner.model.train()
        if (args.finetune_backbone=='first' and t==0) or args.finetune_backbone=='all':
            network_inner.model.base.train(True)
            network_inner.base_freeze = False
            for param in network_inner.model.base.parameters():
                param.requires_grad = True
        else:
            network_inner.model.base.train(False)
            network_inner.base_freeze = True
            for param in network_inner.model.base.parameters():
                param.requires_grad = False
                
        if loader_new is not None:
            train_main_epoch(epoch, t, args.detector_params['num_classes_fine_tasks'][t], loader_new, network_inner, optimizer_main,  device=args.device, OOD_class=novelty_detector,\
                        feature_name=args.clf_layers_input, cut_epoch_short=args.cut_epoch_short,\
                        target_ind=1, cuda=True, display_freq=args.log_interval)

        val_loss = test_main(epoch, t, test_loaders, network_inner, novelty_detector, args.dir_save, \
            feature_name=args.clf_layers_input, target_ind=args.dset_prep['scenario_classif'], \
                cuda=True, device=args.device)

        # Update schedulers 
        if train_technique==1 or train_technique==4:
            scheduler_main.step(val_loss)
        else:
            scheduler_main.step()

        if novelty_detector.name == 'odin':
            if train_technique==1 or train_technique==4:
                novelty_detector.h_scheduler.step(val_loss)
            else:
                novelty_detector.h_scheduler.step()






# -----------------------------------------------------------------------------------------

class BinaryHashLinear(nn.Module):
    def __init__(self, n_in, n_out, period, key_pick='hash', learn_key=True):
        super(BinaryHashLinear, self).__init__()
        self.key_pick = key_pick
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, period)).astype(np.float32)
        o = torch.from_numpy(rand_01*2 - 1)

        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

    def forward(self, x, task):
        o = self.o[:, task]
        m = x*o
        r = torch.mm(m, self.w)
        return r
    

class BasicBlockPSP(nn.Module):
    def __init__(self, in_planes, out_planes, num_tasks):
        super(BasicBlockPSP, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.psp = BinaryHashLinear(in_planes, out_planes, num_tasks, key_pick='hash', learn_key=True)
    def forward(self, x_dict):
        x = x_dict['x']
        task_num = x_dict['task']
        x = self.relu(self.psp(x, task_num))
        return {'x':x, 'task':task_num}



class MyInnerClf_PSP(nn.Module):
    def __init__(self, FC_layers, num_tasks):
        super(MyInnerClf_PSP, self).__init__()
        
        self.FC_list = []
        for i in range(len(FC_layers)-1):
            self.FC_list.append(BasicBlockPSP(FC_layers[i], FC_layers[i+1], num_tasks))
        self.main = nn.Sequential(*self.FC_list)
        
    def forward(self, x):
        return self.main(x)





class Resnet_PSP(nn.Module):
    def __init__(self, num_classes, num_tasks, resnet_arch='resnet18', FC_layers=[7680, 4096],
    base_freeze=True, pretrained_weights=None, apply_clf_ODIN=True):
        super(Resnet_PSP,self).__init__()
        '''
        Resnet backbone + FC classifier for classification. 
        FC layers are PSP-layers
        Multi-head output
        '''

        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.base_freeze = base_freeze

        
        if pretrained_weights is not None:
            print('Loading personal pretrained weights')
            pretrained = False
        else:
            pretrained = True

        self.apply_clf_ODIN = apply_clf_ODIN


        # --------- Embedding ---------
        if 'contrastive' in resnet_arch:
            print('load contrastive backbone')
            if 'wide' in resnet_arch:
                resnet = resnet50_w2_contrastive(pretrained=pretrained)
            else:
                resnet = resnet50_contrastive(pretrained=pretrained)
        else:
            resnet = getattr(models, resnet_arch)(pretrained=pretrained)

        if pretrained_weights is not None:
            resnet.load_state_dict(torch.load(pretrained_weights), strict=False)
        


        self.feat_size = clf_input_size(resnet_arch)

        # ----- FC classifier  
        self.base = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_1 = self.feat_size
        

        # --------- Classifier ---------
        FC_layers = [self.feat_1] + FC_layers
        if self.apply_clf_ODIN:
            self.output_size = FC_layers[-1]
        else:
            self.output_size = FC_layers[0]
        self.classifier_penultimate = FC_layers[-1]

        
        if len(FC_layers)>1:
            self.FC = MyInnerClf_PSP(FC_layers, self.num_tasks)
        else:
            self.FC = None
        
            
        for i in range(0,self.num_tasks):
            setattr(self, "fc%d" % i, nn.Linear(FC_layers[-1], self.num_classes[i]))


    def forward(self, x, task_num, base_apply=True):
        
        # --- extract embedding
        if base_apply:
            if self.base_freeze:
                with torch.no_grad():
                    x = self.base(x)
            else:
                x = self.base(x)
                
        if len(x.shape)==4:
            x = F.avg_pool2d(x, x.size()[3])
        x = torch.flatten(x, 1)

        # --- Classification 
        if self.FC is not None:
            x = self.FC({'x':x, 'task':task_num})
            x = x['x']
                
        clf_outputs = getattr(self, "fc%d" % task_num)(x)
            
        return clf_outputs

    def forward_nologit(self, x, task_num, base_apply=True):
        # --- extract embedding
        if base_apply:
            if self.base_freeze:
                with torch.no_grad():
                    x = self.base(x)
            else:
                x = self.base(x)
                
        if len(x.shape)==4:
            x = F.avg_pool2d(x, x.size()[3])
        x = torch.flatten(x, 1)

        # --- Classification 
        if self.FC is not None:
            x = self.FC({'x':x, 'task':task_num})
            x = x['x']
            
        return x


    def forward_nologit_odin(self, x, task_num, base_apply=True):
        '''Choose clf layer to tap into'''
        # --- extract embedding
        if base_apply:
            if self.base_freeze:
                with torch.no_grad():
                    x = self.base(x)
            else:
                x = self.base(x)
                
            if len(x.shape)==4:
                x = F.avg_pool2d(x, x.size()[3])
            x = torch.flatten(x, 1)

        # --- Classification
        if self.apply_clf_ODIN: 
            # --- Classification 
            if self.FC is not None:
                x = self.FC({'x':x, 'task':task_num})
                x = x['x']
        return x

    
    def process_features(self, x):
        '''Embedding extraction'''
        with torch.no_grad():
            x = self.base(x)
            if len(x.shape)==4:
                x = F.avg_pool2d(x, x.size()[3])
            x = torch.flatten(x, 1)
        return x
    



def resnet50_contrastive(pretrained=True, **kwargs):
    """
    ResNet-50 pre-trained with SwAV.
    Note that `fc.weight` and `fc.bias` are randomly initialized.
    Achieves 75.3% top-1 accuracy on ImageNet when `fc` is trained.
    """
    model = models.resnet.resnet50(pretrained=False, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar",
            map_location="cpu",
        )
        # removes "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # load weights
        model.load_state_dict(state_dict, strict=False)

    return model



def resnet50_w2_contrastive(pretrained=True, **kwargs):
    """
    Wide ResNet-50 pre-trained with SwAV.
    Note that `fc.weight` and `fc.bias` are randomly initialized.
    """
    print('LOAD W2-resnet50 SWAV')
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50w2')
    # models.resnet.wide_resnet50_2(pretrained=False, **kwargs)
    # if pretrained:
    #     state_dict = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deepcluster/swav_RN50w2_400ep_pretrain.pth.tar",
    #         map_location="cpu",
    #     )
    #     # removes "module."
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     # load weights
    #     model.load_state_dict(state_dict, strict=False)

    return model

