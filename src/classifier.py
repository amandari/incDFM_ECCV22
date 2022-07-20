import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms



class Resnet(nn.Module):
    def __init__(self, total_classes, resnet_arch='resnet18', FC_layers=[7680, 4096], resnet_base=-1, multihead_type='single', 
    base_freeze=True, pretrained_weights=None):
        super(Resnet,self).__init__()
        '''
        Resnet backbone + FC classifier for classification. 
        pretrained_weights = Path 
        '''

        self.total_classes = total_classes
        self.num_Sup = 0 # cumulative number of superclasses 
        self.multihead_type = multihead_type
        self.base_freeze = base_freeze

        self.embedding_sizes = {'resnet18':512, 'resnet34':512, 'resnet50':2048, 'resnet50_contrastive':2048}
        
        if pretrained_weights is not None:
            print('Loading personal pretrained weights')
            pretrained = False
        else:
            pretrained = True


        # --------- Embedding ---------
        if 'contrastive' in resnet_arch:
            print('load contrastive backbone')
            resnet = resnet50_contrastive(pretrained=pretrained)
        else:
            resnet = getattr(models, resnet_arch)(pretrained=pretrained)

        if pretrained_weights is not None:
            resnet.load_state_dict(torch.load(pretrained_weights), strict=False)
        


        self.feat_size = self.embedding_sizes[resnet_arch]
        self.base = nn.Sequential(*list(resnet.children())[:resnet_base])

        # ----- FC classifier  
        self.base = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_1 = self.feat_size         
    

        # --------- Classifier ---------
        FC_layers = [self.feat_1] + FC_layers
        self.FC = []
        for i in range(len(FC_layers)-1):
            self.FC.append(nn.Linear(FC_layers[i], FC_layers[i+1]))
            self.FC.append(nn.ReLU())

    
        self.classifier_penultimate = FC_layers[-1]

        if len(FC_layers)>1:
            self.FC = nn.Sequential(*self.FC)
        else:
            self.FC = None
            
        self.final_fc = nn.Linear(FC_layers[-1], self.total_classes)

        # multihead_type TODO

    def forward(self, x, base_apply=True):
        
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
            x = self.FC(x)

        if self.multihead_type=='single':
            clf_outputs = self.final_fc(x)
            
        return clf_outputs

    def forward_nologit(self, x, base_apply=True):
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
            x = self.FC(x)
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