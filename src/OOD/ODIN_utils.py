
import math

from matplotlib.pyplot import xcorr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

import sys

# ---------- main functions 
def averages_variances():
    r_mean = 125.3/255
    g_mean = 123.0/255
    b_mean = 113.9/255
    r_std = 63.0/255
    g_std = 62.1/255
    b_std = 66.7/255
    return r_mean, g_mean, b_mean, r_std, g_std, b_std


def calc_tnr(id_test_results, ood_test_results):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=.95)]

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    print(scores)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result

# -------------------- OOD 

def norm(x):
    norm = torch.norm(x, p=2, dim=1)
    x = x / (norm.expand(1, -1).t() + .0001)
    return x

#self.weights = torch.nn.Parameter(torch.randn(size = (num_classes, in_features)) * math.sqrt(2 / (in_features)))
class CosineDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CosineDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        x = norm(x)
        w = norm(self.h.weight)

        ret = (torch.matmul(x,w.T))
        return ret

class EuclideanDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(EuclideanDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        x = x.unsqueeze(2) #(batch, latent, 1)
        h = self.h.weight.T.unsqueeze(0) #(1, latent, num_classes)
        ret = -((x -h).pow(2)).mean(1)
        return ret
        
class InnerDeconf(nn.Module):
    def __init__(self, in_features, num_classes):
        super(InnerDeconf, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.h(x)


class DeconfNet(nn.Module):
    def __init__(self, underlying_model, in_features, num_classes, h, baseline):
        super(DeconfNet, self).__init__()
        
        self.num_classes = num_classes

        self.underlying_model = underlying_model
        
        self.h = h
        
        self.baseline = baseline

        if baseline:
            self.ones = nn.Parameter(torch.Tensor([1]), requires_grad = True)
        else:
            self.g = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        
        self.softmax = nn.Softmax()
    
    def forward(self, x, base_apply=True):

        output = self.underlying_model.forward_nologit_odin(x, base_apply=base_apply)

        # print('output.shape', output.shape)
        numerators = self.h(output)
        
        if self.baseline:
            denominators = torch.unsqueeze(self.ones.expand(len(numerators)), 1)
        else:
            denominators = self.g(output)
        
        # Now, broadcast the denominators per image across the numerators by division
        quotients = numerators / denominators

        # logits, numerators, and denominators
        return quotients, numerators, denominators


    def forward_test(self, x, device, base_apply=True):
        '''
        TODO - fix the add_grad thing by adding the noise to the interm rep if freezing base
        '''
                
        if self.underlying_model.base_freeze:
            x = x.to(device)
            # print('x', x.shape)
            if len(x.shape)>2:
                x = self.underlying_model.process_features(x)
            x_can_grad = Variable(x, requires_grad = True)
            output = self.underlying_model.forward_nologit_odin(x_can_grad, base_apply=False)

        else:
            x_can_grad = Variable(x.to(device), requires_grad = True)
            output = self.underlying_model.forward_nologit_odin(x_can_grad, base_apply=base_apply)

        # print('output.shape', output.shape)
        numerators = self.h(output)
        
        if self.baseline:
            denominators = torch.unsqueeze(self.ones.expand(len(numerators)), 1)
        else:
            denominators = self.g(output)
        
        # Now, broadcast the denominators per image across the numerators by division
        quotients = numerators / denominators

        # logits, numerators, and denominators
        return quotients, numerators, denominators, x_can_grad



# -----------------------------------------  loaders 

class Normalizer:
    def __init__(self, r_mean, g_mean, b_mean, r_std, g_std, b_std):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        self.r_std = r_std
        self.g_std = g_std
        self.b_std = b_std

    def __call__(self, batch):
        batch[:, 0] = (batch[:, 0] - self.r_mean) / self.r_std
        batch[:, 1] = (batch[:, 1] - self.g_mean) / self.g_std
        batch[:, 2] = (batch[:, 2] - self.b_mean) / self.b_std

class GaussianIterator:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_idx = 0
        self.transformers = transformers

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        self.batch_idx += 1
        batch = torch.randn(self.batch_size, 3, 32, 32) + 0.5
        batch = torch.clamp(batch, 0, 1)

        # Run in-place transformers on the batch, such as normalization
        for t in self.transformers:
            t(batch)

        return batch, None

class UniformIterator:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_idx = 0
        self.transformers = transformers

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        self.batch_idx += 1
        batch = torch.rand(self.batch_size, 3, 32, 32)

        # Run in-place transformers on the batch, such as normalization
        for t in self.transformers:
            t(batch)

        return batch, None

class GeneratingLoader:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.transformers = transformers
    
    def __len__(self):
        return self.num_batches

class GaussianLoader(GeneratingLoader):
    def __init__(self, batch_size, num_batches, transformers):
        super(GaussianLoader, self).__init__(batch_size, num_batches, transformers)
    
    def __iter__(self):
        return GaussianIterator(self.batch_size, self.num_batches, self.transformers)

class UniformLoader(GeneratingLoader):
    def __init__(self, batch_size, num_batches, transformers):
        super(UniformLoader, self).__init__(batch_size, num_batches, transformers)
    
    def __iter__(self):
        return UniformIterator(self.batch_size, self.num_batches, self.transformers)


# ------------------ generating loaders 


class Normalizer:
    def __init__(self, r_mean, g_mean, b_mean, r_std, g_std, b_std):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        self.r_std = r_std
        self.g_std = g_std
        self.b_std = b_std

    def __call__(self, batch):
        batch[:, 0] = (batch[:, 0] - self.r_mean) / self.r_std
        batch[:, 1] = (batch[:, 1] - self.g_mean) / self.g_std
        batch[:, 2] = (batch[:, 2] - self.b_mean) / self.b_std

class GaussianIterator:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_idx = 0
        self.transformers = transformers

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        self.batch_idx += 1
        batch = torch.randn(self.batch_size, 3, 32, 32) + 0.5
        batch = torch.clamp(batch, 0, 1)

        # Run in-place transformers on the batch, such as normalization
        for t in self.transformers:
            t(batch)

        return batch, None

class UniformIterator:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.batch_idx = 0
        self.transformers = transformers

    def __next__(self):
        if self.batch_idx >= self.num_batches:
            raise StopIteration

        self.batch_idx += 1
        batch = torch.rand(self.batch_size, 3, 32, 32)

        # Run in-place transformers on the batch, such as normalization
        for t in self.transformers:
            t(batch)

        return batch, None

class GeneratingLoader:
    def __init__(self, batch_size, num_batches, transformers):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.transformers = transformers
    
    def __len__(self):
        return self.num_batches

class GaussianLoader(GeneratingLoader):
    def __init__(self, batch_size, num_batches, transformers):
        super(GaussianLoader, self).__init__(batch_size, num_batches, transformers)
    
    def __iter__(self):
        return GaussianIterator(self.batch_size, self.num_batches, self.transformers)

class UniformLoader(GeneratingLoader):
    def __init__(self, batch_size, num_batches, transformers):
        super(UniformLoader, self).__init__(batch_size, num_batches, transformers)
    
    def __iter__(self):
        return UniformIterator(self.batch_size, self.num_batches, self.transformers)