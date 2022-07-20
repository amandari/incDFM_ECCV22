import yaml
import os
import numpy as np
import re
from torch.utils.data import Dataset
import random
import PIL
import torch
import shutil
from datetime import datetime
import json
from argparse import Namespace



def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def results_saving_setup(args):
    
    with open(args.standard_config_file) as fid:
        args_fixed = Namespace(**yaml.load(fid, Loader=yaml.SafeLoader))

    args = Namespace(**vars(args), **vars(args_fixed))
    
    # Create save directories 
    makedirectory(args.dir_results)
    if args.test_num>=0:
        save_name = 'Test_%d'%(args.test_num)
    else: 
        save_name = datetime.now().strftime(r'%d%m%y_%H%M%S')
    dir_save = '%s/%s/%s/'%(args.dir_results, args.dset_name, save_name)
    # check if exists, if yes, overwrite. 
    if os.path.exists(dir_save) and os.path.isdir(dir_save):
        shutil.rmtree(dir_save)
    makedirectory(dir_save)
    
    # Name of 
    file_acc_avg = '%s/%s.txt'%(dir_save, 'acc_avg')
    file_acc_pertask = '%s/%s.txt'%(dir_save, 'acc_pertask')

    args.dir_save = dir_save
    args.file_acc_avg = file_acc_avg
    args.file_acc_pertask = file_acc_pertask
    
    return args


def save_ood_config_simple(args):

    # TODO add config for each novelty_detector
    map_ood_config = {'incdfm':args.incdfm_config, 'dfm':args.dfm_config, \
        'mahal':args.mahal_config, 'softmax':args.softmax_config, 'odin':args.odin_config}
    args.ood_config = map_ood_config[args.novelty_detector_name]
    
    with open(args.ood_config) as fid:
        args_ood = Namespace(**yaml.load(fid, Loader=yaml.SafeLoader))

    args = Namespace(**vars(args), **vars(args_ood))

    # save config of experiment
    with open('%sconfig_model.json'%(args.dir_save), 'w') as fp:
        json.dump(vars(args), fp, sort_keys=True, indent=1)
        
    return args
        
    
    

















def config_saving(args):
    makedirectory(args.dir_results)
    if args.test_num>=0:
        save_name = 'Test_%d'%(args.test_num)
    else: 
        save_name = datetime.now().strftime(r'%d%m%y_%H%M%S')
    dir_save = '%s/%s/%s/'%(args.dir_results, args.dset_name, save_name)
    # check if exists, if yes, overwrite. 
    if os.path.exists(dir_save) and os.path.isdir(dir_save):
        shutil.rmtree(dir_save)
    makedirectory(dir_save)
    # save config of experiment
    dict_args = vars(args)
    with open('%sconfig_model.json'%(dir_save), 'w') as fp:
        json.dump(dict_args, fp, sort_keys=True, indent=1)
    file_acc_avg = '%s/%s.txt'%(dir_save, 'acc_avg')
    file_acc_pertask = '%s/%s.txt'%(dir_save, 'acc_pertask')

    args.dir_save = dir_save
    args.file_acc_avg = file_acc_avg
    args.file_acc_pertask = file_acc_pertask
    
    return args











def list_to_2D(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


def memory_equivalence(size_MB, latent_dim, quantizer_dict=None):
    '''
    size_MB (float): size in Megabytes for coreset
    '''
    size_MB = size_MB*1000000 # convert to bytes 
    if quantizer_dict:
        cost_centroids = 4*quantizer_dict['num_codebooks']*(latent_dim/quantizer_dict['codebook_size'])
        num_samples = int(np.round((size_MB-cost_centroids)/latent_dim))
    else:
        num_samples = int(np.round(size_MB/(latent_dim*4))) #float32 = 4bytes 

    return num_samples




def image2arr(pathimg):
    raw = PIL.Image.open(pathimg).convert("RGB")
    raw = np.array(raw) 
    raw = np.rollaxis(raw, 2, 0)
    return raw



def makedirectory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def countlines(txtfile):
    Content = txtfile.read()
    CoList = Content.split("\n")
    Counter=0
    for i in CoList:
        if i:
            Counter += 1
    return Counter


def divide_integer_K(N,K, shuff=True):
    '''Divide an integer into equal parts exactly'''
    array=np.zeros(K,)
    for i in range(K):
        array[i] = int(N / K)    # integer division

    # divide up the remainder
    for i in range(N%K):
        array[i] += 1
        
    array = array.astype(int)

    if shuff:
        np.random.shuffle(array)
        
    return array


# Order list by integer in string of item 
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



class DSET_wrapper(Dataset):
    """
    Make subset of data a dataset object
    images, labels: tensor
    """
    def __init__(self, features, labels, transform=None, target_transform=None):
        
        self.features = features
        self.labels = labels

        self.transform = transform 
        self.target_transform = target_transform

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index):
           
        feat, target = self.features[index,...], self.labels[index]

        if self.transform is not None: ##input the desired tranform 

            feat = self.transform(feat)

        if self.target_transform is not None:
            
            target = self.target_transform(target)
        
        return feat, target


def get_args_from_yaml(args, params_parser):
    with open(args.config_file) as fid:
        args_yaml = yaml.load(fid, Loader=yaml.SafeLoader)
    ad = args.__dict__
    # print(args_yaml)
    for k in ad.keys():
        dv = params_parser.get_default(k)
        if dv is not None:  # ad[k] will not be None if it has a default value
            if ad[k] == dv and k in args_yaml:
                ad[k] = args_yaml[k]
        elif ad[k] is None:
            if k in args_yaml:
                ad[k] = args_yaml[k]
    return args

# def extract_features(feature_extractor, data_loader, num_channels=512, spatial_feat_dim=-1, device=0):
#     """
#     Extract image features and put them into arrays.
#     :param feature_extractor: pre-trained model to extract features
#     :param data_loader: (not shuffled) data loader of images for which we want features (images, labels, item_ixs)
#     :param num_channels: number of channels in desired features
#     :param spatial_feat_dim: spatial dimension of desired features (-1 means no spatial dimension)
#     :return: numpy arrays of features, labels
#     """
#     # allocate space for features and labels
#     if spatial_feat_dim<0:
#         features_data = np.empty((data_loader.dataset.__len__(), num_channels), dtype=np.float32)
#     else:
#         features_data = np.empty((data_loader.dataset.__len__(), num_channels, spatial_feat_dim, spatial_feat_dim), dtype=np.float32)
#     labels_data = np.empty((data_loader.dataset.__len__(), 1), dtype=np.int)

#     # put features and labels into arrays
#     start_ix = 0
#     for batch_ix, batch in enumerate(data_loader):
#         batch_x=batch[0]
#         batch_y=batch[1]
        
#         batch_feats = feature_extractor(batch_x.to(device))
#         end_ix = start_ix + len(batch_feats)
#         features_data[start_ix:end_ix] = batch_feats.cpu().numpy().astype(np.float32)
#         labels_data[start_ix:end_ix] = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()

#     return features_data, labels_data

