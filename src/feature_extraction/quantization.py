
import sys
import numpy as np
import torch
import faiss
import time
import torch
from torch.utils.data import Dataset


class DSET_latents(Dataset):
    """
    Make subset of data a dataset object
    images, labels (optional): tensor
    """
    def __init__(self, features, labels=None):
        
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
           
        if self.labels:
            feat, target = self.features[index,...], self.labels[index]
            return feat, target, index

        feat = self.features[index,...]
        return feat, index


def fit_pq(feats_base_init,  num_channels_init, num_codebooks,
           codebook_size, spatial_feat_dim=-1):
    """
    Fit the PQ model and then quantize and store the latent codes of the data used to train the PQ in a dictionary to 
    be used later as a replay buffer.
    :param feats_base_init: numpy array of base init features that will be used to train the PQ
    :param num_channels: number of channels in desired features
    :param spatial_feat_dim: spatial dimension of desired features
    :param num_codebooks: number of codebooks for PQ, .i.e. number of centroids 
    :param codebook_size: size of each codebook for PQ, .i.e. dimension of encoded quantized input 

    Output:
    pq (object): trained product quantizer object from faiss implementation 
    """

    if spatial_feat_dim>0:
        train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
    else:
        train_data_base_init = feats_base_init

    train_data_base_init = np.reshape(train_data_base_init, (-1, num_channels_init))

    # 1) Train the Product Quantizer
    print('\nTraining Product Quantizer')
    start = time.time()
    print('codebook size: ', codebook_size)
    nbits = int(np.log2(codebook_size)) # numbe of bits 
    print('encode with %d bits'%nbits)
    print('Quantize with %d number of codebooks'%num_codebooks)
    pq = faiss.ProductQuantizer(num_channels_init, num_codebooks, nbits)
    pq.train(train_data_base_init)
    print("Completed in {} secs".format(time.time() - start))
    return pq


def encode(pq, feats, num_codebooks, num_channels_init=512, spatial_feat_dim=-1):
    '''encode data for storage in buffer in quantized format'''
    start_time = time.time()

    # 0) Reshape features if needed
    if spatial_feat_dim>0:
        feats = np.transpose(feats, (0, 2, 3, 1))
        feats = np.reshape(feats, (-1, num_channels_init))
    num_samples = feats.shape[0]

    # 1) Wrap the data into a dataloader 
    dset_latent = DSET_latents(feats)
    dloader_latent = torch.utils.data.DataLoader(dset_latent, batch_size=100,
                                              shuffle=False, num_workers=2)

    # 3) Encode and store codes
    if spatial_feat_dim<0:
        latent_codes = np.empty((num_samples, num_codebooks), dtype=np.uint8)
    else:
        latent_codes = np.empty((num_samples, spatial_feat_dim, spatial_feat_dim, num_codebooks), dtype=np.uint8)
    start_ix = 0
    for i, batch in enumerate(dloader_latent):
        data_batch = batch[0].numpy()
        end_ix = start_ix + data_batch.shape[0]
        # print('data_batch', data_batch.shape)
        # print(data_batch)
        # sys.exit()
        # codes are the quantized version of each input or batch of inputs in this case
        
        codes = pq.compute_codes(data_batch) 
        if spatial_feat_dim>0:
            codes = np.reshape(codes, (-1, spatial_feat_dim, spatial_feat_dim, num_codebooks))

        latent_codes[start_ix:end_ix,...] = codes

    print("Encoding completed in {} secs".format(time.time() - start_time))
    return latent_codes



def decode(data_codes, pq, num_codebooks, num_channels=512, spatial_feat_dim=-1):
    '''use pretrained PQ to decode coded samples in buffer (batch)'''
    if spatial_feat_dim>0:
        data_codes = np.reshape(data_codes, (data_codes.shape[0] * spatial_feat_dim * spatial_feat_dim, num_codebooks))

    # print('data_codes', data_codes, data_codes.shape)
    data_batch_reconstructed = pq.decode(data_codes)
    if spatial_feat_dim>0:
        data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                (-1, spatial_feat_dim, spatial_feat_dim,
                                                num_channels))
        data_batch_reconstructed = np.transpose(data_batch_reconstructed, (0, 3, 1, 2))
    else:
        data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                (-1, num_channels))
    return data_batch_reconstructed


def encode_n_decode(data_batch, pq, num_channels=512, spatial_feat_dim=-1):
    '''Used for inference - to have test data under same resolution as train
        data_batch (ndarray): Num_samples x Features x space x space or just Num_samples x Features 
    '''
    if spatial_feat_dim>0:
        data_batch = np.transpose(data_batch, (0, 2, 3, 1))
        data_batch = np.reshape(data_batch, (-1, num_channels))

    # print('data_batch', data_batch.shape)
    codes = pq.compute_codes(data_batch)
    data_batch_reconstructed = pq.decode(codes)
    if spatial_feat_dim>0:
        data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                            (-1,spatial_feat_dim, spatial_feat_dim, num_channels))
        data_batch_reconstructed = np.transpose(data_batch_reconstructed, (0, 3, 1, 2))
    
    return data_batch_reconstructed


def get_centroids_pq(pq):
    centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
    return centroids