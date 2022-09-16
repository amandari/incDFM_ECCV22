import os
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import abc

from matplotlib import pyplot
import seaborn as sn
import pandas as pd
import matplotlib

matplotlib.use('Agg')


class singleclass_gaussian():

    def __init__(self):
        pass

    def __repr__(self):
        out = 'Single class Gaussian: {}-dimensional, {} covariance'.format(self.dim, self.covar_type)
        return out

    def fit(self, data):
        dim = data.shape[0]
        self.dim = dim
        self.mean_vec = np.zeros((1, dim), dtype=data.dtype)
        self.u_mat = np.zeros((dim, dim), dtype=data.dtype)
        self.sigma_mat = np.zeros((1, dim), dtype=data.dtype)
        self.eps = np.finfo(data.dtype).eps
        N = data.shape[1]
        self.mean_vec = np.mean(data, axis=1)
        # Interpret mean as 2D array of size 1 x n, normalize to calculate covariance
        data_centered = (data - self.mean_vec.reshape(-1, 1)) / np.sqrt(N)
        self.u_mat, self.sigma_mat, _ = np.linalg.svd(data_centered, full_matrices=False)

    def nll(self, x):
        S_local = np.copy(self.sigma_mat)
        v = np.dot(x - self.mean_vec, self.u_mat / S_local)
        m = np.sum(v * v, axis=1) + 2*np.sum(np.log(S_local))
        return m

    def score_samples(self, oi):
        nll = np.zeros((oi.shape[0], 1))
        nll = self.nll(oi)
        return -nll

