import numpy as np
from sklearn.covariance import MinCovDet

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


class mcd_gaussian():

    def __init__(self, support_fraction=None):
        self.support_fraction = support_fraction
        pass

    def __repr__(self):
        out = 'Robust (MCD) single class Gaussian: {}-dimensional, {} covariance'.format(self.dim)
        return out

    def fit(self, data_class):
        dim = data_class.shape[1]
        self.dim = dim
        if self.support_fraction is None:
            self.cov = MinCovDet(random_state=0).fit(data_class.T)
        else:
            self.cov = MinCovDet(random_state=0, support_fraction=self.support_fraction).fit(data_class.T)
        s, d = np.linalg.slogdet(self.cov.covariance_)
        self.log_cov_det = s*d

    def score_samples(self, oi, regularize=-1):
        nll = self.cov.mahalanobis(oi) + self.log_cov_det
        return -nll





