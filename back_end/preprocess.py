""" i-vectors preprocessing """

import numpy as np
from scipy.linalg import inv, cholesky, norm, svd
import multiprocessing as mp
from utils.my_utils import segment_data


epsilon = 1e-6
default_preprocess = ['center', 'lda', 'wccn', 'whiten', 'ln']


class Preprocessor:
    def __init__(self, X=None, spk_ids=None, pca_lat_dim=None, lda_lat_dim=150, n_procs=4, paras_dir='plda',
                 preprocess='center-lda-whiten-ln'):

        self.X = X
        self.spk_ids = spk_ids
        self.lda_lat_dim = lda_lat_dim
        self.pca_lat_dim = pca_lat_dim
        self.n_procs = n_procs
        self.preprocess_list = preprocess.split('-')

        self.lda_paras_file = f'{paras_dir}/lda_paras.npz'
        self.wccn_paras_file = f'{paras_dir}/wccn_paras.npz'
        self.pca_whiten_paras_file = f'{paras_dir}/pca_whiten_paras.npz'

    def train(self):
        assert all([prep in default_preprocess for prep in self.preprocess_list]), \
            "Inappropriate preprocessing not in ['center', 'lda', 'wccn', 'whiten', 'ln']."

        if any([prep in ['lda', 'wccn', 'whiten'] for prep in self.preprocess_list]):
            for i in range(len(self.preprocess_list)):
                self.X = self._transform(self.X, preprocess_list=[self.preprocess_list[i - 1]]) if i >= 1 else self.X

                if self.preprocess_list[i] == 'lda':
                    lda_train(self.X, self.spk_ids, self.lda_lat_dim, n_procs=self.n_procs, file=self.lda_paras_file)
                elif self.preprocess_list[i] == 'wccn':
                    wccn_train(self.X, self.spk_ids, n_procs=self.n_procs, file=self.wccn_paras_file)
                elif self.preprocess_list[i] == 'whiten':
                    pca_whiten_train(self.X, pca_lat_dim=self.pca_lat_dim, file=self.pca_whiten_paras_file)
                else:
                    continue

    def transform(self, data, mu=None):
        return self._transform(data, mu=mu, preprocess_list=self.preprocess_list)

    def _transform(self, data, mu=None, preprocess_list=None):
        assert all([prep in default_preprocess for prep in self.preprocess_list]), \
            "Inappropriate preprocessing not in ['center', 'lda', 'wccn', 'whiten', 'ln']."

        for prep in preprocess_list:
            if prep == 'center':
                data = center(data, mu=mu)
            elif prep == 'lda':
                data = lda_transform(data, self.lda_paras_file)
            elif prep == 'wccn':
                data = wccn_transform(data, self.wccn_paras_file)
            elif prep == 'whiten':
                data = pca_whiten_transform(data, self.pca_whiten_paras_file)
            elif prep == 'ln':
                data = length_norm(data)
            else:
                raise NotImplementedError

        return data


def center(data, mu=None):
    """ data: row vectors, [n_samples, feat_dim] """

    mu = data.mean(0) if mu is None else mu
    return data - mu


def pca_whiten_train(data, pca_lat_dim=None, file='pca_whiten_paras.npz'):
    """ data: row vectors, [n_samples, feat_dim]
        :return
        np.diag(1. / np.sqrt(s + epsilon)) @ U.T @ data_centered  for column vectors """

    mu = data.mean(0)
    data = center(data, mu=mu)
    cov = data.T @ data / data.shape[0]
    U, s, _ = svd(cov)
    if pca_lat_dim is not None:
        U, s = U[:, :pca_lat_dim], s[:pca_lat_dim]

    np.savez(file, mu=mu, U=U, s=s)
    print(f'PCA whitening parameters saved to {file}.')


def pca_whiten_transform(data, pca_whiten_paras_file):
    """ data: row vectors, [n_samples, feat_dim] """

    # print(f'Load PCA whitening parameters from {pca_whiten_paras_file}.')
    pca_whiten_paras = np.load(pca_whiten_paras_file)
    mu, U, s = pca_whiten_paras['mu'], pca_whiten_paras['U'], pca_whiten_paras['s']

    return (data - mu) @ U / np.sqrt(s + epsilon)


def length_norm(data):
    return data / norm(data, axis=1)[:, None] * data.shape[1]  # np.sqrt(data.shape[1])


def wccn_train(data, labels, n_procs=4, file='wccn_paras.npz'):
    within_cov = compute_within_class_cov(data, labels, n_procs)
    wccn_transform_mat = cholesky(inv(within_cov), lower=True)

    np.savez(file, mu=data.mean(0), U=wccn_transform_mat)
    print(f'WCCN transformation matrix saved to {file}.')


def wccn_transform(data, wccn_paras_file):
    # print(f'Load WCCN transformation matrix from {wccn_paras_file}.')
    wccn_paras = np.load(wccn_paras_file)
    mu, wccn_transform_mat = wccn_paras['mu'], wccn_paras['U']

    return (data - mu) @ wccn_transform_mat


def lda_train(data, labels, lat_dim, n_procs=4, file='lda_paras.npz'):
    mu = data.mean(0)
    between_cov = compute_between_class_cov(data, labels, n_procs)
    within_cov = compute_within_class_cov(data, labels, n_procs, between_covariance=between_cov)

    within_cov_U, within_cov_sv, _ = svd(within_cov)
    floor_sv = epsilon
    floor_idx = within_cov_sv < floor_sv
    if np.any(floor_idx):
        print('Warning: within-class covariance matrix of the training data is NOT FULL rank.')
        # within_cov_sv[floor_idx] = floor_sv

    U_whitening = within_cov_U / np.sqrt(within_cov_sv)
    between_cov_whitened = U_whitening.T @ between_cov @ U_whitening
    between_cov_U, between_cov_sv, _ = svd(between_cov_whitened)
    U = U_whitening @ between_cov_U
    s, U = between_cov_sv[:lat_dim], U[:, :lat_dim]

    # if np.any(floor_idx):
    #     print('Warning: within-class covariance matrix of the training data is NOT FULL rank.')
    #     within_cov = within_cov + np.eye(within_cov.shape[0]) * epsilon
    # s, U = eigh(a=between_cov, b=within_cov)
    # s, U = s[::-1][:lat_dim], U[:, ::-1][:, :lat_dim]

    np.savez(file, mu=mu, U=U, s=s)
    print(f'LDA model saved to {file}.')


def lda_transform(data, lda_paras_file):
    # print(f'Load LDA model from {lda_paras_file}.')
    lda_paras = np.load(lda_paras_file)
    mu, U, s = lda_paras['mu'], lda_paras['U'], lda_paras['s']

    return (data - mu) @ U


def compute_within_class_cov(data, labels, n_procs=4, between_covariance=None):
    """
    labels_unique = np.unique(labels)
    within_cov = []

    for lb in labels_unique:
        data_se = center(data[labels == lb])
        within_cov.append(data_se.T @ data_se)
    within_cov = np.sum(within_cov, 0) / data.shape[0] """

    data_centered = center(data)
    tot_cov = data_centered.T @ data_centered / data_centered.shape[0]
    between_cov = between_covariance if between_covariance is not None else \
        compute_between_class_cov(data, labels, n_procs)
    within_cov = tot_cov - between_cov

    return within_cov


def compute_between_class_cov(data, labels, n_procs=4):
    data_centered = center(data)

    if n_procs > 1:
        """ Compute between_cov using multiple processes
            First segment the whole data into n_procs segments, then call the MP function """

        # n_procs = 4  # Watch out the memory usage
        data_sorted_seg, labels_seg = segment_data(data_centered, labels, n_procs)

        pool = mp.Pool(n_procs)
        between_cov_subprocess = pool.starmap(_compute_between_class_cov,
                                              [[data_sorted_seg[i], labels_seg[i]] for i in range(n_procs)])
        pool.close()
        between_cov = np.sum(between_cov_subprocess, 0) / data_centered.shape[0]
    else:
        """ Compute between_cov using single process """
        between_cov = _compute_between_class_cov(data_centered, labels) / data_centered.shape[0]

    return between_cov


def _compute_between_class_cov(data, labels):
    """ data should be centered before computing between_cov """
    feat_dim = data.shape[1]
    between_cov = np.zeros((feat_dim, feat_dim))
    labels_unique, n_samples_per_class = np.unique(labels, return_counts=True)

    for lb, n in zip(labels_unique, n_samples_per_class):
        mean_se = data[labels == lb].mean(0)
        between_cov = between_cov + n * mean_se[:, None] * mean_se

    return between_cov
