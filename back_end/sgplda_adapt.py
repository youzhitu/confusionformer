import numpy as np
from scipy.linalg import svd


epsilon = 1e-6


def coral_adapt(X_src, X_tar, regular_fac=1.):
    """ CORAL adaptation
        transform_mat = Sigma_src ^ -0.5 * Sigma_tar ^ 0.5
        Sigma_src = U * S * U^T
        Sigma_src ^ -0.5 = U * diag(1.0 / sqrt(s)) * U^T """

    X_src = X_src - X_src.mean(0)
    X_tar = X_tar - X_tar.mean(0)
    cov_src = X_src.T @ X_src / X_src.shape[0] + regular_fac * np.eye(X_src.shape[1])
    cov_tar = X_tar.T @ X_tar / X_tar.shape[0] + regular_fac * np.eye(X_tar.shape[1])
    U_src, s_src, Vh_src = svd(cov_src)
    U_tar, s_tar, Vh_tar = svd(cov_tar)
    transform_mat = U_src @ np.diag(1. / np.sqrt(s_src + epsilon)) @ U_src.T \
        @ U_tar @ np.diag(np.sqrt(s_tar)) @ U_tar.T

    return X_src @ transform_mat


def feat_dist_adapt(X_src, X_tar, regular_fac=1.):
    """ feature-Distribution adaptor """

    mu_src, mu_tar = X_src.mean(0), X_tar.mean(0)
    X_src = X_src - mu_src
    X_tar = X_tar - mu_tar
    cov_src = X_src.T @ X_src / X_src.shape[0] + regular_fac * np.eye(X_src.shape[1])
    cov_tar = X_tar.T @ X_tar / X_tar.shape[0] + regular_fac * np.eye(X_tar.shape[1])
    U_src, s_src, Vh_src = svd(cov_src)
    U_tar, s_tar, Vh_tar = svd(cov_tar)
    transform_mat = U_src @ np.diag(1. / np.sqrt(s_src + epsilon)) @ U_src.T \
        @ U_tar @ np.diag(np.sqrt(s_tar)) @ U_tar.T

    return X_src @ transform_mat
