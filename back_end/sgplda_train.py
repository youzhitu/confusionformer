""" Simplified Gaussian PLDA model training """

import numpy as np
from scipy.linalg import inv, eigvalsh
from utils.my_utils import segment_data
from back_end.preprocess import center, pca_whiten_train, pca_whiten_transform
import multiprocessing as mp


class SGPLDATrainer:
    def __init__(self, X, spk_ids, lat_dim=150, n_iters=10, init='randn'):
        self.X = X
        self.spk_ids = spk_ids
        self.lat_dim = lat_dim
        self.n_iters = n_iters
        self.init = init

        self._compute_global_vars()

        self.V, self.Sigma = np.zeros((self.feat_dim, self.lat_dim)), np.zeros((self.feat_dim, self.feat_dim))
        self.precision = np.zeros_like(self.Sigma)
        self.E_z = np.zeros((self.n_spks, self.lat_dim))
        self.E_z_z_t = np.zeros((self.n_spks, self.lat_dim, self.lat_dim))
        self.V_t_Sigma_inv_V = np.zeros((self.lat_dim, self.lat_dim))

        self._init_params()

    def _compute_global_vars(self):
        self.n_samples, self.feat_dim = self.X.shape
        self.mu = self.X.mean(0)
        self.X_centered = center(self.X)

        self.spk_ids_unique, self.n_utts_per_spk = np.unique(self.spk_ids, return_counts=True)
        self.n_spks = self.spk_ids_unique.shape[0]
        self.n_utts_per_spk_unique = np.unique(self.n_utts_per_spk)

        self.__compute_sum_x_per_spk()
        self.XX_t_centered = self.X_centered.T @ self.X_centered

    def __compute_sum_x_per_spk(self, n_processors=4):
        assert np.count_nonzero(self.n_utts_per_spk == 1) == 0, f'Some speakers have only one utterance.'

        if n_processors > 1:
            X_seg, spk_ids_seg = segment_data(self.X_centered, self.spk_ids, n_processors)
            pool = mp.Pool(n_processors)
            sum_x_per_spk = pool.starmap(compute_sum_x_per_spk,
                                         [[X_seg[i], spk_ids_seg[i]] for i in range(n_processors)])
            pool.close()
            self.sum_x_per_spk = np.concatenate(sum_x_per_spk)  # [n_spks, feat_dim]
        else:
            self.sum_x_per_spk = compute_sum_x_per_spk(self.X_centered, self.spk_ids)

    def _init_params(self):
        self.V = self.__init_V_randn() if self.init == 'randn' else self.__init_V_pca()
        self.Sigma = self.__init_Sigma()

    def __init_V_randn(self):
        return np.random.randn(self.feat_dim, self.lat_dim)

    def __init_V_pca(self):
        pca_paras_tmp_file = 'plda/V_pca_paras.npy'
        cluster_means = self.sum_x_per_spk / self.n_utts_per_spk[:, None]
        pca_whiten_train(cluster_means.T, pca_paras_tmp_file)
        pca_cluster = pca_whiten_transform(cluster_means.T, pca_paras_tmp_file)
        return pca_cluster[:, :self.lat_dim]

    def __init_Sigma(self):
        temp = np.random.randn(self.feat_dim * 4, self.feat_dim)
        Sigma = 0.1 * temp.T @ temp / temp.shape[0]
        # Sigma = 0.1 * np.eye(self.feat_dim)
        return Sigma

    def train(self, sgplda_paras_file='sgplda_paras_file.npz'):
        print(f'Training the SGPLDA model on {self.n_spks} speakers with {self.n_samples} ivectors.')
        print(f'No. of speakers with different utterances: {self.n_utts_per_spk_unique.shape[0]}\n')

        llh_log, elbo_log = [], []

        self.e_step()
        # llh = self.compute_llh()
        elbo = self.compute_elbo()

        # llh_log.append(llh)
        elbo_log.append(elbo)
        # print_indicator(0, llh_log, indicator='llh')
        print_indicator(0, elbo_log, indicator='elbo')

        for n in range(self.n_iters):
            print(f'EM iteration {n + 1}/{self.n_iters}:')
            self.m_step()
            # t_s = time()
            self.e_step()
            # print(f'Time of e_step: {time() - t_s}s.')
            # llh = self.compute_llh()
            elbo = self.compute_elbo()

            # llh_log.append(llh)
            elbo_log.append(elbo)
            # print_indicator(n + 1, llh_log, indicator='llh')
            print_indicator(n + 1, elbo_log, indicator='elbo')

        self.save_sgplda_paras(sgplda_paras_file)

    def e_step(self):
        self.precision = inv(self.Sigma)
        V_t_Sigma_inv = self.V.T @ self.precision
        self.V_t_Sigma_inv_V = V_t_Sigma_inv @ self.V

        posterior_cov = []  # Pre-compute posterior_cov to avoid too many matrix inversion operations
        for n_utts in self.n_utts_per_spk_unique:
            posterior_cov.append(inv(np.eye(self.lat_dim) + n_utts * self.V_t_Sigma_inv_V))
        posterior_cov = np.asarray(posterior_cov)

        E_z, E_z_z_t = [], []
        for i, n_utts in enumerate(self.n_utts_per_spk):
            L_inv = posterior_cov[self.n_utts_per_spk_unique == n_utts].squeeze(0)
            # L_inv = inv(np.eye(self.lat_dim) + utt * self.V_t_Sigma_inv_V)
            E_z_tmp = L_inv @ V_t_Sigma_inv @ self.sum_x_per_spk[i]
            E_z.append(E_z_tmp)
            E_z_z_t.append(L_inv + E_z_tmp[:, None] * E_z_tmp)
        self.E_z, self.E_z_z_t = np.asarray(E_z), np.asarray(E_z_z_t)

    def m_step(self):
        self.V = self.V_numerator @ inv(self.V_denominator)
        self.Sigma = (self.XX_t_centered - self.V @ self.V_numerator.T) / self.n_samples

    def compute_llh(self):
        """ Uncompleted currently """
        # Lambda = self.V @ self.V.T + self.Sigma
        # log_det_Lambda = log_det(Lambda)
        # Lambda_inv = inv(Lambda)
        # llh = - self.n_samples * (self.feat_dim * np.log(2 * np.pi) + log_det_Lambda) / 2 \
        #       - np.trace(Lambda_inv @ self.XX_t_centered) / 2
        # return llh

    def compute_elbo(self):
        """ ELBO = sum_i{E_q[log(p(x_i, z_i) / q(z_i))]}
                 = sum_i{sum_j{E_q[log(p(x_ij|z_i))] - KL(q(z_i)||p(z_i))}
                 = sum_ij{E_q[log(p(x_ij|z_i))]} - (sum_i{E_q[log(q(z_i))]} - sum_i{E_q[log(p(z_i))]})
            p(x_ij|z_i) = N(Vz_i, Sigma), p(z_i) = N(0, I), q(z_i) = N(E_z_i, inv(L_i)) """
        return self._reconstrction_llh() - self._kl_q_p()

    def _reconstrction_llh(self):
        """ sum_ij{E_q[log(p(x_ij|z_i))]} = const - 0.5 * tr{sum_ij{x_ij * x_ij^T * inv(Sigma)}}
                                            + tr{sum_ij{x_ij * <z_i>^T * V^T * inv(Sigma)}}
                                            - 0.5 * tr{V^T * inv(Sigma) * V * sum_ij{<z_i * z_i^T>}}
            const = - 0.5 * N_samples * (feat_dim * log(2 * pi) + log(det(Sigma))) """

        _, log_det_Sigma = np.linalg.slogdet(self.Sigma)  # Faster than log_det (using Cholesky decomposition)
        self.V_numerator = self.sum_x_per_spk.T @ self.E_z  # [feat_dim, lat_dim]
        self.V_denominator = np.einsum('i, ijk->jk', self.n_utts_per_spk, self.E_z_z_t)

        sum_i_V_E_z_x_per_spk_t = self.V @ self.V_numerator.T
        neg_CE_tmp = self.XX_t_centered - 2 * sum_i_V_E_z_x_per_spk_t + self.V @ self.V_denominator @ self.V.T
        neg_CE = - self.n_samples * (self.feat_dim * np.log(2 * np.pi) + log_det_Sigma) / 2 \
                 - trace_dot(neg_CE_tmp, self.precision) / 2

        return neg_CE

    def _kl_q_p(self):
        """ sum_i{KL(q(z_i)||p(z_i))} = sum_i{E_q[log(q(z_i))]} - sum_i{E_q[log(p(z_i))]}

            sum_i{E_q[log(q(z_i))]} = - 0.5 * n_spks * lat_dim * np.log(2 * np.pi)
                                      - 0.5 * sum_i{log_det(posterior_cov_i)}
                                      - 0.5 * n_spks * lat_dim
            sum_i{E_q[log(p(z_i))]} = - 0.5 * n_spks * lat_dim * np.log(2 * np.pi)
                                      - 0.5 * tr{sum_i{<z_i * z_i^T>}} """

        posterior_precision_eigvals = eigvalsh(self.V_t_Sigma_inv_V) * self.n_utts_per_spk[:, None] + 1
        sum_i_log_det_posterior_cov = - np.log(posterior_precision_eigvals).sum()

        return - (sum_i_log_det_posterior_cov + self.n_spks * self.lat_dim - np.einsum('ijj->', self.E_z_z_t)) / 2

    def _neg_cross_entropy_p_z(self):
        return - self.n_spks * self.lat_dim * np.log(2 * np.pi) / 2 - np.einsum('ijj->', self.E_z_z_t) / 2

    def _neg_entropy_q_z(self):
        posterior_precision_eigvals = np.outer(self.n_utts_per_spk, eigvalsh(self.V_t_Sigma_inv_V)) + 1
        sum_i_log_det_posterior_cov = - np.log(posterior_precision_eigvals).sum()
        return - self.n_spks * self.lat_dim * (np.log(2 * np.pi) + 1) / 2 - sum_i_log_det_posterior_cov / 2

    def save_sgplda_paras(self, sgplda_paras_file):
        np.savez(sgplda_paras_file, mu=self.mu, V=self.V, Sigma=self.Sigma)
        print(f'SGPLDA model saved to {sgplda_paras_file}.')


def compute_sum_x_per_spk(X, spk_ids):
    spk_ids_unique = np.unique(spk_ids)
    sum_x_per_spk = []  # sum_j(x_ij - mu), j~[1, H_i]

    for spk_id in spk_ids_unique:
        sum_x_per_spk_tmp = X[spk_ids == spk_id].sum(0)
        sum_x_per_spk.append(sum_x_per_spk_tmp)
    return np.asarray(sum_x_per_spk)


def print_indicator(n_iter, llh_log, indicator='elbo'):
    if n_iter > 0:
        increased_percent = 100 * (llh_log[-1] - llh_log[-2]) / abs(llh_log[-2])
        print(f'{indicator} after iter {n_iter}: {llh_log[-1]:.4f}, increased by {increased_percent:.2f}%.')
    else:
        print(f'{indicator} of initialization: {llh_log[-1]:.4f}')


def is_symmetric(mat):
    return np.allclose(mat, mat.T)


def trace_dot(A, B):
    """ Compute the trace of A@B """
    return np.einsum('ij, ji->', A, B)


