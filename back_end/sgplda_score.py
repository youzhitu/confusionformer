""" Simplified Gaussian PLDA model scoring """

import numpy as np
import pandas as pd
from scipy.linalg import inv, cholesky
from utils.my_utils import get_filtered_idx
from time import perf_counter
from pathlib import Path


class SGPLDAScorer:
    def __init__(self, X_enroll, X_test, enroll_ids, test_ids, trials_file='trials', sgplda_paras_file='sgplda.npz',
                 scoring_type='multi_session'):
        """ scoring_type: 'multi_session', 'ivec_averaging' or 'score_averaging' """

        self.X_enroll = X_enroll
        self.X_test = X_test
        self.enroll_ids = enroll_ids
        self.test_ids = test_ids
        self.trials_file = trials_file
        self.sgplda_paras_file = sgplda_paras_file
        self.scoring_type = scoring_type

        self._init_scorer()
        self.llh_p = None  # np.array([0])

    def _init_scorer(self):
        self._load_trials()
        self._load_trials_idx()
        self._load_sgplda_paras()
        self._compute_global_vars()

    def _load_trials(self):
        # print('Loading trials ...')
        self.trials_enroll_ids, self.trials_test_ids = load_trials(self.trials_file)

    def _load_trials_idx(self):
        # print('Indexing trials ...')
        self.X_enroll_idx, self.n_utts_per_spk, self.X_test_idx = \
            index_trials(self.trials_file, self.enroll_ids, self.test_ids, self.trials_enroll_ids, self.trials_test_ids)

    def _load_sgplda_paras(self):
        sgplda_paras = np.load(self.sgplda_paras_file)
        self.mu, self.V, self.Sigma = sgplda_paras['mu'], sgplda_paras['V'], sgplda_paras['Sigma']

    def _compute_global_vars(self):
        self.feat_dim, self.lat_dim = self.V.shape
        self.precision = inv(self.Sigma)
        self.V_t_Sigma_inv_V = self.V.T @ self.precision @ self.V
        self.marginal_precision = inv(self.V @ self.V.T + self.Sigma)

    def score(self, scores_file=None, X_enroll_ex=None, X_test_ex=None, is_snorm=False, select_ratio=0.1, n_top=500):
        """ PLDA scoring main function
        :param scores_file: str
        :param X_enroll_ex: external enrollment i/x-vectors (relative to those for trials scoring), e.g. cohort
        :param X_test_ex: external test i/x-vectors (relative to self.X_test for trials scoring), e.g. cohort
        :param is_snorm: bool
        :param select_ratio: float
        :param n_top: int

        If X_enroll_ex and X_test_ex are not provided, which is the default setting, scores will be computed
        according to the trials list. Otherwise snorm scoring will be performed if is_snorm is True in which
        snorm scores ("scores_ec" and "scores_ct") are first computed and raw scores are then normalized;
        if is_snorm is False, scoring will be performed between the X_enroll_ex-X_test_ex pairs.

        Snorm consists of two stages: Znorm and Tnorm. Note that if multi-session scoring is selected, the "is_tnorm"
        flag should be enabled to make sure the "enroll_ivecs" in the scoring function are correctly picked up. """

        if X_enroll_ex is not None and X_test_ex is not None:
            if is_snorm:
                t_s = perf_counter()
                scores_ec = self.plda_scoring(self.X_enroll, X_enroll_ex, is_trials_scoring=False, is_tnorm=False)
                print(f'Time of Z-norm scoring: {perf_counter() - t_s} s.')

                t_s = perf_counter()
                scores_ct = self.plda_scoring(X_test_ex, self.X_test, is_trials_scoring=False, is_tnorm=True)
                print(f'Time of T-norm scoring: {perf_counter() - t_s} s.')

                t_s = perf_counter()
                scores = self.plda_scoring(self.X_enroll, self.X_test, is_trials_scoring=True, is_tnorm=False)
                print(f'Time of scoring: {perf_counter() - t_s} s.')

                scores = self.snorm(scores_ec, scores_ct, scores, select_ratio=select_ratio, n_top=n_top)
            else:
                scores = self.plda_scoring(X_enroll_ex, X_test_ex, is_trials_scoring=False, is_tnorm=True)
        else:
            scores = self.plda_scoring(self.X_enroll, self.X_test, is_trials_scoring=True, is_tnorm=False)

        if scores_file is not None:
            self.save_scores(scores_file, scores)

        return scores

    def plda_scoring(self, X_enroll, X_test, is_trials_scoring=True, is_tnorm=False):
        """ Given a prob ivector (test ivector) x^p, and a gallery of enrollment ivectors x^g_1, x^g_2, ..., x^g_n,
        the log-likelihood ratio between the same-speaker hypothesis and the different-speaker hypothesis is computed
        as the score of this enroll-test trial.
        llr = log(p(x^p, x^g_1, x^g_2, ..., x^g_n) / (p(x^p) * p(x^g_1, x^g_2, ..., x^g_n)))
            = log(p(x^p|x^g_1, x^g_2, ..., x^g_n) / p(x^p))
            = - log(det(Sigma_p_cond_g)) / 2 - (x^p - mu_p_cond_g)^T * inv(Sigma_p_cond_g) * (x^p - mu_p_cond_g) / 2
              + log(det(V * V^T + Sigma)) / 2 + x^p^T * inv(V * V^T + Sigma) * x^p / 2

        p(x^p|x^g_1, x^g_2, ..., x^g_n) = N(mu_p_cond_g, Sigma_p_cond_g)
            mu_p_cond_g = V * inv(I + n * V^T * inv(Sigma) * V) * V^T * inv(Sigma) * [x^g_1 + x^g_2 + ... + x^g_n]
            Sigma_p_cond_g = V * inv(I + n * V^T * inv(Sigma) * V) * V^T + Sigma
        p(x^p) = N(0, V * V^T + Sigma) """

        if self.scoring_type == 'multi_session':
            if not is_tnorm:
                n_utts_per_spk_unique = np.unique(self.n_utts_per_spk)
                print(f'Length of n_utts_per_spk: {self.n_utts_per_spk.shape[0]}.')
                print(f'Length of n_utts_per_spk_unique: {n_utts_per_spk_unique.shape[0]}.')
            else:
                n_utts_per_spk_unique = np.array([1])  # For computing tnorm scores
        else:
            n_utts_per_spk_unique = np.array([1])  # For 'ivec_averaging' and 'score_averaging'

        precomp_stats_unique = self.precompute_unique_stats(n_utts_per_spk_unique)
        self.llh_p = self.precompute_llh_p(X_test)
        scores = self.scoring_core(X_enroll, X_test,
                                   n_utts_per_spk_unique, precomp_stats_unique, is_trials_scoring, is_tnorm)
        return scores

    def precompute_unique_stats(self, n_utts_per_spk_unique):
        """ Compute precision matrix of x^p|x^g_1, x^g_2, ..., x^g_n and
        posterior covariance of V*z|x^g_1, x^g_2, ..., x^g_n for each UNIQUE n_utts_per_spk to SAVE computation
        e.g.,
        <n_utts_per_spk_unique -- posterior_cov_Vz_unique -- precision_p_cond_g_unique>
        if enroll_unique: [1081_sre16, 1082_sre16, 1083_sre16, ...], n_utts_per_spk: [1, 1, 3, ...],
        then: n_utts_per_spk_unique: [1, 3], we only need to compute the statistics TWICE. """

        precision_p_cond_g_unique, posterior_cov_Vz_unique = [], []

        for n_utts in n_utts_per_spk_unique:
            posterior_cov_z = inv(np.eye(self.lat_dim) + n_utts * self.V_t_Sigma_inv_V)
            posterior_cov_Vz_unique.append(self.V @ posterior_cov_z @ self.V.T)
            precision_p_cond_g_unique.append(inv(posterior_cov_Vz_unique[-1] + self.Sigma))

        return np.asarray(precision_p_cond_g_unique), np.asarray(posterior_cov_Vz_unique)

    def precompute_llh_p(self, X_test):
        """ Precompute llh of the different-speaker hypothesis, i.e.,
            llh_p = - log(det(V * V^T + Sigma)) / 2 - x^p^T * inv(V * V^T + Sigma) * x^p / 2
            p(x^p) = N(0, V * V^T + Sigma) """

        const = log_det(self.marginal_precision) / 2
        sec_term_p = sec_ord_term(X_test - self.mu, self.marginal_precision)
        return const - sec_term_p / 2

    def scoring_core(self, X_enroll, X_test, n_utts_per_spk_unique, precomp_stats_unique, is_trials_scoring=True,
                     is_tnorm=False):
        """ Compute scores for every enrolled speaker
            :param
            n_utts_per_spk_unique: No. of utterances for each enrolled speaker in the enrollment data, used in the
                trials scoring and the Z-norm scoring (enrollment v.s. cohort scoring for S-norm)
            is_trials_scoring: flag indicating whether the current scoring is trials scoring or S-norm scoring,
                defalt: trials scoring
            is_tnorm: flag indicating whether the current S-norm scoring is Z-norm scoring or T-norm scoring, note
                that it is only valid when is_trials_scoring is disabled, default: fault, i.e. Z-norm scoring
            """

        loop_range = X_enroll.shape[0] if is_tnorm else self.n_utts_per_spk.shape[0]
        scores = []

        for i in range(loop_range):
            enroll_ivecs = X_enroll[self.X_enroll_idx[i]] if not is_tnorm else np.expand_dims(X_enroll[i], axis=0)
            test_ivecs = X_test[self.X_test_idx[i]] if is_trials_scoring else X_test
            llh_p = self.llh_p[self.X_test_idx[i]] if is_trials_scoring else self.llh_p

            mu_p_cond_g, precision_p_cond_g = \
                self._prepare_stats_p_cond_g(enroll_ivecs, precomp_stats_unique, n_utts_per_spk_unique)

            if self.scoring_type == 'score_averaging':
                mu_p_cond_g = np.repeat(mu_p_cond_g, test_ivecs.shape[0], axis=1).T
                test_ivecs = np.tile(test_ivecs, (enroll_ivecs.shape[0], 1))
                llh_p_cond_g = self._compute_llh_p_cond_g(test_ivecs, mu_p_cond_g, precision_p_cond_g)
                llh_p_cond_g = llh_p_cond_g.reshape(enroll_ivecs.shape[0], -1).mean(0)
            else:
                llh_p_cond_g = self._compute_llh_p_cond_g(test_ivecs, mu_p_cond_g, precision_p_cond_g)

            scores.append(llh_p_cond_g - llh_p)  # llr = llh_p_cond_g - llh_p

        return np.concatenate(scores)

    def _prepare_stats_p_cond_g(self, enroll_ivecs, precomp_stats_unique, n_utts_per_spk_unique):
        """ Prepare mu and precision matrix w.r.t. p(x^p|x^g_1, x^g_2, ..., x^g_n) for a SINGLE enrolled speaker
        for computing llr
        p(x^p|x^g_1, x^g_2, ..., x^g_n) = N(mu_p_cond_g, Sigma_p_cond_g)
                mu_p_cond_g = V * inv(I + n * V^T * inv(Sigma) * V) * V^T * inv(Sigma) * [x^g_1 + x^g_2 + ... + x^g_n]
                Sigma_p_cond_g = V * inv(I + n * V^T * inv(Sigma) * V) * V^T + Sigma """

        precision_p_cond_g_unique, posterior_cov_Vz_unique = precomp_stats_unique
        precision_p_cond_g = precision_p_cond_g_unique[0]

        if self.scoring_type == 'multi_session':
            n_utts_per_spk_se_idx = n_utts_per_spk_unique == enroll_ivecs.shape[0]
            posterior_cov_Vz_se = posterior_cov_Vz_unique[n_utts_per_spk_se_idx].squeeze(0)
            mu_p_cond_g = posterior_cov_Vz_se @ self.precision @ enroll_ivecs.sum(0)
            precision_p_cond_g = precision_p_cond_g_unique[n_utts_per_spk_se_idx].squeeze(0)
        elif self.scoring_type == 'ivec_averaging':
            mu_p_cond_g = posterior_cov_Vz_unique[0] @ self.precision @ enroll_ivecs.mean(0)
        else:
            mu_p_cond_g = posterior_cov_Vz_unique[0] @ self.precision @ enroll_ivecs.T  # For 'score_averaging'

        return mu_p_cond_g, precision_p_cond_g

    def _compute_llh_p_cond_g(self, test_ivecs, mu_p_cond_g, precision_p_cond_g):
        """ Compute llh of the same-speaker hypothesis, i.e.,
            p(x^p|x^g_1, x^g_2, ..., x^g_n) = N(mu_p_cond_g, Sigma_p_cond_g)
                mu_p_cond_g = V * inv(I + n * V^T * inv(Sigma) * V) * V^T * inv(Sigma) * [x^g_1 + x^g_2 + ... + x^g_n]
                Sigma_p_cond_g = V * inv(I + n * V^T * inv(Sigma) * V) * V^T + Sigma """

        const = log_det(precision_p_cond_g) / 2
        test_ivecs_centered = test_ivecs - mu_p_cond_g - self.mu
        sec_term_p_cond_g = sec_ord_term(test_ivecs_centered, precision_p_cond_g)

        return const - sec_term_p_cond_g / 2

    def save_scores(self, scores_file, scores):
        Path(Path(scores_file).parent).mkdir(exist_ok=True)

        if scores_file.endswith('npz'):
            np.savez(scores_file, enroll=self.trials_enroll_ids, test=self.trials_test_ids, llr=scores)
        else:
            scores = np.vstack([self.trials_enroll_ids, self.trials_test_ids, scores]).T
            pd.DataFrame(scores).to_csv(scores_file, index=False, header=False, sep=' ')

    def snorm(self, scores_ec, scores_ct, scores, select_ratio=0.1, n_top=500):
        """ Adaptive S-norm of scores
            select_ratio: ratio of the top scores selected for each enrollment or test ivector for S-norm
            Note that the order of the T-norm scoring pair (cohort v.s. test or test v.s. cohort) does not affect the
            speed of Snorm. """

        N_ec_se = int(scores_ec.shape[0] / self.n_utts_per_spk.shape[0] * select_ratio) if n_top is None else n_top
        N_ct_se = int(scores_ct.shape[0] / self.X_test.shape[0] * select_ratio) if n_top is None else n_top

        scores_ec = np.sort(scores_ec.reshape(self.n_utts_per_spk.shape[0], -1), axis=1, kind='mergesort')[:, -N_ec_se:]
        mu_ec, std_ec = np.mean(scores_ec, axis=1), np.std(scores_ec, axis=1)

        scores_ct = np.sort(scores_ct.reshape(-1, self.X_test.shape[0]), axis=0, kind='mergesort')[-N_ct_se:]
        mu_tc, std_tc = np.mean(scores_ct, axis=0), np.std(scores_ct, axis=0)

        # trials_enroll_ids_idx = get_filtered_idx(self.enroll_ids, self.trials_enroll_ids)
        trials_enroll_ids_idx = get_filtered_idx(np.unique(self.enroll_ids), self.trials_enroll_ids)  # sre
        mu_ec, std_ec = mu_ec[trials_enroll_ids_idx], std_ec[trials_enroll_ids_idx]

        trials_test_ids_idx = get_filtered_idx(self.test_ids, self.trials_test_ids)
        mu_tc, std_tc = mu_tc[trials_test_ids_idx], std_tc[trials_test_ids_idx]

        return ((scores - mu_ec) / std_ec + (scores - mu_tc) / std_tc) / 2


def load_trials(trials_file):
    """
    Load trials list
    The trials list should be sorted by 'enroll_id' and can be different with the original trials. This is
    for accelerating scoring, which requires an indexing operation on the eval data (see index_trials()).
    :return
        trials_enroll_ids, trials_test_ids: np tensor, [len(trials), ]
    """

    trials_file = f'{trials_file}.npz' if Path(f'{trials_file}.npz').exists() else trials_file

    if trials_file.endswith('.npz'):
        trails = np.load(trials_file, allow_pickle=True)
        return trails['enroll'], trails['test']
    else:
        trials = pd.read_csv(trials_file, sep=' ', names=['enroll', 'test', 'key'], dtype=str)
        trials = trials.sort_values(by='enroll', kind='mergesort')
        trials_enroll_ids, trials_test_ids = trials['enroll'].values, trials['test'].values
        np.savez(f'{trials_file}.npz', enroll=trials_enroll_ids, test=trials_test_ids, key=trials['key'].values)

        return trials_enroll_ids, trials_test_ids


def index_trials(trials_file, enroll_ids=None, test_ids=None, trials_enroll_ids=None, trials_test_ids=None):
    """
    Index enrollment and test data for accelerating scoring
    Note the enrolled speaker IDs in the trials list (trials_enroll_ids) must be sorted.
    :param trials_file: str
    :param enroll_ids: ndarray, speaker IDs of enrollment data
    :param test_ids: ndarray, utterances IDs of test data
    :param trials_enroll_ids: ndarray, enrolled speaker IDs in the trials list
    :param trials_test_ids: ndarray, test utterances IDs in the trials list
    :return:
        enroll_idx: list, index to select unique enrolled speakers in the enroll_ids,
                            len(enroll_idx) == len(enroll_ids_unique)
        n_utts_per_spk_enroll: ndarray, No. of utterances for each enrolled speaker
        test_idx: list, index to select test utterances that corresponds to each unique
                            enrolled speaker in the trials pairs, len(test_idx) == len(enroll_ids_unique)
    """

    file_split = trials_file.split('/')
    trial_name = file_split[-1].split('.')[0]
    index_file = f'{"/".join(file_split[:-1])}/index_{trial_name}.npz'

    if Path(index_file).exists():
        eval_index_npz = np.load(index_file, allow_pickle=True)
        enroll_idx, n_utts_per_spk_enroll, test_idx = eval_index_npz['enroll'], eval_index_npz['n_utt'], \
            eval_index_npz['test']
        enroll_idx = [x.astype(int) for x in enroll_idx]
        test_idx = [x.astype(int) for x in test_idx]

        return enroll_idx, n_utts_per_spk_enroll, test_idx
    else:
        enroll_ids_unique, n_utts_per_spk_enroll = np.unique(enroll_ids, return_counts=True)
        enroll_idx, test_idx = [], []

        for enroll_id in enroll_ids_unique:
            enroll_id_se_idx = np.where(enroll_ids == enroll_id)[0]
            trials_test_id_se = trials_test_ids[trials_enroll_ids == enroll_id]
            test_id_se_idx = get_filtered_idx(test_ids, trials_test_id_se)

            enroll_idx.append(enroll_id_se_idx)
            test_idx.append(test_id_se_idx)

        np.savez(index_file, enroll=np.asarray(enroll_idx, dtype='object'), n_utt=n_utts_per_spk_enroll,
                 test=np.asarray(test_idx, dtype='object'))

        return enroll_idx, n_utts_per_spk_enroll, test_idx


def log_det(mat):
    """ Compute the logdet of a symmetric matrix """
    L = cholesky(mat, lower=True)
    return 2 * np.sum(np.log(np.diag(L)))


def sec_ord_term(X, P):
    """ Compute the second order term of an exponent term of a Gaussian distribution: x^T * P * x
        X: row vectors, [N, F]
        P: precision matrix, [F, F]
        return: diag(X * P * X^T) """
    tmp = X @ P
    return np.einsum('ij, ij->i', tmp, X)
