""" Compute EER, minDCF """

import numpy as np
import pandas as pd


def eval_performance(scores, trial_keys, p_targets, c_miss=1, c_fa=1):
    """
    :param scores, np tensor (of float) or str
    :param trial_keys, np tensor (of int) or str
    :param p_targets, list (of float)
    :param c_miss, float
    :param c_fa, float
    :return
        eer: float
        minDCF: list (of float)
    """

    scr_trials = None

    if isinstance(scores, str):
        if scores.endswith('npz'):
            scr_trials = np.load(scores, allow_pickle=True)
        else:
            scr_trials = pd.read_csv(scores, sep=' ', names=['enroll', 'test', 'llr'], dtype=str)

        scr_trials = pd.DataFrame({'enroll': scr_trials['enroll'], 'test': scr_trials['test'],
                                   'llr': scr_trials['llr']})
        scores = scr_trials['llr'].values.astype(float)

    if isinstance(trial_keys, str):
        if trial_keys.endswith('npz'):
            key_trials = np.load(trial_keys, allow_pickle=True)
        else:
            key_trials = pd.read_csv(trial_keys, sep=' ', names=['enroll', 'test', 'key'], dtype=str)

        key_trials = pd.DataFrame({'enroll': key_trials['enroll'], 'test': key_trials['test'],
                                   'key': key_trials['key']})

        if isinstance(scores, str):
            assert scr_trials[['enroll', 'test']].equals(key_trials[['enroll', 'test']]), \
                'enroll-test pairs are different in score_trials and key_trials!'

        trial_keys = key_trials['key'].values
        trial_keys = (trial_keys == 'target').astype(int) if 'target' in trial_keys[0] else trial_keys.astype(int)

    fnr, fpr = compute_error_rates(scores, trial_keys)
    eer = compute_eer(fnr, fpr)

    minDCF = []

    for p_tar in p_targets:
        minDCF.append(compute_minDCF(fnr, fpr, p_tar, c_miss=c_miss, c_fa=c_fa))

    return eer, minDCF


def compute_error_rates(scores, labels, weights=None):
    """ Compute false negative rates (FNRs) and false positive rates (FPRs) given trial scores and their labels
        A weights option is also provided to equalize the counts over score partitions.
        :param scores: np tensor
        :param labels: np tensor
        :param weights: np tensor
        :return
            fnr, fpr: np tensor
    """

    sorted_idx = np.argsort(scores, kind='mergesort')
    labels = labels[sorted_idx]

    if weights is not None:
        weights = weights[sorted_idx]
    else:
        weights = np.ones(labels.shape[0])

    tgt_wghts = weights * (labels == 1)
    imp_wghts = weights * (labels == 0)

    fnr = np.cumsum(tgt_wghts) / np.sum(tgt_wghts)
    fpr = 1 - np.cumsum(imp_wghts) / np.sum(imp_wghts)

    return fnr, fpr


def compute_eer(fnr, fpr):
    """ Compute the equal error rate (EER) given FNR and FPR values calculated for a range of operating points
        on the DET curve """

    diff_pm_fa = fnr - fpr
    x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
    x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
    a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))

    return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_minDCF(fnr, fpr, p_target, c_miss=1, c_fa=1):
    """ Compute normalized minimum detection cost function (DCF) given the costs for false accepts and false rejects
        as well as a prior probability for target speakers """

    c_det = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
    c_det_min = min(c_det)
    c_default = min(c_miss * p_target, c_fa * (1 - p_target))

    return c_det_min / c_default

