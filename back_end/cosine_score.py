""" Cosine scoring """

import numpy as np
from back_end.sgplda_score import SGPLDAScorer


class CosineScorer(SGPLDAScorer):
    def __init__(self, *args, **kwargs):
        super(CosineScorer, self).__init__(*args, **kwargs)

    def _init_scorer(self):
        self._load_trials()
        self._load_trials_idx()

    def score(self, scores_file=None, X_enroll_in=None, X_test_in=None, is_snorm=False, select_ratio=0.1, n_top=500):
        """ Scoring main function
        If X_enroll_in and X_test_in are not provided, which is the default setting, scores will be computed
        according to the trials list.
        If both X_enroll_in and X_test_in are provided, but is_snorm is disabled, scores will be simply
        computed for each {X_enroll_in, X_test_in} pair.
        If both X_enroll_in and X_test_in are provided, and is_snorm is enabled, Snorm will be performed in which
        snorm scores ("scores_ec" and "scores_ct") are first computed and raw scores are then normalized.

        Snorm consists of two stages: Znorm and Tnorm. Note that if multi-session scoring is selected,
        the "is_tnorm" flag should be enabled to make sure the "enroll_ivecs" in the scoring function
        are correctly picked up. """

        if X_enroll_in is not None and X_test_in is not None:
            if is_snorm:
                scores_ec = (self.X_enroll @ X_enroll_in.T).ravel()
                scores_ct = (X_test_in @ self.X_test.T).ravel()
                scores = self.cosine_scoring(self.X_enroll, self.X_test, is_trials_scoring=True, is_tnorm=False)
                scores = self.snorm(scores_ec, scores_ct, scores, select_ratio=select_ratio, n_top=n_top)
            else:
                scores = self.cosine_scoring(X_enroll_in, X_test_in, is_trials_scoring=False, is_tnorm=True)
        else:
            scores = self.cosine_scoring(self.X_enroll, self.X_test, is_trials_scoring=True, is_tnorm=False)

        if scores_file is not None:
            self.save_scores(scores_file, scores)

        return scores

    def cosine_scoring(self, X_enroll, X_test, is_trials_scoring=True, is_tnorm=False):
        """ Compute scores for every enrolled speaker
            :param
            is_trials_scoring: indicating whether the current scoring is trials scoring or S-norm scoring
                default: trials scoring
            is_tnorm: indicating whether the current S-norm scoring is Z-norm scoring or T-norm scoring
                Note that it is only valid when is_trials_scoring is disabled, default: False (Z-norm scoring)
            """

        n_unique_enroll_ids = X_enroll.shape[0] if is_tnorm else self.n_utts_per_spk.shape[0]
        scores = []

        for i in range(n_unique_enroll_ids):
            enroll_ivecs = X_enroll[self.X_enroll_idx[i]] if not is_tnorm else np.expand_dims(X_enroll[i], axis=0)
            test_ivecs = X_test[self.X_test_idx[i]] if is_trials_scoring else X_test

            if self.scoring_type == 'score_averaging':
                cos_scores = np.einsum('ik, jk->ij', enroll_ivecs, test_ivecs)
                cos_scores = cos_scores.reshape(enroll_ivecs.shape[0], -1).mean(0)
            else:
                cos_scores = test_ivecs @ enroll_ivecs.mean(0)

            scores.append(cos_scores)

        return np.concatenate(scores)
