""" Backend processing for CN-Celeb """

# python be_cnceleb.py --lda_lat_dim=250 --plda_lat_dim=250 --is_plda --ckpt_num=20 --ckpt_time=20210727_2001

import numpy as np
from time import perf_counter
import argparse
from utils.my_utils import init_logger, rd_data_frame
from back_end.preprocess import Preprocessor
from back_end.sgplda_train import SGPLDATrainer
from back_end.sgplda_score import SGPLDAScorer
from back_end.cosine_score import CosineScorer
from utils.eval_metrics import eval_performance
from pathlib import Path


# Set parameters
parser = argparse.ArgumentParser(description='be_cnceleb')
parser.add_argument('--remote', action='store_true', default=False, help='remote or local')
parser.add_argument('--task', default='cnceleb', help='cnceleb')
parser.add_argument('--is_plda', action='store_true', default=False, help='PLDA or cosine scoring')
# parser.add_argument('--preprocess', default='center-lda-whiten-ln', help='pre-process')
parser.add_argument('--lda_lat_dim', type=int, default=180, help='dimension of latent space for LDA')
parser.add_argument('--plda_lat_dim', type=int, default=180, help='dimension of latent space for PLDA')
parser.add_argument('--n_iters', type=int, default=20, help='No. of EM iterations for PLDA training')
parser.add_argument('--is_fix_model', action='store_true', default=False, help='fix the trained model')
parser.add_argument('--scoring_type', default='multi_session', help='multi_session, ivec_averaging, score_averaging')
parser.add_argument('--is_snorm', action='store_true', default=True, help='perform S-norm')
parser.add_argument('--n_top', type=int, default=800, help='ratio of top cohort scores selected for snorm')
parser.add_argument('--ckpt_time', default='20210727_2001', help='20210727_2001')
parser.add_argument('--ckpt_num', type=int, default=20, help='index. of checkpoint')
args = parser.parse_args()

# ------------------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------------------
if args.remote:
    eval_dir = './eval'
    eval_meta_dir = './meta/eval'
    trials_dir = './trials'
else:
    eval_dir = '../eval'
    eval_meta_dir = '../meta/eval'
    trials_dir = '../trials'

logger = init_logger(logger_name='be_cnceleb', log_path='log/be_cnceleb.log')
preprocess = 'center-lda-whiten-ln' if args.is_plda else 'ln'

# logger.info(f'-------------------------------------')
# for arg, val in vars(args).items():
#     logger.info(f'[*] {arg}: {val}')
# logger.info(f'[*] preprocess: {preprocess}')
# logger.info(f'-------------------------------------')

paras_dir = f'{eval_dir}/plda/cnceleb1'
Path(paras_dir).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------------------
# Load PLDA training data
# ------------------------------------------------------------------------------------------------
trn_xvec_file = f'{eval_dir}/xvectors/plda_cnceleb_train/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
trn_info_file = f'{eval_meta_dir}/plda_cnceleb_train_path2info'

X_trn = np.load(trn_xvec_file)
trn_info = rd_data_frame(trn_info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
spk_ids_trn, n_samples_trn = trn_info['spk_id'].values, trn_info['n_sample'].values
global_mean = X_trn.mean(0)


# ------------------------------------------------------------------------------------------------
# Initialize preprocessor
# ------------------------------------------------------------------------------------------------
# logger.info(f'Initializing a preprocessor...')
# t_s = perf_counter()
preprocessor = Preprocessor(
    X_trn, spk_ids_trn, lda_lat_dim=args.lda_lat_dim, paras_dir=paras_dir, preprocess=preprocess)
if not args.is_fix_model:
    preprocessor.train()
# logger.info(f'Time of initializing the preprocessor: {perf_counter() - t_s}s.\n')


# ------------------------------------------------------------------------------------------------
# Train an SGPLDA model
# ------------------------------------------------------------------------------------------------
plda_model_file = f'{paras_dir}/sgplda_paras.npz'

if args.is_plda:
    if not args.is_fix_model:
        X = preprocessor.transform(X_trn)
        logger.info(f'Training an SGPLDA model...')

        t_s = perf_counter()
        plda_trainer = SGPLDATrainer(X, spk_ids_trn, lat_dim=args.plda_lat_dim, n_iters=args.n_iters)
        plda_trainer.train(plda_model_file)
        logger.info(f'Time of the SGPLDA training: {perf_counter() - t_s}s.\n')


# ------------------------------------------------------------------------------------------------
# Compute cnceleb1 scores
# ------------------------------------------------------------------------------------------------
# Load cohort for snorm
X_snorm = X_trn
X_snorm = preprocessor.transform(X_snorm, mu=global_mean)

# Load cnceleb1 enrollment
eval_enroll_xvecs_file = f'{eval_dir}/xvectors/cnceleb1_enroll/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
enroll_info_file = f'{eval_meta_dir}/cnceleb1_enroll_path2info'

X_enroll_eval = np.load(eval_enroll_xvecs_file)
X_enroll_eval = preprocessor.transform(X_enroll_eval, mu=global_mean)
eval_enroll_ids = rd_data_frame(enroll_info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values

# Load cnceleb1 test
eval_test_xvecs_file = f'{eval_dir}/xvectors/cnceleb1_test/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
test_info_file = f'{eval_meta_dir}/cnceleb1_test_path2info'

X_test_eval = np.load(eval_test_xvecs_file)
X_test_eval = preprocessor.transform(X_test_eval, mu=global_mean)
eval_test_ids = rd_data_frame(test_info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values

# scoring
print(f'scoring ckpt_path: {args.ckpt_time}-{args.ckpt_num}...')
eval_trials_file = f'{trials_dir}/cnceleb1/trials'

if args.is_plda:
    eval_scores_file = f'{eval_dir}/scores/scores_{args.task}_plda.npz'
    eval_scores_snorm_file = f'{eval_dir}/scores/scores_{args.task}_plda_snorm.npz'
else:
    eval_scores_file = f'{eval_dir}/scores/scores_{args.task}.npz'
    eval_scores_snorm_file = f'{eval_dir}/scores/scores_{args.task}_snorm.npz'

if args.is_plda:
    print(f'cnceleb1 PLDA scoring...')
    t_s = perf_counter()
    scorer = SGPLDAScorer(
        X_enroll_eval, X_test_eval, eval_enroll_ids, eval_test_ids, trials_file=eval_trials_file,
        sgplda_paras_file=plda_model_file, scoring_type=args.scoring_type)
    scorer.score(eval_scores_file)
    print(f'Time of the cnceleb1 PLDA scoring: {perf_counter() - t_s}s.\n')
else:
    print(f'cnceleb1 Cosine scoring...')
    t_s = perf_counter()
    scorer = CosineScorer(
        X_enroll_eval, X_test_eval, eval_enroll_ids, eval_test_ids, trials_file=eval_trials_file,
        scoring_type=args.scoring_type)
    scorer.score(eval_scores_file)
    print(f'Time of the cnceleb1 Cosine scoring: {perf_counter() - t_s}s.\n')

if args.is_snorm:
    if args.is_plda:
        print(f'cnceleb1 PLDA scoring with S-norm...')
        t_s = perf_counter()
        scorer.score(eval_scores_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
        print(f'Time of the cnceleb1 PLDA scoring with S-norm: {perf_counter() - t_s}s.\n')
    else:
        print(f'cnceleb1 Cosine scoring with S-norm...')
        t_s = perf_counter()
        scorer.score(eval_scores_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
        print(f'Time of the cnceleb1 Cosine scoring with S-norm: {perf_counter() - t_s}s.\n')


# ------------------------------------------------------------------------------------------------
# Compute EER, minDCF and actDCF
# ------------------------------------------------------------------------------------------------
eval_trials_file = f'{eval_trials_file}.npz'
p_targets = [0.01, 0.001]

logger.info(f'==============================================================================')
logger.info(f'cnceleb1, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
eer, minDCFs = eval_performance(eval_scores_file, eval_trials_file, p_targets, c_miss=1, c_fa=1)
logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
logger.info(f'==============================================================================\n')

if args.is_snorm:
    logger.info(f'==============================================================================')
    logger.info(f'cnceleb1-snorm, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
    eer, minDCFs = eval_performance(eval_scores_snorm_file, eval_trials_file, p_targets, c_miss=1, c_fa=1)
    logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
    logger.info(f'==============================================================================\n')


logger.info('To the END.\n\n')
