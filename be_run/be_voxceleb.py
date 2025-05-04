""" Backend processing for Voxceleb1 """

# python be_voxceleb.py --lda_lat_dim=180 --plda_lat_dim=180 --is_plda --ckpt_num=20 --ckpt_time=20210727_2001

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
parser = argparse.ArgumentParser(description='be_voxceleb')
parser.add_argument('--eval_dir', default='./eval', help='directory of evaluation')
parser.add_argument('--eval_meta_dir', default='./meta/eval', help='directory of evaluation meta info')
parser.add_argument('--is_plda', action='store_true', default=False, help='PLDA or cosine scoring')
# parser.add_argument('--preprocess', default='center-lda-whiten-ln', help='pre-process')
parser.add_argument('--lda_lat_dim', type=int, default=250, help='dimension of latent space for LDA')
parser.add_argument('--plda_lat_dim', type=int, default=250, help='dimension of latent space for PLDA')
parser.add_argument('--n_iters', type=int, default=20, help='No. of EM iterations for PLDA training')
parser.add_argument('--is_fix_model', action='store_true', default=False, help='fix the trained model')
parser.add_argument('--scoring_type', default='multi_session', help='multi_session, ivec_averaging, score_averaging')
parser.add_argument('--is_snorm', action='store_true', default=True, help='perform S-norm')
parser.add_argument('--n_top', type=int, default=150, help='ratio of top cohort scores selected for snorm')
parser.add_argument('--ckpt_time', default='20230522_1711', help='checkpoint time')
parser.add_argument('--ckpt_num', type=int, default=42, help='index. of checkpoint')
args = parser.parse_args()

# ------------------------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------------------------
logger = init_logger(logger_name='be_voxceleb', log_path='log/be_voxceleb.log')
preprocess = 'center-lda-whiten-ln' if args.is_plda else 'ln'

# logger.info(f'-------------------------------------')
# for arg, val in vars(args).items():
#     logger.info(f'[*] {arg}: {val}')
# logger.info(f'[*] preprocess: {preprocess}')
# logger.info(f'-------------------------------------')

paras_dir = f'{args.eval_dir}/plda/voxceleb1'
Path(paras_dir).mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------------------------
# Load PLDA training data
# ------------------------------------------------------------------------------------------------
trn_xvec_file = f'{args.eval_dir}/xvectors/plda_vox2_train/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
trn_info_file = f'{args.eval_meta_dir}/plda_vox2_train_path2info'

X_trn = np.load(trn_xvec_file)
trn_info = rd_data_frame(trn_info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
spk_ids_trn, n_samples_trn = trn_info['spk_id'].values, trn_info['n_sample'].values

# global_mean = X_trn.mean(0)

# ------------------------------------------------------------------------------------------------
# Initialize preprocessor
# ------------------------------------------------------------------------------------------------
# logger.info(f'Initializing a preprocessor...')
t_s = perf_counter()
preprocessor = Preprocessor(X_trn, spk_ids_trn, lda_lat_dim=args.lda_lat_dim, paras_dir=paras_dir,
                            preprocess=preprocess)
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
# Compute scores
# ------------------------------------------------------------------------------------------------
# Load x-vectors
enroll_xvecs_file = f'{args.eval_dir}/xvectors/voxceleb1_all/xvector_{args.ckpt_time}_{args.ckpt_num}.npy'
enroll_info_file = f'{args.eval_meta_dir}/voxceleb1_all_path2info'
voxceleb1_test_idx_file = f'trials/voxceleb1/index_voxceleb1_test.npy'

X_enroll = np.load(enroll_xvecs_file)
enroll_ids = rd_data_frame(enroll_info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])['utt_id'].values
voxceleb1_test_idx = np.load(voxceleb1_test_idx_file)

# X_enroll -= global_mean
X_enroll = preprocessor.transform(X_enroll)
X_test = X_enroll.copy()
test_ids = enroll_ids.copy()

# Load cohort for snorm
X_snorm = None

if args.is_snorm:
    X_snorm = X_trn

    X_snorm -= X_snorm.mean(0)
    X_snorm = preprocessor.transform(X_snorm)
    # logger.info(f'Time of cohort loading: {perf_counter() - t_s}s.\n')

# Perform scoring
trials_file = f'trials/voxceleb1/trials_voxceleb'
trials_all_file = f'trials/voxceleb1/trials_voxceleb_all'
trials_hard_file = f'trials/voxceleb1/trials_voxceleb_hard'

scores_file = f'{args.eval_dir}/scores/scores_voxceleb1.npz'
scores_snorm_file = f'{args.eval_dir}/scores/scores_voxceleb1_snorm.npz'
scores_all_file = f'{args.eval_dir}/scores/scores_voxceleb1_all.npz'
scores_all_snorm_file = f'{args.eval_dir}/scores/scores_voxceleb1_all_snorm.npz'
scores_hard_file = f'{args.eval_dir}/scores/scores_voxceleb1_hard.npz'
scores_hard_snorm_file = f'{args.eval_dir}/scores/scores_voxceleb1_hard_snorm.npz'

print(f'scoring ckpt_path: {args.ckpt_time}-{args.ckpt_num}...')

if args.is_plda:
    logger.info(f'voxceleb1 PLDA scoring...')
    t_s = perf_counter()
    scores_file = f'{args.eval_dir}/scores/scores_voxceleb1_plda.npz'
    scorer = SGPLDAScorer(X_enroll, X_test, enroll_ids, test_ids, trials_file=trials_file,
                          sgplda_paras_file=plda_model_file, scoring_type=args.scoring_type)
    scorer.score(scores_file)
    logger.info(f'Time of the voxceleb1 PLDA scoring: {perf_counter() - t_s}s.\n')

    if args.is_snorm:
        logger.info(f'voxceleb1 PLDA scoring with S-norm...')
        t_s = perf_counter()
        scores_snorm_file = f'{args.eval_dir}/scores/scores_voxceleb1_plda_snorm.npz'
        scorer.score(scores_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
        logger.info(f'Time of the voxceleb1 PLDA scoring with S-norm: {perf_counter() - t_s}s.\n')
else:
    print(f'voxceleb1_test Cosine scoring...')
    t_s = perf_counter()
    scorer = CosineScorer(X_enroll[voxceleb1_test_idx], X_test[voxceleb1_test_idx], enroll_ids[voxceleb1_test_idx],
                          test_ids[voxceleb1_test_idx], trials_file=trials_file, scoring_type=args.scoring_type)
    scorer.score(scores_file)
    print(f'Time of the voxceleb1 Cosine scoring: {perf_counter() - t_s}s.\n')

    if args.is_snorm:
        print(f'voxceleb1 Cosine scoring with S-norm...')
        t_s = perf_counter()
        scorer.score(scores_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
        print(f'Time of the voxceleb1 Cosine scoring with S-norm: {perf_counter() - t_s}s.\n')

    print(f'voxceleb1_all Cosine scoring...')
    t_s = perf_counter()
    scorer = CosineScorer(X_enroll, X_test, enroll_ids, test_ids, trials_file=trials_all_file,
                          scoring_type=args.scoring_type)
    scorer.score(scores_all_file)
    print(f'Time of the voxceleb1_all Cosine scoring: {perf_counter() - t_s}s.\n')

    if args.is_snorm:
        print(f'voxceleb1_all Cosine scoring with S-norm...')
        t_s = perf_counter()
        scorer.score(scores_all_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
        print(f'Time of the voxceleb1_all Cosine scoring with S-norm: {perf_counter() - t_s}s.\n')

    print(f'voxceleb1_hard Cosine scoring...')
    t_s = perf_counter()
    scorer = CosineScorer(X_enroll, X_test, enroll_ids, test_ids, trials_file=trials_hard_file,
                          scoring_type=args.scoring_type)
    scorer.score(scores_hard_file)
    print(f'Time of the voxceleb1_hard Cosine scoring: {perf_counter() - t_s}s.\n')

    if args.is_snorm:
        print(f'voxceleb1_hard Cosine scoring with S-norm...')
        t_s = perf_counter()
        scorer.score(scores_hard_snorm_file, X_snorm, X_snorm, is_snorm=True, n_top=args.n_top)
        print(f'Time of the voxceleb1_hard Cosine scoring with S-norm: {perf_counter() - t_s}s.\n')


# ------------------------------------------------------------------------------------------------
# Compute EER, minDCF and actDCF
# ------------------------------------------------------------------------------------------------
trials_file = f'{trials_file}.npz'
trials_all_file = f'{trials_all_file}.npz'
trials_hard_file = f'{trials_hard_file}.npz'
p_targets = [0.01, 0.001]

logger.info(f'==============================================================================')
logger.info(f'voxceleb1_test, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
eer, minDCFs = eval_performance(scores_file, trials_file, p_targets, c_miss=1, c_fa=1)
# logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}, minDCF (p_tar=1e-3): {minDCFs[1]:.3f}')
logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
logger.info(f'==============================================================================\n')

if args.is_snorm:
    logger.info(f'==============================================================================')
    logger.info(f'voxceleb1_test-snorm, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
    eer, minDCFs = eval_performance(scores_snorm_file, trials_file, p_targets, c_miss=1, c_fa=1)
    logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
    logger.info(f'==============================================================================\n')

logger.info(f'==============================================================================')
logger.info(f'voxceleb1_all, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
eer, minDCFs = eval_performance(scores_all_file, trials_all_file, p_targets, c_miss=1, c_fa=1)
logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
logger.info(f'==============================================================================\n')

if args.is_snorm:
    logger.info(f'==============================================================================')
    logger.info(f'voxceleb1_all-snorm, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
    eer, minDCFs = eval_performance(scores_all_snorm_file, trials_all_file, p_targets, c_miss=1, c_fa=1)
    logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
    logger.info(f'==============================================================================\n')

logger.info(f'==============================================================================')
logger.info(f'voxceleb1_hard, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
eer, minDCFs = eval_performance(scores_hard_file, trials_hard_file, p_targets, c_miss=1, c_fa=1)
logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
logger.info(f'==============================================================================\n')

if args.is_snorm:
    logger.info(f'==============================================================================')
    logger.info(f'voxceleb1_hard-snorm, ckpt_time: {args.ckpt_time}, ckpt_num: {args.ckpt_num}...')
    eer, minDCFs = eval_performance(scores_hard_snorm_file, trials_hard_file, p_targets, c_miss=1, c_fa=1)
    logger.info(f'EER: {eer * 100:.2f} %, minDCF (p_tar=1e-2): {minDCFs[0]:.3f}')
    logger.info(f'==============================================================================\n')

logger.info('To the END.\n\n')
