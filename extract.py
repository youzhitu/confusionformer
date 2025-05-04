""" Extract embeddings """

import numpy as np
from utils.my_utils import select_plda_trn_info
from front_end.model_tdnn import TDNN
from front_end.model_res2net import Res2Net
from front_end.model_resnet import ResNet
from front_end.model_former import Former
from front_end.extractor import Extractor
import argparse
import os
from time import perf_counter
import shutil


# Set global parameters
parser = argparse.ArgumentParser(description='extract')
parser.add_argument('--model', default='confusionformer',
                    help='tdnn, resnet, res2net, conformer, confusionformer, transformer')
parser.add_argument('--filters', default='1024-1024-1024-1024-1536',
                    help='No. of channels of convolutional layers, 512-512-512-512-512-512-512-512-1536')
parser.add_argument('--kernel_sizes', default='5-3-3-3-1',
                    help='kernel size of convolutional layers, 5-1-3-1-3-1-3-1-1')
parser.add_argument('--dilations', default='1-2-3-4-1', help='dilation of convolutional layers, 1-1-2-1-3-1-4-1-1')
parser.add_argument('--pooling', default='ctdstats-128-0', help='stats, attention-500-1, ctdstats-256-0')
parser.add_argument('--emb_dims', default='192', help='embedding network config, 512-512')
parser.add_argument('--output_act', default='amsoftmax-0.25-30', help='softmax, amsoftmax-0.25-30, aamsoftmax-0.25-30')
parser.add_argument('--n_speaker', type=int, default=17982, help='vox1-17982, sre-5402, cnceleb-8361, ffsvc-374')
parser.add_argument('--former_cfg', default='(12,256,0.15,0,1e0,0,1e-12,0)|(4,127,2,0.1)|(4,0.1)|(2,15,0.1)|'
                    '(16,16,8,5,0,0)|fusion', help='(n_blks, d_model, drop_path, layer_scale, ls_init, '
                    'rmsn, ln_eps, mfa), (n_heads, max_rel_dist, att_ds, att_drop), (ff_expansion, ff_drop), '
                    '(conv_expansion, conv_kernel, conv_drop), (stft_cfg), rel_att_type')

parser.add_argument('--feat_dim', type=int, default=80, help='dimension of acoustic features')
parser.add_argument('--n_workers', type=int, default=2, help='No. of workers used in the dataloader')
parser.add_argument('--device', default='cuda:0', help='cuda, cpu')
parser.add_argument('--ckpt_dir', nargs='?', help='directory of model checkpoint')
parser.add_argument('--ckpt_num', nargs='?', type=int, help='checkpoint number for resuming training, default: None')
parser.add_argument('--n_jobs', type=int, default=1, help='No. of jobs for extracting x-vectors')
parser.add_argument('--job', type=int, default=0, help='job index of extraction')
parser.add_argument('--selected_dur', nargs='?', type=int,
                    help='duration of randomly selected utterances in No. of frames, e.g., 500 means 5s.')

parser.add_argument('--eval_dir', default='eval', help='directory of evaluation')
parser.add_argument('--task', default='voxceleb1', help='voxceleb1, sre, cnceleb1')
parser.add_argument('--extract_feat_map', action='store_true', default=False, help='whether to extract feature maps')
parser.add_argument('--speed_aug', action='store_true', default=False, help='whether to extract feature maps')
args = parser.parse_args()

print(f'-------------------------------------')
for arg, val in vars(args).items():
    print(f'[*] {arg}: {val}')
print(f'-------------------------------------\n')


if __name__ == '__main__':
    # ------------------------------------------------------------------------------------------------
    # Create a model instance
    # ------------------------------------------------------------------------------------------------
    model_args = {
        'feat_dim': args.feat_dim, 'filters': args.filters, 'kernel_sizes': args.kernel_sizes,
        'dilations': args.dilations, 'pooling': args.pooling, 'emb_dims': args.emb_dims,
        'n_class': args.n_speaker, 'output_act': args.output_act}

    if args.model == 'tdnn':
        model = TDNN(**model_args)
    elif args.model == 'resnet':
        model = ResNet(**model_args)
    elif args.model == 'res2net':
        model = Res2Net(**model_args)
    elif args.model in ['conformer', 'confusionformer', 'transformer']:
        model = Former(**model_args, cfg=args.former_cfg, name=args.model)
    else:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------------
    # Set data sources for x-vector extraction and select info for PLDA training
    # ------------------------------------------------------------------------------------------------
    meta_dir = 'meta/eval'
    plda_dir = f'{args.eval_dir}/plda'
    os.makedirs(plda_dir, exist_ok=True)

    print('Setting data sources...')
    ts = perf_counter()

    if args.task == 'voxceleb1':
        if args.job == 0:  # only args.job == 0 activates plda training data selection
            select_plda_trn_info(meta_dir, plda_dir, source='vox2_train', n_utts_per_spk=1)

        data_sources = ['plda_vox2_train', 'voxceleb1_all']  # 'train', 'voxceleb1_test', 'plda_train'
        # data_sources = ['voxceleb1_test']
        n_utts_per_partition = 10000
    elif args.task == 'cnceleb1':
        if args.job == 0:
            select_plda_trn_info(meta_dir, plda_dir, source='cnceleb_train')

        data_sources = ['plda_cnceleb_train', 'cnceleb1_enroll', 'cnceleb1_test']
        n_utts_per_partition = 2000
    else:
        raise NotImplementedError

    print(f'time of subset selection: {perf_counter() - ts} s\n')

    # ------------------------------------------------------------------------------------------------
    # Extract x-vectors
    # ------------------------------------------------------------------------------------------------
    for source in data_sources:
        print(f'Extracting x-vectors for {source}...')

        ts = perf_counter()

        ckpt_time = '_'.join(args.ckpt_dir.split('/')[-1].split('_')[-2:])
        if args.ckpt_num is None:
            ckpt_num = max([int(ckpt.split('-')[-1]) for ckpt in os.listdir(args.ckpt_dir)])
        else:
            ckpt_num = args.ckpt_num
        ckpt_path = f'{args.ckpt_dir}/ckpt-{ckpt_num}'

        extract_dir = f'extract/{ckpt_time}_{ckpt_num}_{source}'  # for saving temporary x-vectors
        os.makedirs(extract_dir, exist_ok=True)
        xvec_dir = f'{args.eval_dir}/xvectors/{source}'  # for saving final x-vectors
        os.makedirs(xvec_dir, exist_ok=True)

        extractor = Extractor(source=source, model=model, ckpt_path=ckpt_path, device=args.device,
                              extract_dir=extract_dir, extract_feat_map=args.extract_feat_map,
                              n_utts_per_partition=n_utts_per_partition, n_jobs=args.n_jobs,
                              job=args.job, selected_dur=args.selected_dur, n_workers=args.n_workers,
                              speed_aug=args.speed_aug)  #  & source.startswith('plda_')
        xvectors = extractor.extract()

        if xvectors.size:
            np.save(f'{xvec_dir}/xvector_{ckpt_time}_{ckpt_num}.npy', xvectors)

            n_non_zeros = [np.count_nonzero(np.abs(x) > .01) for x in xvectors]
            print(f'n_non_zeros > 0.01 -- mean: {np.mean(n_non_zeros):3}, std: {np.std(n_non_zeros):3}')
            n_non_zeros = [np.count_nonzero(np.abs(x) > .1) for x in xvectors]
            print(f'n_non_zeros > 0.1 -- mean: {np.mean(n_non_zeros):3}, std: {np.std(n_non_zeros):3}')
            n_non_zeros = [np.count_nonzero(np.abs(x) > 1.) for x in xvectors]
            print(f'n_non_zeros > 1.0 -- mean: {np.mean(n_non_zeros):3}, std: {np.std(n_non_zeros):3}\n')

            shutil.rmtree(extract_dir)  # Remove temporary x-vectors

        print(f'time of extraction: {perf_counter() - ts} s\n')

    print('To the END.\n\n')
    print()
