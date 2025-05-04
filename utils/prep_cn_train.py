""" Prepare train info for cnceleb """

import numpy as np
import pandas as pd
from utils.my_utils import rd_data_frame, wr_data_frame
from utils.prep_vox_train import create_spk2info, update_path2info
import os


meta_dir = './meta'
cn_meta_dir = f'{meta_dir}/cnceleb'
cn2_meta_dir = f'{meta_dir}/cnceleb2'
cn1_meta_dir = f'{meta_dir}/cnceleb1'
eval_meta_dir = f'{meta_dir}/eval'

srcs = ['cnceleb1', 'cnceleb2', 'cnceleb']


def main():
    # ------------------------------------------------------------------------------------------------
    # Extract the training part of cnceleb1/cnceleb path2info
    # ------------------------------------------------------------------------------------------------
    is_split_trn1 = False

    if is_split_trn1:
        cn1_path2info_file = f'{cn1_meta_dir}/path2info'
        cn_path2info_file = f'{cn_meta_dir}/path2info'

        cn1_path2info = rd_data_frame(cn1_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        cn_path2info = rd_data_frame(cn_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])

        enroll_path2info_file = f'{eval_meta_dir}/cnceleb1_enroll_path2info'
        test_path2info_file = f'{eval_meta_dir}/cnceleb1_test_path2info'

        enroll_path2info = rd_data_frame(enroll_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        test_path2info = rd_data_frame(test_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        eval_spk_ids = pd.concat([enroll_path2info['spk_id'], test_path2info['spk_id']], axis=0).unique()

        cn1_path2info = cn1_path2info[~cn1_path2info['spk_id'].isin(eval_spk_ids)]
        wr_data_frame(cn1_path2info_file, cn1_path2info)
        print(f'No. of training utts in cn1_path2info: {cn1_path2info.shape[0]}')  # 107953

        cn_path2info = cn_path2info[~cn_path2info['spk_id'].isin(eval_spk_ids)]
        wr_data_frame(cn_path2info_file, cn_path2info)
        print(f'No. of training utts in cn_path2info: {cn_path2info.shape[0]}\n')  # 632740

    # ------------------------------------------------------------------------------------------------
    # Add labels to path2info
    # ------------------------------------------------------------------------------------------------
    is_add_labels = False

    if is_add_labels:
        n_frames_min = 200

        for src in srcs:
            print(f'Creating path2info, spk2info for {src}...')

            path2info_file = f'{meta_dir}/{src}/path2info'
            spk2info_file = f'{meta_dir}/{src}/spk2info'
            path2info_ori_file = f'{meta_dir}/{src}/path2info_ori'

            if not os.path.exists(path2info_ori_file):
                print(f'{path2info_ori_file} does not exist, copying {path2info_file}...')
                os.system(f'cp {path2info_file} {path2info_ori_file}')
            else:
                print(f'{path2info_ori_file} exists, skipping creation for {src}...')
                continue

            try:
                path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
            except (Exception,):
                path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])

            print(f'No. of utts BEFORE selection: {path2info.shape[0]}')  # 107953 524787 632740
            path2info = path2info[path2info['n_sample'] >= 160 * n_frames_min + 240]  # Frame selection
            path2info.index = np.arange(path2info.shape[0])
            print(f'No. of utts AFTER selection: {path2info.shape[0]}')  # 72390 489717 562107

            spk2info = create_spk2info(path2info['spk_id'])  # Create spk2info
            path2info = update_path2info(path2info, spk2info)  # Update path2info, add labels
            print(f'No. of {src} spks: {spk2info.shape[0]}\n')  # 788 1995 2783

            wr_data_frame(spk2info_file, spk2info)
            wr_data_frame(path2info_file, path2info)

        print('spk2info creation DONE.\n')

    # ------------------------------------------------------------------------------------------------
    # Create train/validation spk2info, path2info
    # ------------------------------------------------------------------------------------------------
    is_create_val = False

    if is_create_val:
        for src in srcs:
            print(f'Creating {src} train/validation...')

            spk2info_file = f'{meta_dir}/{src}/spk2info'
            spk2info = rd_data_frame(spk2info_file, x_keys=['spk_id', 'n_utt', 'utt_offset', 'label'])

            path2info_file = f'{meta_dir}/{src}/path2info'
            path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])

            # Create validation
            utt_idx_per_spk = np.random.randint(0, spk2info['n_utt'])
            utt_offsets = spk2info['utt_offset'].values + utt_idx_per_spk
            val_path2info = path2info.iloc[utt_offsets]

            n_utts = 1024
            val_idx = np.random.permutation(val_path2info.shape[0])
            val_path2info = val_path2info.iloc[val_idx[:n_utts]]
            val_path2info = val_path2info.sort_values(by=['utt_path'], kind='mergesort')
            val_path2info.index = np.arange(val_path2info.shape[0])
            val_spk2info = create_spk2info(val_path2info['spk_id'])

            wr_data_frame(f'{meta_dir}/{src}/val_spk2info', val_spk2info)
            wr_data_frame(f'{meta_dir}/{src}/val_path2info', val_path2info)
            wr_data_frame(f'{eval_meta_dir}/{src}_val_path2info', val_path2info.drop(['label'], axis=1))  # for eval

            # Create train
            train_path2info = path2info[~path2info['utt_path'].isin(val_path2info['utt_path'])]
            train_path2info.index = np.arange(train_path2info.shape[0])
            train_spk2info = create_spk2info(train_path2info['spk_id'])
            train_path2info = update_path2info(train_path2info, train_spk2info)

            wr_data_frame(f'{meta_dir}/{src}/train_spk2info', train_spk2info)
            wr_data_frame(f'{meta_dir}/{src}/train_path2info', train_path2info)
            wr_data_frame(f'{eval_meta_dir}/{src}_train_path2info', train_path2info.drop(['label'], axis=1))  # for eval

        print('Train/validation creation DONE.\n')

    # ------------------------------------------------------------------------------------------------
    # Create validation trials
    # ------------------------------------------------------------------------------------------------
    is_create_val_trials = False

    if is_create_val_trials:
        print()

    print('To the END.\n\n')


if __name__ == '__main__':
    main()
