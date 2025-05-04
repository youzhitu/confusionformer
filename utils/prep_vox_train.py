"""
Prepare train/validation info for voxceleb

This code takes './meta/vox2/path2info' as input and produces './meta/vox2/train_path2info' and
'./meta/vox2/train_spk2info', which are used for training the spk-emb-nns. The spk2info has the following keys
['spk_id', 'n_utt', 'utt_offset', 'label']. n_utt denotes No. of utts per spk, and utt_offset means the utt offset
of the current speaker among all utts.
id00012 164 0 0
id00015 500 164 1
id00016 179 664 2

Also, 1024 utts from vox1 are selected as a validation dataset.
"""

import numpy as np
import pandas as pd
from utils.my_utils import rd_data_frame, wr_data_frame, get_filtered_idx
import os


meta_dir = './meta'
eval_meta_dir = f'{meta_dir}/eval'

srcs = ['vox', 'vox1', 'vox2']


def main():
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

            print(f'path2info.shape[0] BEFORE selection: {path2info.shape[0]}')
            path2info = path2info[path2info['n_sample'] >= 160 * n_frames_min + 240]  # Frame selection
            path2info.index = np.arange(path2info.shape[0])
            print(f'path2info.shape[0] AFTER selection: {path2info.shape[0]}\n')

            spk2info = create_spk2info(path2info['spk_id'])  # Create spk2info
            path2info = update_path2info(path2info, spk2info)  # Update path2info, add labels

            wr_data_frame(spk2info_file, spk2info)
            wr_data_frame(path2info_file, path2info)

        print('spk2info creation DONE.')

    # ------------------------------------------------------------------------------------------------
    # Create train/validation spk2info, path2info
    # ------------------------------------------------------------------------------------------------
    is_create_val = False

    if is_create_val:
        for src in ['vox1']:  # srcs
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

        print('Train/validation creation DONE.')

    # ------------------------------------------------------------------------------------------------
    # Create validation trials
    # ------------------------------------------------------------------------------------------------
    is_create_val_trials = False

    if is_create_val_trials:
        print()

    print('To the END.\n\n')


def create_spk2info(spk_ids):
    """
    :param spk_ids: Series, one column of path2info
    :return:
        spk2info: DataFrame, with keys ['spk_id', 'n_utt', 'utt_offset', 'label']
    """

    spk_ids_unique, n_utts_per_spk = np.unique(spk_ids, return_counts=True)

    # Maintain the order of spk_ids in the path2info
    spk_ids_unique_ori = spk_ids.unique()

    if not np.array_equal(spk_ids_unique, spk_ids_unique_ori):
        ori_idx = get_filtered_idx(spk_ids_unique, spk_ids_unique_ori)
        spk_ids_unique, n_utts_per_spk = spk_ids_unique[ori_idx], n_utts_per_spk[ori_idx]

    spk_ids_unique = pd.Series(spk_ids_unique, name='spk_id')
    n_utts_per_spk = pd.Series(n_utts_per_spk, name='n_utt')
    utt_offsets = pd.Series(np.hstack([0, np.cumsum(n_utts_per_spk)[:-1]]), name='utt_offset')
    labels = pd.Series(np.arange(spk_ids_unique.shape[0]), name='label')
    spk2info = pd.concat([spk_ids_unique, n_utts_per_spk, utt_offsets, labels], axis=1)

    return spk2info


def update_path2info(path2info, spk2info):
    """
    Add a 'label' column to path2info
    :param path2info: DataFrame, with keys ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']
    :param spk2info: DataFrame, with keys ['spk_id', 'n_utt', 'utt_offset', 'label']
    :return:
        path2info: DataFrame, with keys ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label']
    """

    labels = spk2info['label'].repeat(spk2info['n_utt'])
    labels.index = np.arange(labels.shape[0])

    if 'label' in path2info.keys():
        path2info = path2info.drop(['label'], axis=1)

    path2info = pd.concat([path2info, labels], axis=1)

    return path2info


if __name__ == '__main__':
    main()
