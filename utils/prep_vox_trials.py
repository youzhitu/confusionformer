""" Prepare vox-E, vox-H trials """

import numpy as np
import pandas as pd
from utils.my_utils import rd_data_frame, wr_data_frame, get_filtered_idx


trials_list_dir = '../corpus/voxceleb1/test'
trials_dir = './trials/voxceleb1'
meta_dir = './meta'


# Convert trials list, the trials list should be sorted by 'enroll_id'
for trial_type in ['ori', 'all', 'hard']:
    trials_file = f'{trials_list_dir}/list_test_{trial_type}_clean.txt'
    trials = rd_data_frame(trials_file, x_keys=['key', 'enroll_id', 'test_id'])

    trials_keys = trials['key']
    trials_keys = trials_keys.str.replace('0', 'nontarget', regex=False)
    trials_keys = trials_keys.str.replace('1', 'target', regex=False)

    trials_enroll_ids = trials['enroll_id']
    trials_enroll_ids = trials_enroll_ids.str.replace('/', '-', regex=False).str.split('.', n=0, expand=True)[0]

    trials_test_ids = trials['test_id']
    trials_test_ids = trials_test_ids.str.replace('/', '-', regex=False).str.split('.', n=0, expand=True)[0]

    trials = pd.concat([trials_enroll_ids, trials_test_ids, trials_keys], axis=1)
    trials.columns = ['enroll_id', 'test_id', 'key']
    trials = trials.sort_values(by=['enroll_id'], kind='mergesort')  # sort
    trials.index = np.arange(trials.shape[0])
    trials_file = f'{trials_dir}/trials_voxceleb_{trial_type}'
    wr_data_frame(trials_file, trials.astype(str))

    enroll_spks = trials_enroll_ids.str.split('-', n=0, expand=True)[0].unique()
    test_spks = trials_test_ids.str.split('-', n=0, expand=True)[0].unique()
    print(f'{trial_type} -- No. of enroll spks: {len(enroll_spks)}, No. of test spks: {len(test_spks)}')  # 1251/1190

    enroll_utts = trials_enroll_ids.unique()
    test_utts = trials_test_ids.unique()
    print(f'{trial_type} -- No. of enroll utts: {len(enroll_utts)}, No. of test utts: {len(test_utts)}')

    unique_utts = np.unique(np.concatenate([enroll_utts, test_utts]))
    print(f'{trial_type} -- No. of unique utts: {len(unique_utts)}\n')


# Check enroll-test utterance
trials_file = f'{trials_dir}/trials_voxceleb_all'
trials_all = rd_data_frame(trials_file, x_keys=['enroll_id', 'test_id', 'key'])
trials_all_enroll_ids = trials_all['enroll_id'].unique()
print(f'all -- No. of unique enroll_ids: {len(trials_all_enroll_ids)}')  # 145160

trials_file = f'{trials_dir}/trials_voxceleb_hard'
trials_hard = rd_data_frame(trials_file, x_keys=['enroll_id', 'test_id', 'key'])
trials_hard_enroll_ids = trials_hard['enroll_id'].unique()
print(f'hard -- No. of unique enroll_ids: {len(trials_hard_enroll_ids)}')  # 137924

unique_enroll_ids = np.unique(np.concatenate([trials_all_enroll_ids, trials_hard_enroll_ids]))
print(f'No. of unique enroll_ids: {len(unique_enroll_ids)}\n')  # 145160


# Create voxceleb1_all path2info
vox1_dev_path2info_file = f'{meta_dir}/vox1/path2info'
vox1_dev_path2info = rd_data_frame(vox1_dev_path2info_file,
                                   ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])
vox1_test_path2info_file = f'{meta_dir}/eval/voxceleb1_test_path2info'
vox1_test_path2info = rd_data_frame(vox1_test_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
vox1_path2info = pd.concat([vox1_dev_path2info[['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']],
                            vox1_test_path2info], axis=0).sort_values(by=['utt_id'], kind='mergesort')  # sort
vox1_path2info.index = np.arange(vox1_path2info.shape[0])

utt_ids = vox1_path2info['utt_id']
unique_enroll_ids = pd.Series(unique_enroll_ids)
assert unique_enroll_ids[unique_enroll_ids.isin(utt_ids)].shape[0] == unique_enroll_ids.shape[0], \
    'some of the trials utts are not in vox1_path2info!'

vox1_path2info_file = f'{meta_dir}/eval/voxceleb1_all_path2info'
wr_data_frame(vox1_path2info_file, vox1_path2info.astype(str))
print(f'length of voxceleb1_all_path2info: {vox1_path2info.shape[0]}')  # 153516

# for Voxceleb-O only, selecting a subset of vox1_all
vox1_test_idx = get_filtered_idx(utt_ids, vox1_test_path2info['utt_id'])
np.save(f'{trials_dir}/index_voxceleb1_test.npy', vox1_test_idx)

print('DONE!')
