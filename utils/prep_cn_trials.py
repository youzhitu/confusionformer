""" Prepare cnceleb1 trials """

import numpy as np
import pandas as pd
from utils.my_utils import rd_data_frame, wr_data_frame


trials_list_dir = '../corpus/voxceleb1/CN-Celeb_flac/eval/lists'
trials_dir = './eval/trials/cnceleb1'
meta_dir = './meta'


# Convert trials list, the trials list should be sorted by 'enroll_id'
trials_file = f'{trials_list_dir}/trials.lst'
trials = rd_data_frame(trials_file, x_keys=['enroll_id', 'test_id', 'key'])

trials_keys = trials['key']
trials_keys = trials_keys.str.replace('0', 'nontarget', regex=False)
trials_keys = trials_keys.str.replace('1', 'target', regex=False)

trials_enroll_ids = trials['enroll_id']
trials_enroll_ids = trials_enroll_ids.str.split('-', n=1, expand=True)[0]

trials_test_ids = trials['test_id']
trials_test_ids = trials_test_ids.str.split('/', n=1, expand=True)[1].str.split('.', n=1, expand=True)[0]

trials = pd.concat([trials_enroll_ids, trials_test_ids, trials_keys], axis=1)
trials.columns = ['enroll_id', 'test_id', 'key']
trials = trials.sort_values(by=['enroll_id'], kind='mergesort')  # sort
trials.index = np.arange(trials.shape[0])
trials_file = f'{trials_dir}/trials'
wr_data_frame(trials_file, trials.astype(str))

enroll_spks = trials_enroll_ids.unique()
test_spks = trials_test_ids.str.split('-', n=1, expand=True)[0].unique()
print(f'No. of enroll spks: {len(enroll_spks)}, No. of test spks: {len(test_spks)}')  # 196/200

enroll_utts = trials_enroll_ids.unique()
test_utts = trials_test_ids.unique()
print(f'No. of enroll utts: {len(enroll_utts)}, No. of test utts: {len(test_utts)}')  # 196/17777

unique_utts = np.unique(np.concatenate([enroll_utts, test_utts]))
print(f'No. of unique eval utts: {len(unique_utts)}\n')

print('DONE!')
