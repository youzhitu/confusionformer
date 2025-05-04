""" Prepare meta info for cnceleb1-2 for spk-emb-nn training and evaluation """

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchaudio
from utils.my_utils import rd_data_frame, wr_data_frame
from utils.prep_vox_meta import extract_utt_paths


dur_min = 4.
fs = 16000

corpus_dir = '../corpus'

cn1_wav_dir = f'{corpus_dir}/CN-Celeb_flac/data'
cn2_wav_dir = f'{corpus_dir}/CN-Celeb2_flac/data'
cn1_eval_dir = f'{corpus_dir}/CN-Celeb_flac/eval'

meta_dir = './meta'
cn1_meta_dir = f'{meta_dir}/cnceleb1'
Path(cn1_meta_dir).mkdir(parents=True, exist_ok=True)
cn1_path2info_file = f'{cn1_meta_dir}/path2info'

cn2_meta_dir = f'{meta_dir}/cnceleb2'
Path(cn2_meta_dir).mkdir(parents=True, exist_ok=True)
cn2_path2info_file = f'{cn2_meta_dir}/path2info'

cn_meta_dir = f'{meta_dir}/cnceleb'
Path(cn_meta_dir).mkdir(parents=True, exist_ok=True)
cn_path2info_file = f'{cn_meta_dir}/path2info'

eval_meta_dir = f'{meta_dir}/eval'
cn_concat_wav_dir = f'{corpus_dir}/CN-Celeb_flac/data_concat'
Path(cn_concat_wav_dir).mkdir(parents=True, exist_ok=True)


def main():
    # ------------------------------------------------------------------------------------------------
    # Create cnceleb1-2 path2info (with keys ['utt_path', 'utt_id', 'spk_id'])
    # ------------------------------------------------------------------------------------------------
    is_make_cn = False

    if is_make_cn:
        print('Making cnceleb1 path2info...')
        cn1_utt_paths = extract_utt_paths(cn1_wav_dir, pat='*.flac')
        cn1_path2info = create_path2info(cn1_utt_paths)
        wr_data_frame(cn1_path2info_file, cn1_path2info.astype(str))
        print(f'No. of utts of cnceleb1: {cn1_path2info.shape[0]}\n')  # 126532

        print('Making cnceleb2 path2info...')
        cn2_utt_paths = extract_utt_paths(cn2_wav_dir, pat='*.flac')
        cn2_path2info = create_path2info(cn2_utt_paths)
        wr_data_frame(cn2_path2info_file, cn2_path2info.astype(str))
        print(f'No. of utts of cnceleb2: {cn2_path2info.shape[0]}\n')  # 524787

        print('Merging cnceleb1-2 path2info...')
        cn_path2info = pd.concat(
            [cn1_path2info, cn2_path2info], axis=0).sort_values(by=['spk_id', 'utt_path'], kind='mergesort')
        cn_path2info.index = np.arange(cn_path2info.shape[0])
        wr_data_frame(cn_path2info_file, cn_path2info.astype(str))
        print(f'No. of utts of cnceleb: {cn_path2info.shape[0]}\n')  # 651319

        print('cnceleb1-2 path2info created.\n')

    # ------------------------------------------------------------------------------------------------
    # Calculate cnceleb1-2 duration (with keys ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
    # ------------------------------------------------------------------------------------------------
    is_calculate_dur = False

    if is_calculate_dur:
        print('Calculating the duration...')

        cn_path2info = rd_data_frame(cn_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])
        n_samples, dur, _ = get_duration(cn_path2info['utt_path'])

        cn_path2info = pd.concat([cn_path2info, n_samples.astype(int), dur], axis=1)
        cn_path2info.columns = ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']
        wr_data_frame(cn_path2info_file, cn_path2info)

        cn1_path2info = rd_data_frame(cn1_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])
        cn1_mask = cn_path2info['utt_path'].isin(cn1_path2info['utt_path'])
        wr_data_frame(cn1_path2info_file, cn_path2info[cn1_mask])
        wr_data_frame(cn2_path2info_file, cn_path2info[~cn1_mask])

        print('cnceleb1-2 duration creation DONE.')

    # ------------------------------------------------------------------------------------------------
    # update cn_path2info, concatenating short utts into utts longer than 'dur_min' secs
    # ------------------------------------------------------------------------------------------------
    is_update_trn = False

    if is_update_trn:
        print('updating cn_path2info...')

        cn_ori_path2info = rd_data_frame(
            cn_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        cn_ori_path2info = cn_ori_path2info.sort_values(by=['spk_id', 'utt_id'], kind='mergesort')
        cn_ori_path2info.index = np.arange(cn_ori_path2info.shape[0])
        spk_ids_unique, utts_per_spk = np.unique(cn_ori_path2info['spk_id'], return_counts=True)  # 2306
        print(f'cn_ori_path2info -- No. of utts: {cn_ori_path2info.shape[0]}, No. of spks: {spk_ids_unique.shape[0]}')

        cn_path2info = cn_ori_path2info[cn_ori_path2info['dur'] >= dur_min]
        cn_short_path2info = cn_ori_path2info[cn_ori_path2info['dur'] < dur_min]
        cn_short_path2info = cn_short_path2info[cn_short_path2info['dur'] >= .5]  # choose utts in [0.5, 5]
        cn_short_path2info.index = np.arange(cn_short_path2info.shape[0])  # 142973

        spk_ids_unique, utts_per_spk = np.unique(cn_short_path2info['spk_id'], return_counts=True)  # 2306
        concat_utt_paths, concat_utt_ids, concat_spk_ids, concat_n_samples, concat_durs = [], [], [], [], []

        for spk_id in spk_ids_unique:
            path2info = cn_short_path2info[cn_short_path2info['spk_id'] == spk_id]
            utt_ids = path2info['utt_id'].str.rsplit('-', n=2, expand=True)[0]
            utt_ids_unique = np.unique(utt_ids)

            for utt_id in utt_ids_unique:
                path2info_ = path2info[utt_ids == utt_id]
                wav_concat = []

                for pth in path2info_['utt_path']:
                    wav, fs_ = torchaudio.load(pth)
                    if fs_ != fs:
                        print(f'{pth}: sampling rate {fs_} is NOT 16KHz， Skip！')
                        continue
                    wav_concat.append(wav)
                wav_concat = torch.concat(wav_concat, dim=1)

                concat_n_sample = wav_concat.shape[1]
                concat_dur = concat_n_sample / fs
                n_splits = int(concat_dur / dur_min)

                for i in range(n_splits):
                    concat_dir = f'{cn_concat_wav_dir}/{spk_id}'
                    Path(concat_dir).mkdir(parents=True, exist_ok=True)
                    concat_pth = f'{concat_dir}/{utt_id.split("-")[1]}-concat-{i}.flac'
                    concat_utt_id = f'{utt_id}-concat-{i}'

                    concat_utt_paths.append(concat_pth)
                    concat_utt_ids.append(concat_utt_id)
                    concat_spk_ids.append(spk_id)

                    st = int(i * fs * dur_min)
                    end = wav_concat.shape[1] if i == n_splits - 1 else int((i + 1) * fs * dur_min)
                    wav_split = wav_concat[:, st: end]
                    torchaudio.save(concat_pth, wav_split, fs, format='flac', bits_per_sample=16)

                    split_n_sample = end - st
                    concat_n_samples.append(split_n_sample)
                    concat_durs.append(split_n_sample / fs)

        concat_path2info = pd.concat([
            pd.Series(concat_utt_paths), pd.Series(concat_utt_ids), pd.Series(concat_spk_ids),
            pd.Series(concat_n_samples), pd.Series(concat_durs).round(2)], axis=1)
        print(f'cnceleb-concat -- No. of utts: {concat_path2info.shape[0]}, No. of spks: {spk_ids_unique.shape[0]}')
        wr_data_frame(f'{cn_meta_dir}/path2info_concat', concat_path2info.astype(str))

        concat_path2info = rd_data_frame(
            f'{cn_meta_dir}/path2info_concat', ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        cn_path2info = pd.concat([cn_path2info, concat_path2info])
        cn_path2info = cn_path2info.sort_values(by=['spk_id', 'utt_id'], kind='mergesort')
        cn_path2info.index = np.arange(cn_path2info.shape[0])
        cn_path2info.columns = ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']

        unique_spk_ids, utts_per_spk = np.unique(cn_path2info['spk_id'], return_counts=True)
        labels = pd.Series(np.repeat(np.arange(utts_per_spk.shape[0]), utts_per_spk), name='label')
        cn_path2info = pd.concat([cn_path2info, labels], axis=1)
        wr_data_frame(cn_path2info_file, cn_path2info.astype(str))
        print(f'cnceleb -- No. of utts: {cn_path2info.shape[0]}, No. of spks: {unique_spk_ids.shape[0]}')

        print('cnceleb update DONE.')

    # ------------------------------------------------------------------------------------------------
    # check cn_path2info
    # ------------------------------------------------------------------------------------------------
    is_check_cn = False

    if is_check_cn:
        print('checking cn_path2info...')

        cn_path2info = rd_data_frame(cn_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])
        unique_spk_ids, utts_per_spk = np.unique(cn_path2info['spk_id'], return_counts=True)
        print(f'cnceleb -- No. of utts: {cn_path2info.shape[0]}, No. of spks: {unique_spk_ids.shape[0]}')

        wav, fs_ = None, 0
        idx_keep = []

        for i, pth in enumerate(cn_path2info['utt_path']):
            if i % 5000 == 0:
                print(f'{i}/{cn_path2info.shape[0]}...')

            try:
                wav, fs_ = torchaudio.load(pth)
            except (Exception,):
                print(f'{i}-{pth}: audio loading error!')
                continue

            if fs_ != fs:
                print(f'{i}-{pth}: sampling rate is NOT 16Khz!')
                continue

            n_sample = cn_path2info['n_sample'][i]
            if wav.shape[1] != n_sample:
                print(f'{i}-{pth}: n_sample mismatch, {wav.shape[1]} != {n_sample}!')
                cn_path2info['n_sample'][i] = wav.shape[1]

            idx_keep.append(i)

        idx_keep = np.asarray(idx_keep)
        cn_path2info = cn_path2info.iloc[idx_keep]
        cn_path2info = cn_path2info[cn_path2info['dur'] >= dur_min]
        cn_path2info.index = np.arange(cn_path2info.shape[0])
        wr_data_frame(cn_path2info_file, cn_path2info.astype(str))

        unique_spk_ids, utts_per_spk = np.unique(cn_path2info['spk_id'], return_counts=True)
        print(f'cnceleb after check-- No. of utts: {cn_path2info.shape[0]}, No. of spks: {unique_spk_ids.shape[0]}')

        print('cnceleb check DONE.')

    # ------------------------------------------------------------------------------------------------
    # make cn train
    # ------------------------------------------------------------------------------------------------
    is_make_trn = False

    if is_make_trn:
        print('making cn/cn1/cn2 train...')

        path2info = rd_data_frame(cn_path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])
        path2info = path2info[['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']]

        cn1_enroll = rd_data_frame(f'{cn1_eval_dir}/lists/enroll.lst', ['enroll', 'path'])
        cn1_test = rd_data_frame(f'{cn1_eval_dir}/lists/test.lst', ['path'])
        cn1_eval = pd.concat([cn1_enroll['path'], cn1_test['path']])
        cn1_eval_spk_ids = np.unique(cn1_eval.str.split('/', n=1, expand=True)[1].str.split('-', n=1, expand=True)[0])
        path2info = path2info[~path2info['spk_id'].isin(cn1_eval_spk_ids)]  # ignore cn1-eval
        path2info = path2info.sort_values(by=['spk_id', 'utt_id'], kind='mergesort')
        path2info.index = np.arange(path2info.shape[0])

        unique_spk_ids, utts_per_spk = np.unique(path2info['spk_id'], return_counts=True)
        labels = pd.Series(np.repeat(np.arange(utts_per_spk.shape[0]), utts_per_spk), name='label')
        cn_path2info = pd.concat([path2info, labels], axis=1)
        wr_data_frame(cn_path2info_file, cn_path2info.astype(str))
        print(f'cnceleb -- No. of utts: {path2info.shape[0]}, No. of spks: {unique_spk_ids.shape[0]}')

        cn2_path2info = path2info[path2info['spk_id'].str.startswith('id1')]
        cn2_path2info.index = np.arange(cn2_path2info.shape[0])
        unique_spk_ids, utts_per_spk = np.unique(cn2_path2info['spk_id'], return_counts=True)
        labels = pd.Series(np.repeat(np.arange(utts_per_spk.shape[0]), utts_per_spk), name='label')
        cn2_path2info = pd.concat([cn2_path2info, labels], axis=1)
        wr_data_frame(cn2_path2info_file, cn2_path2info.astype(str))
        print(f'cnceleb2 -- No. of utts: {cn2_path2info.shape[0]}, No. of spks: {unique_spk_ids.shape[0]}')

        cn1_path2info = path2info[path2info['spk_id'].str.startswith('id0')]
        cn1_path2info.index = np.arange(cn1_path2info.shape[0])
        unique_spk_ids, utts_per_spk = np.unique(cn1_path2info['spk_id'], return_counts=True)
        labels = pd.Series(np.repeat(np.arange(utts_per_spk.shape[0]), utts_per_spk), name='label')
        cn1_path2info = pd.concat([cn1_path2info, labels], axis=1)
        wr_data_frame(cn1_path2info_file, cn1_path2info.astype(str))
        print(f'cnceleb1 -- No. of utts: {cn1_path2info.shape[0]}, No. of spks: {unique_spk_ids.shape[0]}')

    # ------------------------------------------------------------------------------------------------
    # Create eval meta (with keys ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
    # ------------------------------------------------------------------------------------------------
    is_make_eval = False

    if is_make_eval:
        eval_wav_dir = f'{cn1_eval_dir}'
        eval_paths = extract_utt_paths(eval_wav_dir, pat='*.flac')
        enroll_mask = eval_paths.str.contains('enroll')

        print('Making cnceleb1_enroll path2info...')
        enroll_paths = eval_paths[enroll_mask]
        enroll_paths.index = np.arange(enroll_paths.shape[0])
        utt_ids = enroll_paths.str.rsplit('/', n=1, expand=True)[1].str.split('-', n=1, expand=True)[0]
        n_samples, dur, _ = get_duration(enroll_paths)
        enroll_path2info = pd.concat([enroll_paths, utt_ids, utt_ids, n_samples, dur], axis=1)
        enroll_path2info.columns = ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']
        enroll_path2info = enroll_path2info.sort_values(['utt_id'], kind='mergesort')
        enroll_path2info.index = np.arange(enroll_path2info.shape[0])
        enroll_path2info_file = f'{eval_meta_dir}/cnceleb1_enroll_path2info'
        wr_data_frame(enroll_path2info_file, enroll_path2info)
        print(f'No. of utts of cnceleb1_enroll: {enroll_path2info.shape[0]}\n')  # 196

        print('Making cnceleb1_test path2info...')
        test_paths = eval_paths[~enroll_mask]
        test_paths.index = np.arange(test_paths.shape[0])
        utt_ids = test_paths.str.rsplit('/', n=1, expand=True)[1].str.split('.', n=1, expand=True)[0]
        spk_ids = utt_ids.str.split('-', n=1, expand=True)[0]
        n_samples, dur, _ = get_duration(test_paths)
        test_path2info = pd.concat([test_paths, utt_ids, spk_ids, n_samples, dur], axis=1)
        test_path2info.columns = ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']
        test_path2info = test_path2info.sort_values(['utt_id'], kind='mergesort')
        test_path2info.index = np.arange(test_path2info.shape[0])
        test_path2info_file = f'{eval_meta_dir}/cnceleb1_test_path2info'
        wr_data_frame(test_path2info_file, test_path2info)
        print(f'No. of utts of cnceleb1_test: {test_path2info.shape[0]}\n')  # 17777

        print('cnceleb1 eval make DONE.')

    print('To the END.\n\n')


# noinspection PyUnresolvedReferences
def get_duration(wav_paths):
    fs = torchaudio.info(wav_paths[0]).sample_rate
    n_samples = []
    idx = []

    for i, wav_path in enumerate(wav_paths):
        if i % 5000 == 0:
            print(f'get_duration: {i}/{wav_paths.shape[0]}...')

        try:
            wav_meta = torchaudio.info(wav_path)
            assert wav_meta.sample_rate == fs, f'{i}-{wav_path}: sampling rate {wav_meta.sample_rate} != {fs}!'
            n_samples.append(wav_meta.num_frames)
            idx.append(i)
        except (Exception,):
            print(f'{i}-{wav_path}: Skip!')
            continue

    n_samples = pd.Series(n_samples, name='n_sample')
    durs = pd.Series((n_samples / fs).round(2), name='dur')
    idx = np.asarray(idx)

    return n_samples, durs, idx


def create_path2info(utt_paths, n_split=2):
    utt_paths_ = utt_paths.str.rsplit('/', n=n_split, expand=True)
    spk_ids = utt_paths_[1]
    utt_ids = spk_ids + '-' + utt_paths_[n_split].str.split('.', n=1, expand=True)[0]

    path2info = pd.concat([utt_paths, utt_ids, spk_ids], axis=1)
    path2info.columns = ['utt_path', 'utt_id', 'spk_id']

    return path2info


if __name__ == '__main__':
    main()
