"""
Prepare meta info of voxceleb, musan, and rir for spk-emb-nn training and evaluation

Before running this code, make sure that the raw wave datasets (i.e., voxceleb1, voxceleb2, musan, and rir) is under
the default path '../corpus' (you may change the dir to your preferred path).
After running this code, the meta info will be created under the default './meta' (you may also change the dir).

The vox2 meta info will be written to './meta/vox2/path2info', whose each row has keys of
['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'].
../corpus/voxceleb2/dev/aac/id00012/21Uxsk56VDQ/00001.wav id00012-21Uxsk56VDQ-00001 id00012 150528 9.41 0
../corpus/voxceleb2/dev/aac/id00015/0fijmz4vTVU/00001.wav id00015-0fijmz4vTVU-00001 id00015 113664 7.1 1
../corpus/voxceleb2/dev/aac/id00016/MBYpjluBMn4/00065.wav id00016-MBYpjluBMn4-00065 id00016 83968 5.25 2

The MUSAN music meta info will be written to './meta/musan_rir/music_path2dur',
whose each row has keys of ['utt_path', 'n_sample', 'dur'].
../corpus/musan_split/music/fma-western-art/music-fma-wa-0000-split-0.wav 64000 4.0
../corpus/musan_split/music/fma-western-art/music-fma-wa-0000-split-12.wav 64000 4.0

Similarly, MUSAN noise and speech and RIR meta will be created in 'noise_path2dur', 'speech_path2dur', and
'rir_path2dur' under './meta/musan_rir/'.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torchaudio
from utils.my_utils import rd_data_frame, wr_data_frame
import subprocess
import math
import multiprocessing as mp


corpus_dir = '../corpus'  # the default corpus dir, containing voxceleb1, voxceleb2, musan, and rir
vox1_wav_dir = f'{corpus_dir}/voxceleb1/dev/wav'
vox2_wav_dir = f'{corpus_dir}/voxceleb2/dev/aac'

musan_dir = f'{corpus_dir}/musan'
musan_split_dir = f'{corpus_dir}/musan_split'
musan_srcs = ['music', 'speech', 'noise']

rir_dir = f'{corpus_dir}/RIRS_NOISES/simulated_rirs'
rir_srcs = ['mediumroom', 'smallroom']

meta_dir = './meta'  # the default meta info dir
vox1_meta_dir = f'{meta_dir}/vox1'
Path(vox1_meta_dir).mkdir(parents=True, exist_ok=True)
vox1_path2info_file = f'{vox1_meta_dir}/path2info'

vox2_meta_dir = f'{meta_dir}/vox2'
Path(vox2_meta_dir).mkdir(parents=True, exist_ok=True)
vox2_path2info_file = f'{vox2_meta_dir}/path2info'

vox_meta_dir = f'{meta_dir}/vox'
Path(vox_meta_dir).mkdir(parents=True, exist_ok=True)
vox_path2info_file = f'{vox_meta_dir}/path2info'

musan_rir_meta_dir = f'{meta_dir}/musan_rir'
Path(musan_rir_meta_dir).mkdir(parents=True, exist_ok=True)

eval_meta_dir = f'{meta_dir}/eval'
Path(eval_meta_dir).mkdir(parents=True, exist_ok=True)


# noinspection PyUnresolvedReferences
def main():
    # ------------------------------------------------------------------------------------------------
    # Create path2info (with keys ['utt_path', 'utt_id', 'spk_id']) for voxceleb1-2
    # ------------------------------------------------------------------------------------------------
    is_make_vox = False

    if is_make_vox:
        print('Making voxceleb1 path2info...')
        vox1_utt_paths = extract_utt_paths(vox1_wav_dir, pat='*.wav')
        vox1_path2info = create_path2info(vox1_utt_paths)
        wr_data_frame(vox1_path2info_file, vox1_path2info)

        print('Making voxceleb2 path2info...')
        vox2_utt_paths = extract_utt_paths(vox2_wav_dir, pat='*.m4a')
        vox2_path2info = create_path2info(vox2_utt_paths)
        wr_data_frame(vox2_path2info_file, vox2_path2info)

        print('Merging voxceleb1-2 path2info...')
        vox_path2info = pd.concat([vox1_path2info, vox2_path2info], axis=0).sort_values(
            by=['spk_id', 'utt_path'], kind='mergesort')
        vox_path2info.index = np.arange(vox_path2info.shape[0])
        wr_data_frame(vox_path2info_file, vox_path2info)

        print('voxceleb1-2 path2info created.\n')

    # ------------------------------------------------------------------------------------------------
    # Convert .m4a to .wav for voxceleb2
    # ------------------------------------------------------------------------------------------------
    is_convert_wav = False

    if is_convert_wav:
        print('Converting *.m4a to *.wav for voxceleb2-dev...')
        vox2_utt_paths = rd_data_frame(vox2_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])['utt_path']

        # m4a_to_wav(vox2_utt_paths)  # Single-cpu processing
        n_procs = 2  # Multi-processing
        vox2_seg_paths = segment_paths(vox2_utt_paths, n_procs)
        pool = mp.Pool(n_procs)
        pool.starmap(m4a_to_wav, [[vox2_seg_paths[i]] for i in range(n_procs)])
        pool.close()

        print(f'Conversion DONE.')

    # ------------------------------------------------------------------------------------------------
    # Update the voxceleb2 path2info, changing the extension '.m4a' to '.wav'
    # ------------------------------------------------------------------------------------------------
    is_update_path2info = False

    if is_update_path2info:
        # check if all .m4a files are successfully converted into .wav, otherwise run the conversion step again
        print('Checking vox2_utt_paths extracted from .m4a and .wav...')
        vox2_path2info_m4a = rd_data_frame(vox2_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])
        vox2_utt_paths_m4a_name = vox2_path2info_m4a['utt_path'].str.rsplit('.', n=1, expand=True)[0]

        vox2_utt_paths = extract_utt_paths(vox2_wav_dir, pat='*.wav')
        vox2_utt_paths_name = vox2_utt_paths.str.rsplit('.', n=1, expand=True)[0]
        is_check_pass = vox2_utt_paths_name.equals(vox2_utt_paths_m4a_name)
        print(f'Check PASS: {is_check_pass}.')

        if is_check_pass:
            vox2_path2info = pd.concat([vox2_utt_paths, vox2_path2info_m4a[['utt_id', 'spk_id']]], axis=1)
            wr_data_frame(vox2_path2info_file, vox2_path2info)

            vox1_path2info = rd_data_frame(vox1_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])
            vox_path2info = pd.concat([vox1_path2info, vox2_path2info], axis=0).sort_values(
                by=['utt_path', 'spk_id'], kind='mergesort')
            vox_path2info.index = np.arange(vox_path2info.shape[0])
            wr_data_frame(vox_path2info_file, vox_path2info)

            print('voxceleb2 path2info updated.\n')

    # ------------------------------------------------------------------------------------------------
    # Remove .m4a from voxceleb2 to release disk space (optional)
    # ------------------------------------------------------------------------------------------------
    is_remove_m4a = False

    if is_remove_m4a:
        print('Removing .m4a...')
        vox2_utt_paths = extract_utt_paths(vox2_wav_dir, pat='*.m4a')
        m4a_parent_dirs = vox2_utt_paths.str.rsplit('/', n=1, expand=True)[0].unique()
        n_sess = m4a_parent_dirs.shape[0]

        print(f'No. of sessions of voxceleb2: {n_sess}')

        for i, m4a_parent_dir in enumerate(m4a_parent_dirs):
            subprocess.run(f'rm {m4a_parent_dir}/*.m4a', shell=True)

            if i % 2000 == 0:
                print(f'{i}/{n_sess}...')

        print('Remove DONE.')

    # ------------------------------------------------------------------------------------------------
    # Split musan (long audio) to accelerate wave loading
    # ------------------------------------------------------------------------------------------------
    is_split_musan = False

    if is_split_musan:
        print('Splitting MUSAN...')
        fs = 16000
        wav_len, wav_step = fs * 5, fs * 3
        musan_utt_paths = []

        for musan_src in musan_srcs:
            utt_paths = extract_utt_paths(f'{musan_dir}/{musan_src}', pat='*.wav')
            musan_utt_paths.append(utt_paths)
            split_dirs = utt_paths.str.rsplit('/', n=1, expand=True)[0].unique()
            split_dirs = pd.Series(split_dirs).str.replace(musan_dir, musan_split_dir, regex=False)

            for split_dir in split_dirs:
                Path(split_dir).mkdir(parents=True, exist_ok=True)

        musan_utt_paths = pd.concat(musan_utt_paths, axis=0)
        musan_utt_paths.index = np.arange(musan_utt_paths.shape[0])

        for i, utt_path in enumerate(musan_utt_paths):
            wav, fs = torchaudio.load(utt_path)
            utt_dir = str(Path(utt_path).parent).replace(musan_dir, musan_split_dir)
            utt_name = Path(utt_path).name.split('.')[0]

            for start in range(0, wav.shape[1] - wav_len, wav_step):
                split_path = f'{utt_dir}/{utt_name}-split-{start // fs}.wav'
                torchaudio.save(split_path, wav[:, start: start + wav_len], fs, encoding='PCM_S', bits_per_sample=16)

            if i % 100 == 0:
                print(f'{i}/{musan_utt_paths.shape[0]}...')

        print('MUSAN split DONE.')

    # ------------------------------------------------------------------------------------------------
    # Add duration of voxceleb1-2 to path2info and create path2dur for MUSAN and RIR
    # ------------------------------------------------------------------------------------------------
    is_calculate_dur = False

    if is_calculate_dur:
        print('Calculating the duration...')

        # For voceleb1-2
        vox_path2info = rd_data_frame(vox_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])
        n_samples, dur = get_duration(vox_path2info['utt_path'])
        # n_procs = 3
        # seg_paths = segment_paths(vox_path2info['utt_path'], n_procs)
        # pool = mp.Pool(n_procs)
        # duration = pool.starmap(get_duration, [[seg_paths[i]] for i in range(n_procs)])
        # pool.close()
        # n_samples, dur = np.concatenate(duration, axis=1)

        vox_path2info = pd.concat([vox_path2info, n_samples.astype(int), dur], axis=1)
        vox_path2info.columns = ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur']
        wr_data_frame(vox_path2info_file, vox_path2info)

        vox1_path2info = rd_data_frame(vox1_path2info_file, x_keys=['utt_path', 'utt_id', 'spk_id'])
        vox1_mask = vox_path2info['utt_path'].isin(vox1_path2info['utt_path'])
        wr_data_frame(vox1_path2info_file, vox_path2info[vox1_mask])
        wr_data_frame(vox2_path2info_file, vox_path2info[~vox1_mask])

        print('voxceleb1-2 duration adding DONE.')

        # For MUSAN
        for musan_src in musan_srcs:
            utt_paths = extract_utt_paths(f'{musan_split_dir}/{musan_src}', pat='*.wav')
            n_samples, dur = get_duration(utt_paths)
            path2dur = pd.concat([utt_paths, n_samples.astype(int), dur], axis=1)
            path2dur.columns = ['utt_path', 'n_sample', 'dur']

            path2dur_file = f'{musan_rir_meta_dir}/{musan_src}_path2dur'
            wr_data_frame(path2dur_file, path2dur)
            print(f'{musan_src} path2dur has been written into {path2dur_file}')

        print('MUSAN duration creation DONE.')

        # For RIRS_NOISES
        path2dur = []

        for rir_src in rir_srcs:
            utt_paths = extract_utt_paths(f'{rir_dir}/{rir_src}', pat='*.wav')
            n_samples, dur = get_duration(utt_paths)
            path2dur_tmp = pd.concat([utt_paths, n_samples.astype(int), dur], axis=1)
            path2dur_tmp.columns = ['utt_path', 'n_sample', 'dur']
            path2dur.append(path2dur_tmp)

        path2dur = pd.concat(path2dur, axis=0).sort_values(by=['utt_path'], kind='mergesort')
        path2dur.index = np.arange(path2dur.shape[0])

        path2dur_file = f'{musan_rir_meta_dir}/rir_path2dur'
        wr_data_frame(path2dur_file, path2dur)
        print(f'rir path2dur has been written into {path2dur_file}')

        print('RIRS_NOISES duration creation DONE.')

    # ------------------------------------------------------------------------------------------------
    # Create eval meta for voxceleb1
    # ------------------------------------------------------------------------------------------------
    is_make_eval = False

    if is_make_eval:
        print('Making voxceleb1_test path2info...')
        eval_wav_dir = f'{corpus_dir}/voxceleb1/test/wav'
        eval_paths = extract_utt_paths(eval_wav_dir, pat='*.wav')
        eval_path2info = create_path2info(eval_paths)
        assert eval_path2info.sort_values(['utt_id'], kind='mergesort').equals(eval_path2info), \
            'eval_path2info utt_id is not sorted, this will incur an error in scoring!'

        n_samples, dur = get_duration(eval_paths)
        eval_path2info = pd.concat([eval_path2info, n_samples, dur], axis=1)

        eval_path2info_file = f'{eval_meta_dir}/voxceleb1_test_path2info'
        wr_data_frame(eval_path2info_file, eval_path2info)
        print(f'voxceleb1_test path2info has been written into {eval_path2info_file}')

        # copy and append the last 2 rows as dumy entries for 4-gpu training, so that all gpus obtain
        # an equal No. of utts (ceil(4874 / 4)) for eer computation during training
        eval_path2info4 = pd.concat([eval_path2info, eval_path2info.iloc[-2:]], axis=0)
        eval_path2info4_file = f'{eval_meta_dir}/voxceleb1_test_path2info4'
        wr_data_frame(eval_path2info4_file, eval_path2info4)

        # copy and append the last 4 rows as dumy entries for 6-gpu training
        eval_path2info6 = pd.concat([eval_path2info, eval_path2info.iloc[-4:]], axis=0)
        eval_path2info6_file = f'{eval_meta_dir}/voxceleb1_test_path2info6'
        wr_data_frame(eval_path2info6_file, eval_path2info6)

        print('voxceleb1_test make DONE.')

    print('To the END.\n\n')


# noinspection PyUnresolvedReferences
def get_duration(wav_paths):
    fs = torchaudio.info(wav_paths[0]).sample_rate
    n_samples = []

    for i, wav_path in enumerate(wav_paths):
        wav_meta = torchaudio.info(wav_path)
        assert fs == wav_meta.sample_rate, 'Inconsistent sampling rate, resampling required!'
        n_samples.append(wav_meta.num_frames)

        if i % 10000 == 0:
            print(f'get_duration: {i}/{wav_paths.shape[0]}...')

    n_samples = pd.Series(n_samples, name='n_sample')
    durs = pd.Series((n_samples / fs).round(2), name='dur')

    return n_samples, durs


def m4a_to_wav(m4a_paths):
    for i, m4a_path in enumerate(m4a_paths):
        wav_path = m4a_path[:-4] + '.wav'
        subprocess.run(f'ffmpeg -v 8 -y -i {m4a_path} -f wav -acodec pcm_s16le -ar 16000 {wav_path}', shell=True)

        if i % 5000 == 0:
            print(f'm4a_to_wav: {i}/{m4a_paths.shape[0]}...')


def m4a_to_flac(m4a_paths, saved_dir='/home/tuyouzhi/corpus'):
    for i, m4a_path in enumerate(m4a_paths):
        m4a_dir = str(Path(m4a_path).parent)
        flac_dir = f'{saved_dir}/corpus/{m4a_dir.split("corpus")[-1]}'
        Path(flac_dir).mkdir(parents=True, exist_ok=True)

        flac_name = '.'.join(str(Path(m4a_path).name).split('.')[:-1])
        flac_path = f'{flac_dir}/{flac_name}.flac'
        subprocess.run(f'ffmpeg -v 8 -y -i {m4a_path} -f flac -sample_fmt s16 {flac_path}', shell=True)

        if i % 5000 == 0:
            print(f'm4a_to_flac: {i}/{m4a_paths.shape[0]}...')


def segment_paths(paths, n_procs):
    n_paths_per_seg = int(math.ceil(paths.shape[0] / n_procs))
    return [paths[i * n_paths_per_seg: (i + 1) * n_paths_per_seg] for i in range(n_procs)]


def extract_utt_paths(wav_dir, pat='*.wav'):
    utt_paths = []

    for spk_path in Path(wav_dir).iterdir():
        utt_paths.append(pd.Series(map(str, Path(f'{spk_path}').rglob(pat))))

    utt_paths = pd.concat(utt_paths, axis=0).sort_values(axis=0, kind='mergesort')
    utt_paths.index = np.arange(utt_paths.shape[0])
    utt_paths.name = 'utt_path'

    return utt_paths


def create_path2info(utt_paths, n_split=3):
    utt_paths_ = utt_paths.str.rsplit('/', n=n_split, expand=True)
    utt_ids = utt_paths_[1] + '-'

    for i in range(2, n_split):
        utt_ids += utt_paths_[i] + '-'

    utt_ids += utt_paths_[n_split].str.split('.', n=1, expand=True)[0]
    spk_ids = utt_ids.str.split('-', n=1, expand=True)[0]

    path2info = pd.concat([utt_paths, utt_ids, spk_ids], axis=1)
    path2info.columns = ['utt_path', 'utt_id', 'spk_id']

    return path2info


if __name__ == '__main__':
    main()
