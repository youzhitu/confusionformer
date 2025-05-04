
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp
from pathlib import Path


def init_logger(logger_name='logger', log_path='log.log', device=0, n_gpus=1):
    """
    :param logger_name: str
    :param log_path: str
    :param device: int, index of the gpu where info is logged
    :param n_gpus: int
    :return logger
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO if n_gpus == 1 or device == 0 else logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(pathname)s:%(lineno)s - %(message)s')

    # file handler
    Path(Path(log_path).parent).mkdir(exist_ok=True)
    Path(log_path).touch(exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


# ------------------------------------------------------------------------------------------------
# utilities for front-end (file processing)
# ------------------------------------------------------------------------------------------------
def utt_id2spk_id(utt_id):
    """
    :param utt_id: pd Series
    :return: spk_id: pd Series
    """
    spk_id = utt_id.str.split('-', n=1, expand=True)[0]

    # For SRE, e.g., sw_4743_sw_32295_2, sw_9990-swbdc_sw_43023_1-reverb sw_9990, 3861-sre06-jaof-b-music 3861,
    # 120748_MX6_2036_A-reverb 120748, 103872_SRE08_fqxrj_B-music 103872, 120245_SRE10_eqzhh_A
    if spk_id.str.contains('SRE').any():
        spk_id_split = spk_id.str.split('_', n=2, expand=True)
        spk_id = spk_id_split[0].copy()  # sre08, sre10, mx6
        spk_ids_sw = spk_id_split[spk_id_split[0] == 'sw']  # switchboard
        spk_ids_sw = 'sw_' + spk_ids_sw[1]
        spk_id[spk_ids_sw.index] = spk_ids_sw
    spk_id.name = 'spk_id'

    return spk_id


def get_filtered_idx(data, data_subset):
    """
    Get filtered indices of a subset from complete data, i.e. data_subset = data[filtered_idx] """

    sorted_indices = np.argsort(data, kind='mergesort')  # data[sorted_indices] sorted in ascending order
    filtered_idx = sorted_indices[np.searchsorted(data[sorted_indices], data_subset)]

    return filtered_idx


def rd_data_frame(data_frame_file, x_keys='utt_id spk_id'):
    assert isinstance(x_keys, str) or isinstance(x_keys, list), 'Wrong x_keys!'

    if isinstance(x_keys, str):
        x_keys = x_keys.split(' ')

    dtypes = infer_df_dtypes(x_keys)

    return pd.read_csv(data_frame_file, sep=' ', names=x_keys, dtype=dtypes)


def wr_data_frame(df_file, df):
    """ Write dataframe df into df_file """

    df = df.astype(str)
    keys = df.keys()
    df_merge = df[keys[0]].copy()

    for i in range(1, len(keys)):
        df_merge += ' ' + df[keys[i]]

    df_merge.to_csv(df_file, index=False, header=False)


def infer_df_dtypes(names):
    int_strs = ['n_', 'num_', '_offset', 'label', '_len']
    float_strs = ['dur']
    dtypes = {}

    for name in names:
        if any([int_str in name for int_str in int_strs]):
            dtypes[name] = int
        elif any([float_str in name for float_str in float_strs]):
            dtypes[name] = float
        else:
            dtypes[name] = str

    return dtypes


# ------------------------------------------------------------------------------------------------
# utilities for backend
# ------------------------------------------------------------------------------------------------
def select_plda_trn_xvec(xvecs, spk_ids, n_samples, n_utts_per_spk=2, is_longest=False, segs_dir=None, n_procs=1):
    """ Select a subset of PLDA training data """

    assert xvecs.shape[0] == spk_ids.shape[0], \
        f'No. of xvectors {xvecs.shape[0]} is NOT equal to No. of spk_ids {spk_ids.shape[0]}!'
    assert xvecs.shape[0] == n_samples.shape[0], \
        f'No. of xvectors {xvecs.shape[0]} is NOT equal to No. of spk_ids {n_samples.shape[0]}!'

    if n_procs > 1:
        assert segs_dir is not None, 'segs_dir cannot be None!'
        seg_idx_file = f'{segs_dir}/plda_segment_idx.npy'

        index_seg = get_mp_segment_index(spk_ids, seg_idx_file, n_procs=n_procs).squeeze()
        spk_ids_seg = [spk_ids[idx] for idx in index_seg]
        n_samples_seg = [spk_ids[idx] for idx in index_seg] if is_longest else None
        n_utts_per_spk_seg = [n_utts_per_spk] * n_procs

        pool = mp.Pool(n_procs)
        if is_longest:
            subset_idx = pool.starmap(longest_selection_index,
                                      [[spk_ids_seg[i], n_samples_seg[i], n_utts_per_spk_seg[i]]
                                       for i in range(n_procs)])
        else:
            subset_idx = pool.starmap(random_selection_index,
                                      [[spk_ids_seg[i], n_utts_per_spk_seg[i]] for i in range(n_procs)])
        pool.close()
        subset_idx = np.concatenate([index_seg[i][idx] for i, idx in enumerate(subset_idx)])
    else:
        if is_longest:
            subset_idx = longest_selection_index(spk_ids, n_samples, n_utts_per_spk=n_utts_per_spk)
        else:
            subset_idx = random_selection_index(spk_ids, n_samples, n_utts_per_spk=n_utts_per_spk)

    sorted_indices = np.sort(subset_idx, kind='mergesort')

    return xvecs[sorted_indices], spk_ids[sorted_indices], n_samples[sorted_indices]


def select_plda_trn_info(meta_dir='meta/eval', plda_dir='eval/plda', source='train', n_utts_per_spk=1, n_procs=4):
    """ Select a subset of training info for PLDA training """

    plda_path2info_file = f'{meta_dir}/plda_{source}_path2info'
    print(f'Creating {plda_path2info_file}...')

    if Path(plda_path2info_file).exists():
        print(f'{plda_path2info_file} exists.')
        return

    path2info_file = f'{meta_dir}/{source}_path2info'
    path2info = rd_data_frame(path2info_file, x_keys='utt_path utt_id spk_id n_sample dur')
    spk_ids = path2info['spk_id'].values
    n_samples = path2info['n_sample'].values

    # Obtain selection index
    if n_procs > 1:
        seg_idx_file = f'{plda_dir}/plda_{source}_segment_idx.npy'  # segment indices for mp acceleration
        index_seg = get_mp_segment_index(spk_ids, seg_idx_file, n_procs=n_procs).squeeze()
        spk_ids_seg = [spk_ids[idx] for idx in index_seg]
        n_samples_seg = [n_samples[idx] for idx in index_seg]
        n_utts_per_spk_seg = [n_utts_per_spk] * n_procs

        pool = mp.Pool(n_procs)
        subset_idx = pool.starmap(longest_selection_index,
                                  [[spk_ids_seg[i], n_samples_seg[i], n_utts_per_spk_seg[i]] for i in range(n_procs)])
        pool.close()
        subset_idx = np.concatenate([index_seg[i][idx] for i, idx in enumerate(subset_idx)])
    else:
        subset_idx = longest_selection_index(spk_ids, n_samples, n_utts_per_spk=n_utts_per_spk)

    sorted_indices = np.sort(subset_idx, kind='mergesort')
    wr_data_frame(plda_path2info_file, path2info.iloc[sorted_indices].astype(str))
    print(f'{plda_path2info_file} created.')


def get_mp_segment_index(spk_ids, seg_idx_file, n_procs):
    """ Get indices of segments for multiprocessing acceleration, the No. of segments is equal to
    the No. of processors """

    if not Path(seg_idx_file).exists():
        print(f'{seg_idx_file} does not exist, creating it...')
        spk_ids_unique = np.unique(spk_ids)
        n_spk_ids_unique_per_seg = int(np.ceil(spk_ids_unique.shape[0] / n_procs))

        total_idx = np.arange(spk_ids.shape[0])
        index_seg = []

        for i in range(n_procs):
            spk_ids_unique_seg = spk_ids_unique[n_spk_ids_unique_per_seg * i: n_spk_ids_unique_per_seg * (i + 1)]
            index_seg_tmp = []

            for spk_id in spk_ids_unique_seg:
                seg_idx = total_idx[spk_ids == spk_id]
                index_seg_tmp.append(seg_idx)

            index_seg.append(np.concatenate(index_seg_tmp))

        index_seg = np.asarray([index_seg], dtype=object)
        np.save(seg_idx_file, index_seg)
    else:
        print(f'{seg_idx_file} exists, loading index from it...')
        index_seg = np.load(seg_idx_file, allow_pickle=True)

    return index_seg


def random_selection_index(spk_ids, n_samples, n_utts_per_spk=2):
    """ Randomly select n_utts_per_spk utterances for each speaker, return the indices of
    the selected utterances """

    global_idx = np.arange(spk_ids.shape[0])
    spk_ids_unique = np.unique(spk_ids)
    se_utt_idx = []

    for spk_id in spk_ids_unique:
        mask = np.bitwise_and(spk_ids == spk_id, n_samples <= 10000 * 160 + 240)
        spk_id_idx = global_idx[mask]

        if len(spk_id_idx) >= n_utts_per_spk:
            se_idx = np.random.permutation(spk_id_idx.shape[0])[:n_utts_per_spk]
            se_utt_idx.append(spk_id_idx[se_idx])

    return np.concatenate(se_utt_idx)


def longest_selection_index(spk_ids, n_samples, n_utts_per_spk=2):
    global_idx = np.arange(spk_ids.shape[0])
    spk_ids_unique = np.unique(spk_ids)
    se_utt_idx = []

    for spk_id in spk_ids_unique:
        spk_id_idx = global_idx[spk_ids == spk_id]

        if len(spk_id_idx) >= n_utts_per_spk:
            se_idx = np.argsort(n_samples[spk_id_idx], kind='mergesort')[-n_utts_per_spk:]
            se_utt_idx.append(spk_id_idx[se_idx])

    return np.concatenate(se_utt_idx)


def segment_data(data, labels, n_processors):
    """ Segment the whole data into n_processors segments with approximately equal No. of samples """

    n_samples = data.shape[0]
    sorted_indices = np.argsort(labels, kind='mergesort')  # Sort labels in ascending order
    data_sorted = data[sorted_indices]
    labels_sorted = labels[sorted_indices]
    labels_unique, n_samples_per_class = np.unique(labels_sorted, return_counts=True)

    n_samples_cumsum = np.cumsum(n_samples_per_class)
    seg_point = np.arange(1, n_processors) * n_samples / n_processors

    seg_point_idx = [0]

    for point in seg_point:
        seg_point_idx.append(n_samples_cumsum[n_samples_cumsum >= point][0])
    seg_point_idx.append(n_samples)

    data_sorted_seg, labels_seg = [], []

    for i in range(n_processors):
        seg_idx_start = seg_point_idx[i]
        seg_idx_end = seg_point_idx[i + 1]
        data_sorted_seg.append(data_sorted[seg_idx_start: seg_idx_end])
        labels_seg.append(labels_sorted[seg_idx_start: seg_idx_end])

    return data_sorted_seg, labels_seg
