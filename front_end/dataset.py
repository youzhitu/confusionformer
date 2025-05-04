""" Training and evaluation datasets """

import numpy as np
import random
from scipy import signal
import torchaudio
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from utils.my_utils import rd_data_frame


cwd = '.'


class TrainDataset(Dataset):
    def __init__(self, source='vox2', mode='train', min_len=200, max_len=400, repeats=180, speed_aug=False,
                 noise_reverb_aug=True):
        super().__init__()
        """
        :param mode: str, 'train' or 'val'
        :param min_len: int, minimum No. of frames of a training sample
        :param max_len: int, maximum No. of frames of a training sample
        :param repeats: int, No. of repeat of *spk2info* during one epoch, if repeats == 0, each mini-batch may have 
            samples from the same speaker; otherwise each sample within a mini-batch corresponds to a distinct speaker
        """

        self.source = source
        self.mode = mode
        self.min_len = min_len
        self.max_len = max_len
        self.repeats = repeats
        self.speed_aug = speed_aug
        self.noise_reverb_aug = noise_reverb_aug

        self.seg_len = frame2sample(self.max_len)
        self._load_wav_info()
        self._speed_augment() if self.speed_aug else None
        self._noise_reverb_augment() if self.noise_reverb_aug else None

    def _load_wav_info(self):
        path2info_file = f'{cwd}/meta/{self.source}/{self.mode}_path2info'
        self.path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur', 'label'])
        self.n_utterance = self.path2info.shape[0]
        self.n_speaker = np.unique(self.path2info['spk_id']).shape[0]
        self.path2info = self.path2info[['utt_path', 'n_sample', 'label']]

        if self.repeats:
            spk2info_file = f'{cwd}/meta/{self.source}/{self.mode}_spk2info'
            self.spk2info = rd_data_frame(spk2info_file, x_keys=['spk_id', 'n_utt', 'utt_offset', 'label'])
            assert self.n_speaker == self.spk2info.shape[0], 'Speaker mismatch between spk2info and path2info!'
            self.spk2info = self.spk2info[['n_utt', 'utt_offset']]

        print(f'No. of speakers: {self.n_speaker}, No. of utterances: {self.n_utterance}\n')

    def _speed_augment(self):
        self.speed_perturbator = SpeedPerturb()
        self.n_speed_factors = len(self.speed_perturbator.speed_factors)
        self.n_speaker *= self.n_speed_factors
        print(f'No. of speakers: {self.n_speaker}, No. of utterances: {self.n_utterance}\n')

    def _noise_reverb_augment(self):
        self.noise_reverb_augmentor = NoiseReverbAugment()

    def __len__(self):
        return self.n_speaker // self.n_speed_factors * self.repeats if self.repeats else self.n_utterance

    def __getitem__(self, idx):
        if self.repeats:
            n_utt, utt_offset = self.spk2info.iloc[idx]
            utt_offset += random.randint(0, n_utt - 1)
            utt_path, utt_len, label = self.path2info.iloc[utt_offset]
        else:
            utt_path, utt_len, label = self.path2info.iloc[idx]

        speed_factor = self.speed_perturbator.generate_speed_factor() if self.speed_aug else 1.0
        wav = load_wave(utt_path, utt_len, self.seg_len)  # round(speed_factor * self.seg_len)

        if self.speed_aug:
            wav = self.speed_perturbator.perturb(wav, speed_factor)
            label += (self.n_speaker // self.n_speed_factors) * \
                np.where(self.speed_perturbator.speed_factors == speed_factor)[0][0]

        if self.noise_reverb_aug:
            wav = self.noise_reverb_augmentor.augment(wav)

        return wav.astype(np.float32), label

    def segment_batch(self, batch_data):
        """ Segment samples in the mini-batch, used as the collate_fn in DataLoader """

        if self.min_len == self.max_len:
            data, label = list(zip(*batch_data))

            return torch.tensor(np.stack(data)), torch.tensor(label)

        seg_len = random.randint(frame2sample(self.min_len), self.seg_len)
        seg_batch_data, seg_batch_label = [], []

        for data, label in batch_data:
            seg_offset = random.randint(0, self.seg_len - seg_len)
            seg_batch_data.append(data[seg_offset: seg_offset + seg_len])
            seg_batch_label.append(label)

        return torch.from_numpy(np.asarray(seg_batch_data)), torch.from_numpy(np.asarray(seg_batch_label))


class TrainBatchSampler(Sampler):
    def __init__(self, data_source, batch_size=128, seed=12345):
        super().__init__(data_source)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        self.n_utterance = data_source.n_utterance
        self.n_speaker = data_source.n_speaker
        self.repeats = data_source.repeats
        self.n_speed_factors = data_source.n_speed_factors
        self._compute_num_batches()

    def _compute_num_batches(self):
        n_paths = self.n_speaker // self.n_speed_factors if self.repeats else self.n_utterance
        self.batches = n_paths // self.batch_size

        if dist.is_initialized():
            self.batches //= dist.get_world_size()

        self.batches = self.batches * self.repeats if self.repeats else self.batches
        # print(f'_compute_num_batches(): rank: {dist.get_rank()}, self.batches: {self.batches}')

    def __len__(self):
        return self.batches

    def __iter__(self):
        if self.repeats:
            per_rank_index = []

            for i in range(self.repeats):
                per_rank_index_tmp = self.sample_per_rank_index(self.n_speaker // self.n_speed_factors, iteration=i)
                per_rank_index.append(np.stack([per_rank_index_tmp[i * self.batch_size: (i + 1) * self.batch_size]
                                                for i in range(self.batches // self.repeats)], axis=0))
            per_rank_index = np.concatenate(per_rank_index, axis=0)
        else:
            per_rank_index = self.sample_per_rank_index(self.n_utterance)
            per_rank_index = np.stack([per_rank_index[i * self.batch_size: (i + 1) * self.batch_size]
                                       for i in range(self.batches)], axis=0)

        # print(f'__iter__(): rank: {dist.get_rank()}, self.seed: {self.seed}, '
        #       f'self.epoch: {self.epoch}, per_rank_index: {per_rank_index}-{per_rank_index.shape}')

        return iter(per_rank_index)

    def sample_per_rank_index(self, n_paths, iteration=0):
        rng = np.random.default_rng(seed=self.seed + self.epoch + iteration)
        per_rank_index = rng.permutation(n_paths)

        if dist.is_initialized():
            rank = dist.get_rank()
            n_paths //= dist.get_world_size()
            per_rank_index = per_rank_index[rank * n_paths: (rank + 1) * n_paths]

        return per_rank_index

    def set_epoch(self, epoch):
        self.epoch = epoch * self.repeats if self.repeats else epoch


class EvalDataset(Dataset):
    def __init__(self, source='voxceleb1_test', start=None, end=None, selected_dur=None, speed_aug=False,
                 noise_reverb_aug=False, n_gpus=1):
        super().__init__()
        """
        Dataset of an evaluation data partition, i.e., partition = whole_data[start: end]
        :param start: start index of the data partition
        :param end: end index of the partition
        :param selected_dur: duration of a randomly selected segment in frames, 'None' means using a full-length utt
        """

        self.source = source
        self.start = start
        self.end = end
        self.selected_dur = selected_dur
        self.speed_aug = speed_aug
        self.noise_reverb_aug = noise_reverb_aug
        self.n_gpus = n_gpus

        self._load_wav_info()
        self._speed_augment() if self.speed_aug else None
        self._noise_reverb_augment() if self.noise_reverb_aug else None

    def _load_wav_info(self):
        path2info_file = f'{cwd}/meta/eval/{self.source}_path2info{self.n_gpus}' if self.n_gpus == 4 else \
            f'{cwd}/meta/eval/{self.source}_path2info'
        self.path2info = rd_data_frame(path2info_file, ['utt_path', 'utt_id', 'spk_id', 'n_sample', 'dur'])
        self.path2info = self.path2info[['utt_path', 'n_sample']]
        self.n_utterance = self.path2info.shape[0]

        # update start and end index
        self.start = 0 if self.start is None else self.start
        self.end = self.path2info.shape[0] if self.end is None else self.end
        print(f'length of partition in {self.source}: {self.end - self.start}\n')

    def _speed_augment(self):
        self.speed_perturbator = SpeedPerturb()

    def _noise_reverb_augment(self):
        self.noise_reverb_augmentor = NoiseReverbAugment()

    def __len__(self):
        return max(self.end - self.start, 0)

    def __getitem__(self, idx):
        idx += self.start
        path, n_sample = self.path2info.iloc[idx].values
        seg_len = frame2sample(self.selected_dur) if self.selected_dur is not None else n_sample
        wav = load_wave(path, n_sample, seg_len)

        if self.speed_aug:
            wavs = [wav]

            for speed_factor in self.speed_perturbator.speed_factors[1:]:
                # wav = load_wave(path, n_sample, seg_len)
                wav = self.speed_perturbator.perturb(wav, speed_factor)
                wavs.append(wav)
            wav = np.asarray(wavs)

        if self.noise_reverb_aug:
            if wav.ndim == 2:
                wav = np.asarray([self.noise_reverb_augmentor.augment(wav_) for wav_ in wav])
            else:
                wav = self.noise_reverb_augmentor.augment(wav)

        return wav.astype(np.float32)


class SpeedPerturb(object):
    """ Apply speed perturbation """
    def __init__(self, fs=16000):
        self.fs = fs
        self.speed_factors = [0.9, 1.0, 1.1]
        self._make_one_first()
        # self.tfm = sox.Transformer()

    def _make_one_first(self):
        self.speed_factors.remove(1.0)
        self.speed_factors = np.hstack([1.0, np.sort(self.speed_factors)])

    def generate_speed_factor(self):
        return np.random.choice(self.speed_factors, 1).item()

    def perturb(self, wav, speed_factor=1.0):
        if speed_factor == 1.0:
            return wav

        wav_len = wav.shape[0]

        speed_perturb = torchaudio.transforms.SpeedPerturbation(self.fs, [speed_factor])
        wav_sp = speed_perturb(torch.tensor(wav, dtype=torch.float32))[0].numpy()  # ceil()

        if wav_sp.shape[0] >= wav_len:
            wav_sp = wav_sp[:wav_len]
        else:
            wav_sp = np.pad(wav_sp, (0, wav_len - wav_sp.shape[0]))

        return wav_sp


class NoiseReverbAugment(object):
    """ Add noise (MUSAN) and reverberation (RIR) """
    def __init__(self):
        self.aug_srcs = ['speech', 'music', 'noise', 'rir']
        self.musan_snrs = {'speech': [13, 20], 'noise': [0, 15], 'music': [5, 15]}
        self.n_musan_augs = {'speech': [3, 7], 'noise': [1, 1], 'music': [1, 1]}

        self._load_aug_info()

    def _load_aug_info(self):
        self.aug_path2info = {}

        for src in self.aug_srcs:
            path2info_file = f'{cwd}/meta/musan_rir/{src}_path2dur'
            path2info = rd_data_frame(path2info_file, ['utt_path', 'n_sample', 'dur'])
            self.aug_path2info[src] = path2info[['utt_path', 'n_sample']]

    def augment(self, wav):
        """
        :param wav: np tensor, [segment_length,]
        :return: wav: np tensor, [segment_length,]
        """

        # aug = random.randint(0, 1)
        aug = random.uniform(0., 1.)

        if aug > 0.6:
            aug_type = random.randint(1, 4)

            if aug_type == 1:
                wav = self.reverberate(wav)
            elif aug_type == 2:
                wav += self.make_musan(wav, 'music')
            elif aug_type == 3:
                wav += self.make_musan(wav, 'noise')
            elif aug_type == 4:
                wav += self.make_musan(wav, 'speech')
            # elif aug_type == 5:
            #     wav += self.make_musan(wav, 'speech') + self.make_musan(wav, 'music')

        return wav

    def make_musan(self, wav, musan_src):
        """
        :param wav: np tensor, [segment_length,]
        :param musan_src: str, 'music', 'speech', or 'noise'
        :return:
            noisy wav: np tensor, [segment_length,]
        """

        wav_len = wav.shape[0]
        n_augs = random.randint(self.n_musan_augs[musan_src][0], self.n_musan_augs[musan_src][1])

        path2info = self.aug_path2info[musan_src].sample(n_augs)
        noise = np.stack([load_wave(path, n_sample, wav_len) for path, n_sample in path2info.values], axis=0)
        snrs = np.random.uniform(self.musan_snrs[musan_src][0], self.musan_snrs[musan_src][1], n_augs)

        # p_wav = np.sum(wav ** 2) + 1e-5
        p_wav = 10 * np.log10(np.sum(wav ** 2) + 1e-5)
        # p_noise = np.sum(noise ** 2, axis=1) + 1e-5
        # noise *= (np.sqrt(p_wav / p_noise) * 10 ** (-snrs / 20))[:, None]
        p_noise = 10 * np.log10(np.sum(noise ** 2, axis=1) + 1e-5)
        noise *= np.sqrt(10 ** ((p_wav - p_noise - snrs) / 10))[:, None]

        return np.sum(noise, axis=0)  # + wav

    def reverberate(self, wav):
        """
        :param wav: np tensor, [segment_length,]
        :return:
            reverberated wav: np tensor, [segment_length,]
        """

        wav_len = wav.shape[0]
        path, n_sample = self.aug_path2info['rir'].sample(1).values[0]
        rir = load_wave(path, n_sample, n_sample)
        rir = rir / np.sqrt(np.sum(rir ** 2) + 1e-5)

        return signal.convolve(wav, rir, mode='full')[:wav_len]


# noinspection PyUnresolvedReferences
def load_wave(wav_file, wav_length, segment_length):
    """
    :param wav_file: str
    :param wav_length: int, length of a wave to be sampled from
    :param segment_length: int, length to be sampled
    :return:
        wav: np tensor, [segment_length,]
    """

    if wav_length >= segment_length:
        start = random.randint(0, wav_length - segment_length)
        # wav, _ = soundfile.read(wav_file, start=start, frames=segment_length, dtype='float32')
        wav = torchaudio.load(wav_file, frame_offset=start, num_frames=segment_length)[0][0].numpy()
    else:
        # wav, _ = soundfile.read(wav_file, dtype='float32')
        wav = torchaudio.load(wav_file)[0][0].numpy()
        wav = np.pad(wav, (0, segment_length - wav_length), 'wrap')

    return wav


def frame2sample(n_frames):
    """ 1 frame == 10ms@16kHz """
    return n_frames * 160 + 240
