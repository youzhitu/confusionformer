""" Extract xvectors or feature maps with multiple jobs """

from front_end.dataset import EvalDataset
from front_end.trainer import load_parameters
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Extractor(object):
    def __init__(self, source='voxceleb1_test', model=None, ckpt_path='model_ckpt/ckpt-20', device='cuda:0',
                 extract_dir='extract', extract_feat_map=False, n_utts_per_partition=500, n_jobs=1, job=0,
                 selected_dur=None, n_workers=2, speed_aug=False):
        """
        The extraction is split into several jobs and each job covers a number of partitions. One partition refers to
        a consecutive number of utterances; it is a subset of the whole dataset, i.e., dataset[start_idx: end_idx].
        During extraction, one partition of x-vectors are saved into one npy file and all npy files belonging to a
        specific job are placed in a folder named by the job index.
        :param extract_dir: string, directory of the extracted x-vectors
        :param n_utts_per_partition: int, No. of utterances in a partition
        :param n_jobs: int, No. of jobs
        :param job: int, job index
        """

        self.source = source
        self.model = model
        self.ckpt_path = ckpt_path
        self.device = device if device.startswith('cuda') else f'cuda:{device}'
        self.extract_dir = extract_dir
        self.extract_feat_map = extract_feat_map  # for debug only
        self.n_utts_per_partition = n_utts_per_partition
        self.n_jobs = n_jobs
        self.job = job
        self.selected_dur = selected_dur
        self.n_workers = n_workers
        self.speed_aug = speed_aug

        self._check_breakpoint()
        self._init_setup()

    def _check_breakpoint(self):
        """ Check the breakpoint of the current job so that inference can resume from the breakpoint.
        The objective is to obtain self.start_idx, self.end_idx, and self.n_utts_per_partition. """

        assert self.job in list(range(self.n_jobs)), 'Incorrect job index!'

        self.n_total_utts = EvalDataset(source=self.source).n_utterance
        self.n_utts_per_job = int(math.ceil(self.n_total_utts / self.n_jobs))
        self.n_partitions_per_job = int(math.ceil(self.n_utts_per_job / self.n_utts_per_partition))

        # Get the breakpoint of a specific job
        self.partition_dir = f'{self.extract_dir}/{self.job}'
        os.makedirs(self.partition_dir, exist_ok=True)
        existing_partitions = os.listdir(self.partition_dir)

        self.start_idx = self.n_utts_per_job * self.job
        if existing_partitions:
            _, sorted_partition_idx = sort_existing_partitions(partition_dir=self.partition_dir)
            self.start_idx += self.n_utts_per_partition * (sorted_partition_idx[-1] + 1)

        self.end_idx = min(self.n_utts_per_job * (self.job + 1), self.n_total_utts)

        print(f'Job {self.job} -- start_idx: {self.start_idx}, end_idx: {self.end_idx}, '
              f'n_utts_per_partition: {self.n_utts_per_partition}, n_utts_per_job: {self.n_utts_per_job}, '
              f'n_partitions_per_job: {self.n_partitions_per_job}')

    def _init_setup(self):
        # Create evaluation dataloader
        eval_dataset = EvalDataset(
            source=self.source, start=self.start_idx, end=self.end_idx, selected_dur=self.selected_dur,
            speed_aug=self.speed_aug)
        self.eval_dataloader = DataLoader(dataset=eval_dataset, num_workers=self.n_workers)

        # Load paras from a checkpoint
        self.model = self.model.to(self.device)
        assert os.path.exists(self.ckpt_path), f'checkpoint path {self.ckpt_path} does NOT exist.'
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.model = load_parameters(self.model, ckpt)
        print(f'Model restored from {self.ckpt_path}.')

        # Create an inference model
        if not self.extract_feat_map:
            self.infer_model = self.model.spk_model.spk_encoder

            if hasattr(self.infer_model.emb_layer.fc0, 'act'):
                self.infer_model = nn.Sequential(
                    self.infer_model.feat_layer, self.infer_model.frame_layer,
                    self.infer_model.pool_layer, self.infer_model.emb_layer.fc0.linear)
        else:
            if 'former' in self.model.name:
                self.infer_model = nn.Sequential(
                    self.model.spk_model.spk_encoder.feat_layer, self.model.spk_model.spk_encoder.frame_layer)
            else:
                self.infer_model = nn.Sequential(
                    self.model.spk_model.spk_encoder.feat_layer,
                    self.model.spk_model.spk_encoder.frame_layer.frame_level_layers[:4],
                    self.model.spk_model.spk_encoder.frame_layer.frame_level_layers[-1][0])

    def extract_emb(self, data, utt_idx, test_seg=10, fs=16000, sliding=False, step_size=2):
        """
        Args:
            data: Tensor, [batch_size, seq_len]
            utt_idx: int, utterance index
            test_seg: int, test segment in sec, a large value (e.g., >1000) indicates extracting from a full-length utt
            fs: int, sampling rate
            sliding: bool, whether extract multiple embs by sliding windows
            step_size: int, step size in sec
        Returns:
            Tensor, [batch_size, emb_dim]
        """

        data_offset = 0
        out_feat = []

        # while not out_feat or data_offset < data.size(1) - fs * test_seg:  # extract multiple embeddings for averaging
        while not out_feat:  # extract a single embedding from test_seg secs

            # print(f'Utt-{utt_idx}, Seg-{len(out_feat)}...')
            data_temp = data[:, data_offset: data_offset + fs * test_seg + 240]
            feat_temp = self.infer_model(data_temp).cpu().numpy().flatten()

            assert not np.isnan(feat_temp).any(), f'NaN in Utt-{utt_idx}, Seg-{len(out_feat)} extraction exited!\n\n\n'
            out_feat.append(feat_temp)

            if sliding:
                data_offset += int(fs * step_size)  # sliding step_size secs if extracting multiple embeddings
            else:
                break

        out_feat = np.mean(out_feat, axis=0, keepdims=True).reshape(data.size(0), -1)

        return out_feat

    def extract(self):
        # Perform inference
        print(f'Job {self.job} extraction...')
        self.infer_model.eval()

        offset = self.n_utts_per_job * self.job
        i = self.start_idx  # utterance index
        partition_idx = (i - offset) // self.n_utts_per_partition
        out_feats = []

        with torch.no_grad():
            for data in self.eval_dataloader:
                data = data.to(self.device)
                data = data.squeeze(0) if self.speed_aug else data
                out_feat = self.extract_emb(data, i, test_seg=10, fs=16000, sliding=False, step_size=2)

                if 'former' in self.model.name and self.extract_feat_map:
                    print(f'i: {i}, {out_feat.shape}')
                    np.save(f'{self.partition_dir}/partition_{partition_idx}_utt_{i}.npy', out_feat)
                else:
                    out_feats.append(out_feat)

                    if (i + 1) == self.end_idx or (i + 1 - offset) % self.n_utts_per_partition == 0:
                        # print(f'partition_idx: {partition_idx}, {np.concatenate(out_feats).shape}')
                        np.save(f'{self.partition_dir}/partition_{partition_idx}.npy', np.concatenate(out_feats))
                        partition_idx += 1
                        out_feats = []

                i += 1
                print(f'{i}/{self.end_idx}', end='  ', flush=True) if i % 1e3 == 0 else None
        print()

        # Collect embeddings
        if not self.extract_feat_map:
            n_jobs_finished = self.check_finished_jobs()

            if n_jobs_finished == self.n_jobs:
                print('All jobs have finished, collecting embeddings...')
                return self.collect_embeddings()

        return np.array([])

    def check_finished_jobs(self):
        n_jobs_finished = 0

        for job in range(self.n_jobs):
            if os.path.exists(f'{self.extract_dir}/{job}'):
                n_existing_files = len(os.listdir(f'{self.extract_dir}/{job}'))

                if job == self.n_jobs - 1:
                    n_partitions_last_job = int(math.ceil(
                        (self.n_total_utts - self.n_utts_per_job * (self.n_jobs - 1)) / self.n_utts_per_partition))

                    if n_existing_files == n_partitions_last_job:
                        n_jobs_finished += 1
                    else:
                        print(f'Job {job} is still running.')
                        break
                else:
                    if n_existing_files == self.n_partitions_per_job:
                        n_jobs_finished += 1
                    else:
                        print(f'Job {job} is still running.')
                        break
            else:
                print(f'Job {job} is still running.')
                break

        return n_jobs_finished

    def collect_embeddings(self):
        out_feats = []

        for job in range(self.n_jobs):
            partition_dir = f'{self.extract_dir}/{job}'
            sorted_existing_files, _ = sort_existing_partitions(partition_dir=partition_dir)

            for file_name in sorted_existing_files:
                out_feats.append(np.load(f'{partition_dir}/{file_name}'))

        return np.concatenate(out_feats)


def sort_existing_partitions(partition_dir='./', partition_file_extension='.npy'):
    existing_partition_files = os.listdir(partition_dir)
    assert all([file_name.endswith(partition_file_extension) for file_name in existing_partition_files]), \
        f'Incorrect partition files: files should end with {partition_file_extension}!'

    if existing_partition_files:
        partition_idx = [int(file_name.split('_')[-1].split('.')[0]) for file_name in existing_partition_files]
        sorted_idx = np.argsort(partition_idx, kind='mergesort')
        sorted_partition_idx = np.asarray(partition_idx)[sorted_idx]
        sorted_files = np.asarray(existing_partition_files)[sorted_idx]
    else:
        sorted_files, sorted_partition_idx = np.array([]), np.array([])

    return sorted_files, sorted_partition_idx
