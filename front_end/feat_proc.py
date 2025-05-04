""" feature processing """

from front_end.model_misc import conv1d_unit
import numpy as np
import torch
import torchaudio
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel, WavLMModel


class FeatureExtractionLayer(nn.Module):
    def __init__(self, feat_dim=80, spec_aug=False, pretrain=False):
        super().__init__()

        if pretrain:
            self.fbank = WeightedPreTrainedFeature(emb_nn_feat_dim=feat_dim, model_name='wavlm')
        else:
            self.fbank = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600,
                window_fn=torch.hamming_window, n_mels=feat_dim)

            # self.fbank = torchaudio.transforms.Spectrogram(
            #     n_fft=512, win_length=400, hop_length=160, window_fn=torch.hamming_window)

        if spec_aug:
            self.spec_augment = SpecAugment(time_mask_len_max=5, freq_mask_len_max=8, mix=False)

        self.pretrain = pretrain
        self.spec_aug = spec_aug

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, wav_len], wav_len is the No. of sample points
        Returns:
            Tensor, [batch_size, freq_dim, seq_len], seq_len is the No. of frames
        """

        if self.pretrain:
            x_var, x_mean = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - x_mean) / torch.sqrt(x_var + 1e-7)

            return self.fbank(x)

        with torch.no_grad():
            x = self.fbank(x) + 1e-6
            seq_len = x.shape[-1] // 2 * 2
            x = x[:, :, :seq_len]

            x = x.log()
            x = x - torch.mean(x, dim=2, keepdim=True)  # x = self.instancenorm(x)
            # x = sliding_cmvn(x)

            if self.spec_aug:
                x = self.spec_augment(x)

        return x


class SpecAugment(nn.Module):
    def __init__(self, time_mask_len_max=5, freq_mask_len_max=10, mix=False, n_time_masks=1, n_freq_masks=1):
        super().__init__()
        self.time_mask_len_max = time_mask_len_max
        self.freq_mask_len_max = freq_mask_len_max
        self.mix = mix
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

    def mask_along_dim(self, x, dim):
        """
        Args:
            x: Tensor, [batch_size, freq_dim, seq_len]
            dim: int, 1 for freq dimension or 2 for temporal dimension
        Returns:
            x: Tensor, [batch_size, freq_dim, seq_len]
        """

        x_size = x.size()
        mask_len_max = self.freq_mask_len_max if dim == 1 else self.time_mask_len_max

        mask_len = torch.randint(1, mask_len_max, (1,), device=x.device).item()
        start = torch.randint(0, x_size[dim] - mask_len, (x_size[0], 1), device=x.device)
        mask_range = torch.arange(x_size[dim], device=x.device)
        mask = torch.logical_and(mask_range >= start, mask_range < (start + mask_len))
        mask = mask.unsqueeze(len(x_size) - dim)

        if self.mix:
            return ~mask * x + mask * (x + x[torch.randperm(x_size[0])]) / 2
        return x.masked_fill(mask, 0.)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, freq_dim, seq_len]
        Returns:
            x: Tensor, [batch_size, freq_dim, seq_len]
        """

        for _ in range(self.n_time_masks):
            x = self.mask_along_dim(x, dim=2)

        for _ in range(self.n_freq_masks):
            x = self.mask_along_dim(x, dim=1)

        return x


class WeightedPreTrainedFeature(nn.Module):
    def __init__(self, emb_nn_feat_dim=80, model_name='wav2vec2'):
        super().__init__()

        model_name_dict = {'wav2vec2': 'facebook/wav2vec2-large-lv60',
                           'hubert': 'facebook/hubert-large-ll60k', 'wavlm': 'microsoft/wavlm-large'}

        # Load pre-trained model
        if model_name == 'wav2vec2':
            self.model = Wav2Vec2Model.from_pretrained(model_name_dict[model_name], output_hidden_states=True)
        elif model_name == 'hubert':
            self.model = HubertModel.from_pretrained(model_name_dict[model_name], output_hidden_states=True)
        elif model_name == 'wavlm':
            self.model = WavLMModel.from_pretrained(model_name_dict[model_name], output_hidden_states=True)
        else:
            raise NotImplementedError

        # Freeze the entire model
        for param in self.model.parameters():
            param.requires_grad = False

        # Note: Actual transformer layers are hidden_states[1:] (the first is the output of feature_projection)
        self.layer_weights = nn.Parameter(torch.ones(len(self.model.encoder.layers)))  # One weight per layer

        output_hidden_size = self.model.config.output_hidden_size
        self.conv = conv1d_unit(output_hidden_size, emb_nn_feat_dim, 1, norm='bn', act=nn.GELU())  # optional

    def forward(self, x):
        """
        x: tensor, [B, wav_len]
        return: tensor, [B, D, seq_len]
        """

        # Disable gradient tracking for the frozen model
        with torch.no_grad():
            outputs = self.model(x)

        # Extract hidden states
        hidden_states = outputs.hidden_states[-len(self.model.encoder.layers):]
        hidden_states = torch.stack(hidden_states, dim=-1)

        # Aggregate transformer representations
        # weights = torch.softmax(self.layer_weights, dim=0)
        weights = self.layer_weights
        aggregated_output = torch.einsum('bldn, n -> bld', hidden_states, weights)
        x = self.conv(aggregated_output.transpose(1, 2))

        return x
