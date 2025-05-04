""" TDNN-based speaker embedding network """

from collections import OrderedDict
import torch.nn as nn
from front_end.feat_proc import FeatureExtractionLayer
from front_end.model_misc import conv1d_unit, mlp, LayerNorm
from front_end.pooling import StatsPoolingLayer, AttentivePoolingLayer, ChannelContextStatsPoolingLayer
from front_end.classification import SoftmaxLayer, AMSoftmaxLayer


class TDNN(nn.Module):
    def __init__(self, feat_dim=80, filters='512-512-512-512-1536', kernel_sizes='5-3-3-1-1', dilations='1-2-3-1-1',
                 pooling='stats', emb_dims='512-512', n_class=5994, output_act='softmax', spec_aug=False, name='tdnn',
                 logger=None):
        super().__init__()

        self.feat_dim = feat_dim
        self.filters = [int(filter_) for filter_ in filters.split('-')]
        self.kernel_sizes = [int(kernel_size_) for kernel_size_ in kernel_sizes.split('-')]
        self.dilations = [int(dilation_) for dilation_ in dilations.split('-')]

        assert len(self.filters) == len(self.kernel_sizes) == len(self.dilations), \
            'Unequal length of filters, kernel_sizes, or dilation rates!'

        self.pooling = pooling
        self.emb_dims = [int(emb_dim) for emb_dim in emb_dims.split('-')]
        self.n_class = n_class
        self.output_act = output_act
        self.spec_aug = spec_aug
        self.name = name
        self.logger = logger

        # Create layers
        self.spk_model = nn.Sequential(OrderedDict(
            [('spk_encoder', self.create_spk_encoder()),
             ('spk_cls_head', self.create_spk_cls_head(input_dim=self.emb_dims[-1]))]))

    def create_spk_encoder(self):
        spk_enc = OrderedDict()
        spk_enc['feat_layer'] = FeatureExtractionLayer(feat_dim=self.feat_dim, spec_aug=self.spec_aug)
        spk_enc['frame_layer'] = self.create_frame_level_layers(input_dim=self.feat_dim)
        spk_enc['pool_layer'] = self.create_pooling_layer(input_dim=spk_enc['frame_layer'].output_chnls, norm='bn')
        spk_enc['emb_layer'] = self.create_emb_layers(input_dim=spk_enc['pool_layer'][0].output_dim, norm='bn')

        return nn.Sequential(spk_enc)

    def create_frame_level_layers(self, input_dim=80):
        return TDNNFrameLevelLayers(
            input_dim=input_dim, filters=self.filters, kernel_sizes=self.kernel_sizes, dilations=self.dilations)

    def create_pooling_layer(self, input_dim=1500, norm='bn'):
        pooling_layers = OrderedDict()

        if self.pooling.startswith('attention'):
            hidden_dim, heads = list(map(int, self.pooling.split('-')[-2:]))  # attention-128-1
            pooling_layer = AttentivePoolingLayer(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads)
        elif self.pooling.startswith('ctdstats'):
            hidden_dim, context = list(map(int, self.pooling.split('-')[-2:]))  # ctdstats-128-1
            pooling_layer = ChannelContextStatsPoolingLayer(
                input_dim=input_dim, hidden_dim=hidden_dim, context_dependent=bool(context), norm=norm)
        elif self.pooling == 'stats':
            pooling_layer = StatsPoolingLayer(input_dim=input_dim, use_mean=True)
        else:
            raise NotImplementedError

        pooling_layers['pooling'] = pooling_layer

        if self.name != 'tdnn':
            if norm == 'bn':
                pooling_layers['norm'] = nn.BatchNorm1d(pooling_layer.output_dim)
            elif norm == 'ln':
                pooling_layers['norm'] = LayerNorm(pooling_layer.output_dim)
        else:
            pooling_layers['norm'] = nn.Identity()

        return nn.Sequential(pooling_layers)

    def create_emb_layers(self, input_dim=3000, norm='bn', act=nn.ReLU()):
        norms, acts = [], []

        if len(self.emb_dims) > 1:
            norms += [norm] * (len(self.emb_dims) - 1)
            acts += [act] * (len(self.emb_dims) - 1)
        norms += [norm]
        acts += [act] if len(self.emb_dims) > 1 else [None]

        return mlp(input_dim=input_dim, fc_dims=self.emb_dims, norms=norms, acts=acts)

    def create_spk_cls_head(self, input_dim=512):
        if self.output_act.startswith('amsoftmax'):  # 'amsoftmax-0.25-30'
            _, m, s = self.output_act.split('-')
            spk_cls_layer = AMSoftmaxLayer(in_nodes=input_dim, n_class=self.n_class, m=float(m), s=float(s))
        elif self.output_act == 'softmax':
            spk_cls_layer = SoftmaxLayer(in_nodes=input_dim, n_class=self.n_class)
        else:
            raise NotImplementedError

        return spk_cls_layer

    def forward(self, x, label):
        """
        Args:
            x: Tensor, [batch_size, wav_len]
            label: Tensor, [batch_size,]
        Returns:
            prediction: Tensor, [batch_size, n_class], prediction with margin
            prediction_softmax: Tensor, [batch_size, n_class], prediction without margin
        """

        x = self.spk_model.spk_encoder(x)
        return self.spk_model.spk_cls_head(x) if self.output_act == 'softmax' else self.spk_model.spk_cls_head(x, label)


class TDNNFrameLevelLayers(nn.Module):
    def __init__(self, input_dim, filters, kernel_sizes, dilations, norm='bn'):
        super().__init__()

        frame_level_layers = OrderedDict()
        frame_level_layers['conv0'] = conv1d_unit(
            input_dim, filters[0], kernel_sizes[0], padding='same', dilation=dilations[0], norm=norm)

        for i in range(1, len(filters)):
            frame_level_layers[f'conv{i}'] = conv1d_unit(
                filters[i - 1], filters[i], kernel_sizes[i], padding='same', dilation=dilations[i], norm=norm)

        self.frame_level_layers = nn.Sequential(frame_level_layers)
        self.output_chnls = filters[-1]  # as pooling input dim

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, in_feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, out_feat_dim, seq_len]
        """
        return self.frame_level_layers(x)
