
import ast
from front_end.feat_proc import FeatureExtractionLayer
from front_end.former_blk import ConformerBlock, ConfusionformerBlock, TransformerBlock, PoolFormerBlock2d
from front_end.model_misc import conv1d_unit, conv2d_unit, LayerNorm
from front_end.model_tdnn import TDNN
import math
# from timm.layers import trunc_normal_
import torch
import torch.nn as nn


class Former(TDNN):
    def __init__(self, cfg='(6,256,0.15,0,1e0,0,1e-12,0)|(4,127,2,0.1)|(4,0.1)|(2,15,0.1)|(16,16,8,5,0,0)|skew',
                 name='conformer', **kwargs):
        self.cfg = cfg
        self.name = name
        super().__init__(name=name, **kwargs)

    def create_spk_encoder(self):
        return SpkEncoder(
            input_dim=self.feat_dim, spec_aug=self.spec_aug, create_frame_level_layers=self.create_frame_level_layers,
            create_pooling_layer=self.create_pooling_layer, create_emb_layers=self.create_emb_layers)

    def create_frame_level_layers(self, input_dim=80):
        cfg = self.cfg.split('|')
        n_blks, d_model, drop_path, layer_scale, ls_init, rmsn, ln_eps, mfa = ast.literal_eval(cfg[0])
        heads, rel_dist, att_ds, att_dropout = ast.literal_eval(cfg[1])
        ff_expansion, ff_dropout = ast.literal_eval(cfg[2])
        conv_expansion, conv_kernel_size, conv_dropout = ast.literal_eval(cfg[3])
        stft_cfg = ast.literal_eval(cfg[4])
        rel_att_type = cfg[-1]

        print(f'n_blks: {n_blks}, d_model: {d_model}, drop_path: {drop_path}, '
              f'layer_scale: {layer_scale}, ls_init: {ls_init}, rmsn: {rmsn}, ln_eps: {ln_eps}, '
              f'heads: {heads}, rel_dist: {rel_dist}, att_ds: {att_ds}, att_dropout: {att_dropout}, '
              f'ff_expansion: {ff_expansion}, ff_dropout: {ff_dropout}, '
              f'conv_expansion: {conv_expansion}, conv_kernel: {conv_kernel_size}, conv_dropout: {conv_dropout}, '
              f'stft_cfg: {stft_cfg}, rel_att_type: {rel_att_type}, mfa: {mfa}')

        return ConformerFrameLevelLayers(
            input_dim=input_dim, n_blks=n_blks, d_model=d_model, heads=heads, max_rel_dist=rel_dist,
            att_ds=att_ds, att_dropout=att_dropout, ff_expansion=ff_expansion, ff_dropout=ff_dropout,
            conv_expansion=conv_expansion, conv_kernel_size=conv_kernel_size, conv_dropout=conv_dropout,
            drop_path=drop_path, layer_scale=layer_scale, ls_init=ls_init,
            rmsn=rmsn, ln_eps=ln_eps, stft_cfg=stft_cfg, rel_att_type=rel_att_type, mfa=mfa, name=self.name)


class SpkEncoder(nn.Module):
    def __init__(self, input_dim=80, spec_aug=False, create_frame_level_layers=None, create_pooling_layer=None,
                 create_emb_layers=None):
        super().__init__()

        self.feat_layer = FeatureExtractionLayer(feat_dim=input_dim, spec_aug=spec_aug)
        self.frame_layer = create_frame_level_layers(input_dim=input_dim)
        self.pool_layer = create_pooling_layer(input_dim=self.frame_layer.output_chnls, norm='bn')
        self.emb_layer = create_emb_layers(input_dim=self.pool_layer[0].output_dim, norm='bn')

    def forward(self, x):
        x = self.feat_layer(x)
        x, _, _, _ = self.frame_layer(x)
        x = self.pool_layer(x)
        x = self.emb_layer(x)

        return x


class ConformerFrameLevelLayers(nn.Module):
    def __init__(self, input_dim, n_blks=6, d_model=256, heads=4, max_rel_dist=127, att_ds=2, att_dropout=0.1,
                 ff_expansion=4, ff_dropout=0.1, conv_expansion=2, conv_kernel_size=15, conv_dropout=0.1,
                 drop_path=0.15, layer_scale=False, ls_init=1e-5, rmsn=False, ln_eps=1e-12, stft_cfg=None,
                 rel_att_type='xl', mfa=False, name='conformer'):
        super().__init__()

        self.n_blks = n_blks
        self.d_model = d_model
        self.rel_att_type = rel_att_type
        self.mfa = mfa

        self.stem = ConvNextStem(input_dim=input_dim, d_model=d_model, stem_dropout=0.1, rmsn=rmsn, ln_eps=ln_eps)
        self.rpe = RelPosEncoding(
            n_blks=n_blks, d_model=d_model, heads=heads, max_rel_dist=max_rel_dist, rel_att_type=rel_att_type)

        if name == 'conformer':
            blk_func = ConformerBlock
        elif name == 'confusionformer':
            blk_func = ConfusionformerBlock
        else:
            blk_func = TransformerBlock

        self.conformer_blks = nn.ModuleList([blk_func(
            d_model=d_model, heads=heads, max_rel_dist=max_rel_dist, att_ds=att_ds, att_dropout=att_dropout,
            ff_expansion=ff_expansion, ff_dropout=ff_dropout, conv_expansion=conv_expansion,
            conv_kernel_size=conv_kernel_size, conv_dropout=conv_dropout, drop_path=drop_path,
            layer_scale=layer_scale, ls_init=ls_init, rmsn=rmsn, ln_eps=ln_eps, stft_cfg=stft_cfg,
            rel_att_type=rel_att_type) for _ in range(n_blks)])

        if mfa:
            self.output_chnls = d_model * n_blks  # as pooling input dim
            self.ln = LayerNorm(self.output_chnls, rmsn=rmsn, eps=ln_eps)
        else:
            self.output_chnls = 1024
            self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps)
            self.conv_proj = conv1d_unit(d_model, self.output_chnls, 1)

        # self._init_weights()

    def _init_weights(self):
        init_scale = math.sqrt(math.log(self.n_blks * 2))

        for name, para in self.named_parameters():
            if 'att' in name:
                if 'val_proj' in name or 'out_proj' in name:
                    para.data.mul_(init_scale)

            if 'ff' in name:
                if 'fc1' in name or 'fc2' in name:
                    para.data.mul_(init_scale)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, cat_model_dim/feat_dim, seq_len]
        """

        x = self.stem(x)
        pos_emb = self.rpe(x)

        if 'xl' in self.rel_att_type:
            x = math.sqrt(self.d_model) * x

        x_out = []
        attn, content_score, pos_score = None, None, None

        for i in range(len(self.conformer_blks)):
            pos_emb_ = pos_emb if 'xl' in self.rel_att_type else pos_emb[i]
            x, attn, content_score, pos_score = self.conformer_blks[i](x, pos_emb_)

            if self.mfa:
                x_out.append(x)

        if self.mfa:
            x = torch.concat(x_out, dim=-1)  # [B, T, D]
            x = self.ln(x).transpose(1, 2)  # [B, D, T]
        else:
            x = self.ln(x).transpose(1, 2)  # [B, D, T]
            x = self.conv_proj(x)

        return x, attn, content_score, pos_score


class ConvNextStem(nn.Module):
    def __init__(self, input_dim=80, d_model=256, stem_dropout=0.1, rmsn=False, ln_eps=1e-12, conv_next=True):
        super().__init__()

        chnls = [8, 32, 128]
        strides = [(1, 1), (2, 2), (2, 1)]
        self.sub_conv1 = conv2d_unit(1, chnls[0], 3, stride=strides[0], padding=1, norm=None, act=nn.GELU())
        self.sub_conv2 = conv2d_unit(chnls[0], chnls[1], 3, stride=strides[1], padding=1, norm=None, act=nn.GELU())
        self.sub_conv3 = conv2d_unit(chnls[1], chnls[2], 3, stride=strides[2], padding=1, norm=None, act=nn.GELU())

        if conv_next:
            self.sub_next_conv = ConvNextLayer(chnls[2], expansion=4, rmsn=rmsn, ln_eps=ln_eps)

        freq_strides = [stride[0] for stride in strides]
        freq_dim = input_dim

        for freq_stride in freq_strides:
            freq_dim = (freq_dim - 1) // freq_stride + 1

        self.sub_proj = nn.Linear(freq_dim * chnls[-1], d_model)
        self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps)
        self.dropout = nn.Dropout(p=stem_dropout)
        self.conv_next = conv_next

    def forward(self, x):
        """
        Args:
            x: Tensor, F-banks, [batch_size, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, seq_len, model_dim]
        """

        # print(f'in: {x.shape}')
        x = self.sub_conv1(x.unsqueeze(1))
        # print(f'sub_conv1: {x.shape}')
        x = self.sub_conv2(x)
        # print(f'sub_conv2: {x.shape}')
        x = self.sub_conv3(x)
        # print(f'sub_conv3: {x.shape}')

        if self.conv_next:
            x = self.sub_next_conv(x)

        batch_size, _, _, seq_len = x.size()
        x = x.view(batch_size, -1, seq_len).transpose(1, 2)  # [B, T, D]
        x = self.sub_proj(x)
        x = self.ln(x)
        x = self.dropout(x)

        return x


class ConvNextLayer(nn.Module):
    def __init__(self, in_chanls=128, expansion=4, kernel_size=7, rmsn=False, ln_eps=1e-12):
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_chanls, in_chanls, kernel_size, padding=kernel_size // 2, groups=in_chanls)
        self.ln = LayerNorm(in_chanls, gn=False, groups=in_chanls, in_2d=True, rmsn=rmsn, eps=ln_eps)
        self.pointwise_conv1 = nn.Conv2d(in_chanls, in_chanls * expansion, 1)
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Conv2d(in_chanls * expansion, in_chanls, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, channel, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, channel, feat_dim, seq_len]
        """

        x_in = x
        x = self.depthwise_conv(x)
        x = self.ln(x)
        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        return x + x_in


class PoolStem(nn.Module):
    def __init__(self, input_dim=80, d_model=256, stem_dropout=0.1, rmsn=False, ln_eps=1e-12):
        super().__init__()

        chnls = [32, 128]
        self.sub_conv1 = conv2d_unit(1, chnls[0], 3, stride=(2, 1), padding=1, norm=None, act=nn.GELU())
        self.pool_blk1 = PoolFormerBlock2d(d_model=chnls[0], pool_size=7, ff_expansion=3)

        self.sub_conv2 = conv2d_unit(chnls[0], chnls[1], 3, stride=2, padding=1, norm=None, act=nn.GELU())
        self.pool_blk2 = PoolFormerBlock2d(d_model=chnls[1], pool_size=7, ff_expansion=3)

        self.sub_proj = nn.Linear(((input_dim + 1) // 2 + 1) // 2 * chnls[-1], d_model)
        self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps)
        self.dropout = nn.Dropout(p=stem_dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, seq_len, model_dim]
        """

        # print(f'x: {x.shape}')
        x = self.sub_conv1(x.unsqueeze(1))
        x = self.pool_blk1(x)
        # print(f'pool_blk1: {x.shape}')
        x = self.sub_conv2(x)
        x = self.pool_blk2(x)
        # print(f'pool_blk2: {x.shape}')

        batch_size, _, _, seq_len = x.size()
        x = x.view(batch_size, -1, seq_len).transpose(1, 2)  # [B, T, D]
        x = self.sub_proj(x)
        x = self.ln(x)
        x = self.dropout(x)

        return x


class RelPosEncoding(nn.Module):
    def __init__(self, n_blks=6, d_model=256, heads=4, max_rel_dist=127, rel_att_type='xl'):
        super().__init__()

        self.rel_att_type = rel_att_type

        if rel_att_type in ['skew', 'fusion', 'stft']:
            self.pos_enc = nn.ParameterList([nn.Parameter(
                torch.Tensor(2 * max_rel_dist + 1, d_model // heads)) for _ in range(n_blks)])  # [2*R+1, d_H]

            for i in range(n_blks):
                nn.init.xavier_normal_(self.pos_enc[i])  # trunc_normal_(self.pos_enc, std=.02)

        elif 'dis' in rel_att_type:
            self.pos_enc = nn.ParameterList([nn.Parameter(
                torch.Tensor(2 * max_rel_dist + 1, d_model)) for _ in range(n_blks)])  # [2*R+1, d_model]

            for i in range(n_blks):
                nn.init.xavier_normal_(self.pos_enc[i])

        elif 'xl' in rel_att_type:
            self.pos_enc = SinePositionalEncoding(d_model)  # [1, seq_len, d_model]
            self.drop = nn.Dropout(0.1)
        elif rel_att_type == 'shaw':
            self.pos_enc = nn.ModuleList([
                nn.Embedding(2 * max_rel_dist + 1, self.d_head) for _ in range(n_blks)])  # [2*R+1, d_H]
        elif rel_att_type == 't5':
            self.pos_enc = nn.Parameter(torch.Tensor(2 * max_rel_dist + 1,))  # [2*R+1,]
            nn.init.zeros_(self.pos_enc)
        else:
            self.pos_enc = None

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, seq_len, d_model]
        Returns:
            Tensor, [batch_size, seq_len, d_model/d_head]
        """

        batch_size, seq_len, _ = x.size()

        if 'xl' in self.rel_att_type:
            pos_emb = self.pos_enc(2 * seq_len - 1).repeat(batch_size, 1, 1)
            pos_emb = self.drop(pos_emb)
        else:
            pos_emb = self.pos_enc

        return pos_emb


class SinePositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """

    def __init__(self, d_model=256, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length: int):
        return self.pe[0, :length, :]  # [1, length, d_model]
