
from collections import OrderedDict
from front_end.attention import (RelAttSkew, RelAttFusion, RelAttDisentangle, RelAttDisentangleFusion,
                                 RelAttXL, RelAttXLFusion, RelAttShaw)
from front_end.model_misc import LayerNorm
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
from timm.models.layers import DropPath, trunc_normal_


class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, heads=4, max_rel_dist=127, att_ds=2, att_dropout=0.1, ff_expansion=4,
                 ff_dropout=0.1, conv_expansion=2, conv_kernel_size=15, conv_dropout=0.1, drop_path=0.1,
                 layer_scale=False, ls_init=1e0, rmsn=False, ln_eps=1e-12, stft_cfg=None,
                 rel_att_type='xl', pre_norm=True):
        super().__init__()

        self.ff = FeedForwardModule(
            d_model=d_model, expansion=ff_expansion, dropout=ff_dropout, pre_norm=pre_norm, rmsn=rmsn, ln_eps=ln_eps)

        self.att = SelfAttentionModule(
            d_model=d_model, heads=heads, max_rel_dist=max_rel_dist, att_ds=att_ds, dropout=att_dropout,
            rel_att_type=rel_att_type, stft_cfg=stft_cfg, pre_norm=pre_norm, rmsn=rmsn, ln_eps=ln_eps)

        self.conv = ConvModule(
            d_model=d_model, kernel_size=conv_kernel_size, expansion=conv_expansion, dropout=conv_dropout,
            pre_norm=pre_norm, rmsn=rmsn, ln_eps=ln_eps)

        self.ff1 = FeedForwardModule(
            d_model=d_model, expansion=ff_expansion, pre_norm=pre_norm, dropout=ff_dropout, rmsn=rmsn, ln_eps=ln_eps)

        self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps)  # used for conformer only
        self.ff_ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if not pre_norm else nn.Identity()
        self.att_ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if not pre_norm else nn.Identity()
        self.conv_ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if not pre_norm else nn.Identity()
        self.ff1_ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if not pre_norm else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls_ff = ls_init * nn.Parameter(torch.ones(d_model, ), requires_grad=True) if layer_scale else 1.0
        self.ls_att = ls_init * nn.Parameter(torch.ones(d_model, ), requires_grad=True) if layer_scale else 1.0
        self.ls_conv = ls_init * nn.Parameter(torch.ones(d_model, ), requires_grad=True) if layer_scale else 1.0
        self.ls_ff1 = ls_init * nn.Parameter(torch.ones(d_model, ), requires_grad=True) if layer_scale else 1.0

        # self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)

    def forward(self, x, pos_emb):
        """
        Args:
            x: Tensor, [batch_size, seq_len, feat_dim]
            pos_emb: Tensor, [batch_size, 2 * seq_len - 1, feat_dim]
        Returns:
            Tensor, [batch_size, seq_len, feat_dim]
        """

        x = x + 0.5 * self.drop_path(self.ls_ff * self.ff(x))
        x = self.ff_ln(x)

        x_skip = x
        x, attn, content_score, pos_score = self.att(x, pos_emb)
        x = x_skip + self.drop_path(self.ls_att * x)
        x = self.att_ln(x)

        x = x + self.drop_path(self.ls_conv * self.conv(x))
        x = self.conv_ln(x)

        x = x + 0.5 * self.drop_path(self.ls_ff1 * self.ff1(x))
        x = self.ff1_ln(x)

        x = self.ln(x)

        return x, attn, content_score, pos_score


class ConfusionformerBlock(ConformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ff1, self.ff1_ln, self.ls_ff1 = None, None, None
        self.ln = None

    def forward(self, x, pos_emb):
        """
        Args:
            x: Tensor, [batch_size, seq_len, feat_dim]
            pos_emb: Tensor, [batch_size, 2 * seq_len - 1, feat_dim]
        Returns:
            Tensor, [batch_size, seq_len, feat_dim]
        """

        x_skip = x
        x, attn, content_score, pos_score = self.att(x, pos_emb)
        x = x_skip + self.drop_path(self.ls_att * x)
        x = self.att_ln(x)

        x = x + self.drop_path(self.ls_ff * self.ff(x))
        x = self.ff_ln(x)

        x = x + self.drop_path(self.ls_conv * self.conv(x))
        x = self.conv_ln(x)

        return x, attn, content_score, pos_score


class TransformerBlock(ConformerBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ff1, self.ff1_ln, self.ls_ff1 = None, None, None
        self.conv, self.conv_ln, self.ls_conv = None, None, None
        self.ln = None

    def forward(self, x, pos_emb):
        """
        Args:
            x: Tensor, [batch_size, seq_len, feat_dim]
            pos_emb: Tensor, [batch_size, 2 * seq_len - 1, feat_dim]
        Returns:
            Tensor, [batch_size, seq_len, feat_dim]
        """

        x_skip = x
        x, attn, content_score, pos_score = self.att(x, pos_emb)
        x = x_skip + self.drop_path(self.ls_att * x)
        x = self.att_ln(x)

        x = x + self.drop_path(self.ls_ff * self.ff(x))
        x = self.ff_ln(x)

        return x, attn, content_score, pos_score


class SelfAttentionModule(nn.Module):
    """ Multi-headed self-attention (MHSA) with relative positional encoding """

    def __init__(self, d_model, heads=4, max_rel_dist=127, att_ds=2, dropout=0.1, rel_att_type='skew',
                 stft_cfg=None, pre_norm=True, rmsn=False, ln_eps=1e-12):
        super().__init__()

        self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if pre_norm else nn.Identity()

        if rel_att_type == 'skew':
            self.attention = RelAttSkew(d_model=d_model, heads=heads, max_rel_dist=max_rel_dist)
        elif rel_att_type == 'fusion':
            self.attention = RelAttFusion(d_model=d_model, heads=heads, max_rel_dist=max_rel_dist, att_ds=att_ds)
        elif rel_att_type == 'disentangle':
            self.attention = RelAttDisentangle(d_model=d_model, heads=heads, max_rel_dist=max_rel_dist)
        elif rel_att_type == 'dis_fusion':
            self.attention = RelAttDisentangleFusion(
                d_model=d_model, heads=heads, max_rel_dist=max_rel_dist, att_ds=att_ds)
        elif rel_att_type == 'xl':
            self.attention = RelAttXL(d_model=d_model, heads=heads, max_rel_dist=max_rel_dist)
        elif rel_att_type == 'xl_fusion':
            self.attention = RelAttXLFusion(d_model=d_model, heads=heads, max_rel_dist=max_rel_dist, att_ds=att_ds)
        elif rel_att_type == 'shaw':
            self.attention = RelAttShaw(d_model=d_model, heads=heads, max_rel_dist=max_rel_dist)
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, pos_emb):
        """
        Args:
            x: Tensor, [batch_size, seq_len, feat_dim]
            pos_emb: Tensor, [batch_size, 2 * seq_len - 1, feat_dim]
        Returns:
            Tensor, [batch_size, seq_len, feat_dim]
        """

        x = self.ln(x)
        x, attn, content_score, pos_score = self.attention(x, x, x, pos_emb=pos_emb)

        return self.dropout(x), attn, content_score, pos_score


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1, pre_norm=True, rmsn=False, ln_eps=1e-12, wn=False):
        super().__init__()

        self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if pre_norm else nn.Identity()

        self.ff = nn.Sequential(OrderedDict([
            ('fc1', nn.utils.weight_norm(nn.Linear(d_model, d_model * expansion)) if wn else
                nn.Linear(d_model, d_model * expansion)),
            ('act', nn.SiLU()),
            # ('sub_ln', LayerNorm(d_model * expansion, rmsn=rmsn, eps=ln_eps)),  # useless
            ('fc2', nn.utils.weight_norm(nn.Linear(d_model * expansion, d_model)) if wn else
                nn.Linear(d_model * expansion, d_model)),
            ('dropout', nn.Dropout(p=dropout))]))

        if wn:
            self.ff.fc1.weight_g.data.fill_(1)
            self.ff.fc1.weight_g.requires_grad = False

            self.ff.fc2.weight_g.data.fill_(1)
            self.ff.fc2.weight_g.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, seq_len, feat_dim]
        Returns:
            Tensor, [batch_size, seq_len, feat_dim]
        """

        x = self.ln(x)

        return self.ff(x)


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, expansion=2, pre_norm=True, dropout=0.1, rmsn=False, ln_eps=1e-12):
        super().__init__()

        assert (kernel_size - 1) % 2 == 0, 'kernel_size should be a odd number for "SAME" padding!'
        self.ln = LayerNorm(d_model, rmsn=rmsn, eps=ln_eps) if pre_norm else nn.Identity()

        self.conv = nn.Sequential(OrderedDict(
            [('conv1d_p1', nn.Conv1d(d_model, d_model * expansion, kernel_size=1, padding='same')),
             ('act_p1', nn.GLU(dim=1)),
             ('conv1d_g', nn.Conv1d(
                 d_model * expansion // 2, d_model * expansion // 2, kernel_size,
                 groups=d_model * expansion // 2, padding='same')),
             ('bn_g', nn.BatchNorm1d(d_model * expansion // 2)),
             ('act_g', nn.SiLU()),
             ('conv1d_p2', nn.Conv1d(d_model * expansion // 2, d_model, kernel_size=1, padding='same')),
             ('dropout', nn.Dropout(p=dropout))]))

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, seq_len, feat_dim]
        Returns:
            Tensor, [batch_size, seq_len, feat_dim]
        """

        x = self.ln(x).transpose(1, 2)  # [B, D, T]
        return self.conv(x).transpose(1, 2)


class PoolFormerBlock2d(nn.Module):
    def __init__(self, d_model=256, pool_size=3, ff_expansion=4, dropout=0.1, drop_path=0., use_layer_scale=False,
                 layer_scale_init_value=1e-5):
        super().__init__()

        self.ln1 = LayerNorm(d_model, in_2d=True)
        self.pool = PoolModule2d(pool_size=pool_size)
        self.ln2 = LayerNorm(d_model, in_2d=True)
        self.ff = FeedForwardModule2d(d_model=d_model, expansion_factor=ff_expansion, dropout=dropout)

        # The following two techniques are useful to train deep networks
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale

        if use_layer_scale:
            self.layer_scale1 = nn.Parameter(layer_scale_init_value * torch.ones(d_model,), requires_grad=True)
            self.layer_scale2 = nn.Parameter(layer_scale_init_value * torch.ones(d_model,), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, channel, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, channel, feat_dim, seq_len]
        """

        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale1.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.pool(self.ln1(x)))
            x = x + self.drop_path(self.layer_scale2.unsqueeze(0).unsqueeze(2).unsqueeze(3) * self.ff(self.ln2(x)))
        else:
            x = x + self.drop_path(self.pool(self.ln1(x)))
            x = x + self.drop_path(self.ff(self.ln2(x)))

        return x


class PoolModule2d(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, channel, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, channel, feat_dim, seq_len]
        """

        return self.pool(x) - x


class FeedForwardModule2d(nn.Module):
    def __init__(self, d_model=256, expansion_factor=4, dropout=0.1):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Conv2d(d_model, d_model * expansion_factor, kernel_size=1, padding='same'),
            nn.SiLU(),  # nn.GELU()
            nn.Dropout(p=dropout),
            nn.Conv2d(d_model * expansion_factor, d_model, kernel_size=1, padding='same'),
            nn.Dropout(p=dropout)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, channel, feat_dim, seq_len]
        Returns:
            Tensor, [batch_size, channel, feat_dim, seq_len]
        """

        return self.ff(x)
