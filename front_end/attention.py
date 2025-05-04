
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchaudio.transforms import Spectrogram, InverseSpectrogram


class RelAttSkew(nn.Module):
    """ Music Transformer's relative attention (https://arxiv.org/abs/1809.04281) """

    def __init__(self, d_model=256, heads=4, max_rel_dist=127, dropout=0.0):
        super().__init__()

        assert d_model % heads == 0, 'd_model % heads should be zero!'
        self.d_model = d_model
        self.d_head = int(d_model / heads)
        self.heads = heads
        self.dim_scale = math.sqrt(self.d_model)  # using d_model is better
        self.max_rel_dist = max_rel_dist  # R (<= T-1)

        self.qry_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pos_proj = nn.Sequential(nn.Linear(self.d_head, self.d_head))

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: Tensor, [batch_size, seq_len, d_model]; pos_emb: [2 * seq_len - 1, d_head]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = query.size()

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 3, 1)  # [B, H, d_H, T]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]

        rel_pos_emb = self.pos_proj(pos_emb)  # [2*R+1, d_H], pos_proj shared among heads, better
        pos_score = torch.matmul(query, rel_pos_emb.T)  # [B, H, T, 2*R+1]
        pos_score = relative_shift(pos_score)  # [B, H, T, T]

        content_score = torch.matmul(query, key)  # [B, H, T, T]
        score = (content_score + pos_score) / self.dim_scale

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)  # [B, H, T, T]

        out = torch.matmul(attn, value).transpose(1, 2)  # [B, T, H, d_H]
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out), attn, content_score, pos_score


class RelAttFusion(RelAttSkew):
    """ Attention fusion """

    def __init__(self, att_ds=2, **kwargs):
        super().__init__(**kwargs)

        self.ds = att_ds
        self.q_ds = nn.Conv1d(self.d_head, self.d_head, 1, stride=self.ds, groups=self.d_head, padding=0)
        self.k_ds = nn.Conv1d(self.d_head, self.d_head, 1, stride=self.ds, groups=self.d_head, padding=0)
        self.w = nn.Parameter(torch.tensor(0.5))

        # self.spect = Spectrogram(
        #     n_fft=16, win_length=16, hop_length=8, window_fn=torch.hamming_window, power=None)
        # self.ispect = InverseSpectrogram(
        #     n_fft=16, win_length=16, hop_length=8, window_fn=torch.hamming_window)
        # self.low_freqs = 5

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: Tensor, [batch_size, seq_len, d_model]; rel_pos_emb: [2 * seq_len - 1, d_head/d_model]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = query.size()
        # query = query.to(torch.double)

        # spect_qry = self.spect(query.transpose(1, 2))  # [B, d, Fr, N]
        # spect_qry_real = spect_qry.real
        # spect_qry_real[:, :, self.low_freqs:, :] = 0.
        # spect_qry_imag = spect_qry.imag
        # spect_qry_imag[:, :, self.low_freqs:, :] = 0.

        # query = self.ispect(torch.complex(spect_qry_real, spect_qry_imag), seq_len).transpose(1, 2)  # [B, H, d_H, T]
        # query = query.to(torch.float32)
        # key = query
        # value = query

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 3, 1)  # [B, H, d_H, T]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]

        rel_pos_emb = self.pos_proj(pos_emb)  # [2*R+1, d_H], pos_proj shared among heads, better
        pos_score = torch.matmul(query, rel_pos_emb.T)  # [B, H, T, 2*R+1]
        pos_score = relative_shift(pos_score)  # [B, H, T, T]

        query_ds = self.q_ds(
            query.contiguous().view(batch_size * self.heads, seq_len, self.d_head).transpose(1, 2))  # [B*H, d_H, T//2]
        query_ds = query_ds.view(batch_size, self.heads, self.d_head, -1).transpose(2, 3)  # [B, H, T//2, d_H]
        key_ds = self.k_ds(key.contiguous().view(batch_size * self.heads, self.d_head, seq_len))  # [B*H, d_H, T//2]
        key_ds = key_ds.view(batch_size, self.heads, self.d_head, -1)  # [B, H, d_H, T//2]

        content_score_ds = torch.matmul(query_ds, key_ds)  # [B, H, T//2, T//2]
        content_score_ds = torch.repeat_interleave(content_score_ds / self.ds, self.ds, dim=3)[:, :, :, :seq_len]
        content_score_ds = torch.repeat_interleave(content_score_ds, self.ds, dim=2)[:, :, :seq_len, :]

        content_score = torch.matmul(query, key)  # [B, H, T, T]
        content_scoref = content_score + self.w * content_score_ds
        score = (content_scoref + pos_score) / self.dim_scale
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)  # [B, H, T, T]

        out = torch.matmul(attn, value).transpose(1, 2)  # [B, T, H, d_H]
        out = out.contiguous().view(batch_size, -1, self.d_model)
        out = self.out_proj(out)

        return out, attn, content_score, pos_score


class Downsample(nn.Module):
    def __init__(self, d_head=64, ds_factor=2):
        super().__init__()
        self.d_head = d_head
        self.ds = ds_factor
        self.w = nn.Parameter(torch.Tensor(self.d_head, self.ds))
        nn.init.constant_(self.w, 0.5)
        self.ff = nn.Conv1d(self.d_head, self.d_head, 1, stride=1, groups=self.d_head, padding=0)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, heads, seq_len, d_head]
        Returns:
            Tensor, [batch_size, heads, seq_len // ds_factor, d_head]
        """

        batch_size, heads, seq_len, _ = x.size()
        seq_len_ds = (seq_len + self.ds - 1) // self.ds
        x = self.ff(x.contiguous().view(batch_size * heads, seq_len, self.d_head).transpose(1, 2))  # [B*H, d_H, T]
        x = x.view(batch_size, heads, self.d_head, -1).transpose(2, 3)  # [B, H, T, d_H]

        x = torch.nn.functional.pad(x, (0, 0, 0, seq_len_ds * self.ds - seq_len))
        x = x.view(batch_size, heads, seq_len_ds, self.ds, -1).transpose(3, 4)  # [B, H, L//ds, d, ds]
        # w = torch.concat([1. - self.w.sum(1, keepdim=True), self.w], dim=1)
        w = self.w.softmax(dim=1)
        x = torch.einsum('bhlds, ds -> bhld', x, w)  # [B, H, L//ds, d]

        return x


class RelAttDisentangle(RelAttSkew):
    """ DeBERTa's disentangled relative attention (https://arxiv.org/abs/2006.03654) """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim_scale = math.sqrt(3 * self.d_model)
        self.pos_proj = None
        self.pos_key_proj = nn.Linear(self.d_model, self.d_model)  # content-to-position
        self.pos_qry_proj = nn.Linear(self.d_model, self.d_model)  # position-to-content

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: Tensor, [batch_size, seq_len, d_model]; rel_pos_emb: [2 * seq_len - 1, d_model]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = query.size()

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 3, 1)  # [B, H, d_H, T]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]

        pos_key = self.pos_key_proj(pos_emb).view(-1, self.heads, self.d_head)  # [2*R+1, H, d_H]
        pos_score_c2p = torch.einsum('bhld, rhd -> bhlr', query, pos_key)  # [B, H, T, 2*R+1]
        pos_score_c2p = relative_shift(pos_score_c2p)

        pos_query = self.pos_qry_proj(pos_emb).view(-1, self.heads, self.d_head)  # [2*R+1, H, d_H]
        pos_score_p2c = torch.einsum('rhd, bhdl -> bhrl', pos_query, key)  # [B, H, 2*R+1, T]
        pos_score_p2c = relative_shift(pos_score_p2c.transpose(2, 3)).transpose(2, 3)

        content_score = torch.matmul(query, key)  # [B, H, T, T]
        score = (content_score + pos_score_c2p + pos_score_p2c) / self.dim_scale

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2)  # [B, T, H, d_H]
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out), attn, pos_score_c2p, pos_score_p2c


class RelAttDisentangleFusion(RelAttFusion):
    """ DeBERTa's disentangled relative attention (https://arxiv.org/abs/2006.03654) """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dim_scale = math.sqrt(3 * self.d_model)
        self.pos_proj = None
        self.pos_key_proj = nn.Linear(self.d_model, self.d_model)  # content-to-position
        self.pos_qry_proj = nn.Linear(self.d_model, self.d_model)  # position-to-content

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: Tensor, [batch_size, seq_len, d_model]; rel_pos_emb: [2 * seq_len - 1, d_model]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = query.size()

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 3, 1)  # [B, H, d_H, T]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]

        pos_key = self.pos_key_proj(pos_emb).view(-1, self.heads, self.d_head)  # [2*R+1, H, d_H]
        pos_score_c2p = torch.einsum('bhld, rhd -> bhlr', query, pos_key)  # [B, H, T, 2*R+1]
        pos_score_c2p = relative_shift(pos_score_c2p)

        pos_query = self.pos_qry_proj(pos_emb).view(-1, self.heads, self.d_head)  # [2*R+1, H, d_H]
        pos_score_p2c = torch.einsum('rhd, bhdl -> bhrl', pos_query, key)  # [B, H, 2*R+1, T]
        pos_score_p2c = relative_shift(pos_score_p2c.transpose(2, 3)).transpose(2, 3)

        query_ds = self.q_ds(query.contiguous().view(
            batch_size * self.heads, seq_len, self.d_head).transpose(1, 2))  # [B*H, d_H, T//2]
        query_ds = query_ds.view(batch_size, self.heads, self.d_head, -1).transpose(2, 3)  # [B, H, T//2, d_H]
        key_ds = self.k_ds(key.contiguous().view(batch_size * self.heads, self.d_head, seq_len))  # [B*H, d_H, T//2]
        key_ds = key_ds.view(batch_size, self.heads, self.d_head, -1)  # [B, H, d_H, T//2]

        content_score = torch.matmul(query, key)  # [B, H, T, T]
        content_score_ds = torch.matmul(query_ds, key_ds)  # [B, H, T//2, T//2]
        content_score_ds = torch.repeat_interleave(content_score_ds / self.ds, self.ds, dim=3)[:, :, :, :seq_len]
        content_score_ds = torch.repeat_interleave(content_score_ds, self.ds, dim=2)[:, :, :seq_len, :]
        content_score = (1 - self.w) * content_score + self.w * content_score_ds

        score = (content_score + pos_score_c2p + pos_score_p2c) / self.dim_scale
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2)  # [B, T, H, d_H]
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out), attn, pos_score_c2p, pos_score_p2c


class RelAttXL(RelAttSkew):
    """ Transformer-XL's relative attention (https://arxiv.org/abs/1901.02860) """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]; pos_emb: [batch_size, 2 * seq_len - 1, d_model]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size = value.size(0)

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head)  # [B, T, H, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 1, 3)  # [B, H, T, d_H]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 1, 3)
        pos_emb = self.pos_proj(pos_emb).view(batch_size, -1, self.heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))  # [B, H, T, T]
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_emb.permute(0, 2, 3, 1))  # [B, H, T, 2*T-1]
        pos_score = relative_shift(pos_score)

        score = (content_score + pos_score) / self.dim_scale
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2)
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out), attn, content_score, pos_score


class RelAttXLFusion(RelAttFusion):
    """ Transformer-XL's relative attention (https://arxiv.org/abs/1901.02860) """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pos_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.u_bias = nn.Parameter(torch.Tensor(self.heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]; pos_emb: [batch_size, 2 * seq_len - 1, d_model]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = query.size()

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head)  # [B, T, H, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 1, 3)  # [B, H, T, d_H]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 1, 3)
        pos_emb = self.pos_proj(pos_emb).view(batch_size, -1, self.heads, self.d_head)

        query_ds = self.q_ds(query.contiguous().view(
            batch_size * self.heads, seq_len, self.d_head).transpose(1, 2))  # [B*H, d_H, T//2]
        query_ds = query_ds.view(batch_size, self.heads, self.d_head, -1).transpose(2, 3)  # [B, H, T//2, d_H]
        key_ds = self.k_ds(key.contiguous().view(batch_size * self.heads, self.d_head, seq_len))  # [B*H, d_H, T//2]
        key_ds = key_ds.view(batch_size, self.heads, self.d_head, -1)  # [B, H, d_H, T//2]

        # content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))  # [B, H, T, T]
        content_score = torch.matmul(query.transpose(1, 2), key.transpose(2, 3))  # [B, H, T, T]
        content_score_ds = torch.matmul(query_ds, key_ds)  # [B, H, T//2, T//2]
        content_score_ds = torch.repeat_interleave(content_score_ds / self.ds, self.ds, dim=3)[:, :, :, :seq_len]
        content_score_ds = torch.repeat_interleave(content_score_ds, self.ds, dim=2)[:, :, :seq_len, :]
        content_score = (1 - self.w) * content_score + self.w * content_score_ds  # [B, H, T, T]
        content_score_bias = torch.matmul(self.u_bias.unsqueeze(1).repeat(
            1, seq_len, 1).unsqueeze(0).repeat(batch_size, 1, 1, 1), key.transpose(2, 3))
        content_score = content_score + content_score_bias

        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_emb.permute(0, 2, 3, 1))  # [B, H, T, 2*T-1]
        pos_score = relative_shift(pos_score)

        score = (content_score + pos_score) / self.dim_scale
        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2)
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out), attn, content_score, pos_score


class RelAttShaw(RelAttSkew):
    """ Shaw's relative attention (https://arxiv.org/abs/1803.02155) """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query, key, value, pos_emb):
        """
        Args:
            query, key, value: Tensor, [batch_size, seq_len, d_model]; pos_emb: [2 * seq_len - 1, d_head]
        Returns:
            Tensor, [batch_size, seq_len, d_model]
        """

        batch_size, seq_len, _ = query.size()

        query = self.qry_proj(query).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]
        key = self.key_proj(key).view(batch_size, -1, self.heads, self.d_head).permute(0, 2, 3, 1)  # [B, H, d_H, T]
        value = self.val_proj(value).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)  # [B, H, T, d_H]

        seq = torch.arange(seq_len, device=query.device)
        rel_dist = (seq - seq.unsqueeze(1)).clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos_emb = pos_emb(rel_dist).to(query.device)  # [T, T, d_H]
        rel_pos_emb = self.pos_proj(rel_pos_emb)  # [T, T, d_H], pos_proj shared among heads
        pos_score = torch.einsum('bhld, lrd -> bhlr', query, rel_pos_emb)

        content_score = torch.matmul(query, key)  # [B, H, T, T]
        score = (content_score + pos_score) / self.dim_scale

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, value).transpose(1, 2)  # [B, T, H, d_H]
        out = out.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(out), attn, content_score, pos_score


def relative_shift(score):
    """
    Args:
        score: Tensor, [..., seq_len, seq_len_rel], e.g., [batch_size, heads, seq_len, seq_len_rel]
    Returns:
        Tensor, [..., seq_len, seq_len]
    """

    seq_len_q, seq_len_rel = score.size()[-2:]  # T, 2*R+1
    slice_shape = score.size()[:-2]

    if not seq_len_rel == 2 * seq_len_q - 1:
        pad_len = seq_len_q - (seq_len_rel - 1) // 2 - 1
        score = F.pad(score, (pad_len, pad_len, 0, 0), mode='replicate')  # [..., T, 2*T-1]

    score = F.pad(score, pad=(1, 0), mode='constant', value=0.)
    score = score.view(slice_shape + (-1,))  # flatten the last 2 dimensions
    score = score[..., seq_len_q:].view(slice_shape + (seq_len_q, -1))

    return score[..., :seq_len_q]  # [..., T, T]
