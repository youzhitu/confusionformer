""" Sub-models """

from collections import OrderedDict
from einops.layers.torch import Rearrange
from functools import partial
import numpy as np
from einops import reduce
import torch
from torch.autograd import Function
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def conv1d_unit(in_chanls, out_chanls, kernel_size, stride=1, padding=0, dilation=1, groups=1, act=nn.GELU(),
                norm='bn', post_norm=True, transpose=False):
    conv1d_unit_layers = OrderedDict()

    if transpose:
        conv1d_unit_layers['conv'] = nn.ConvTranspose1d(
            in_chanls, out_chanls, kernel_size, stride=stride, padding=padding, dilation=dilation)
    else:
        conv1d_unit_layers['conv'] = nn.Conv1d(
            in_chanls, out_chanls, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)

    if post_norm:
        if act is not None:
            conv1d_unit_layers['act'] = act

        if norm == 'bn':
            conv1d_unit_layers['norm'] = nn.BatchNorm1d(out_chanls)
        elif norm == 'ln':
            conv1d_unit_layers['norm'] = nn.Sequential(TransposeLayer(), LayerNorm(out_chanls), TransposeLayer())
        else:
            conv1d_unit_layers['norm'] = nn.Identity()
    else:
        if norm == 'bn':
            conv1d_unit_layers['norm'] = nn.BatchNorm1d(out_chanls)
        elif norm == 'ln':
            conv1d_unit_layers['norm'] = nn.Sequential(TransposeLayer(), LayerNorm(out_chanls), TransposeLayer())
        else:
            conv1d_unit_layers['norm'] = nn.Identity()

        if act is not None:
            conv1d_unit_layers['act'] = act

    return nn.Sequential(conv1d_unit_layers)


def conv2d_unit(in_chanls, out_chanls, kernel_size, stride=1, padding=0, act=nn.GELU(), norm='bn', ws=False, spd=False):
    """
    ws: bool, weight standardization for conv2d
    spd: bool, https://arxiv.org/abs/2208.03641, space-to-depth conv2d, no more strided convolutions or pooling
    """
    conv2d_unit_layers = OrderedDict()

    if spd:
        conv2d_unit_layers['conv'] = SPDConv2D(in_chanls, out_chanls, stride=stride, padding=padding)
    else:
        conv2d_fn = WeightStandardizedConv2d if ws else nn.Conv2d
        conv2d_unit_layers['conv'] = conv2d_fn(in_chanls, out_chanls, kernel_size, stride=stride, padding=padding)

    if ws:
        conv2d_unit_layers['norm'] = nn.GroupNorm(4, out_chanls)
    else:
        if norm == 'bn':
            conv2d_unit_layers['norm'] = nn.BatchNorm2d(out_chanls)

    if act is not None:
        conv2d_unit_layers['act'] = act

    return nn.Sequential(conv2d_unit_layers)


def linear_unit(in_nodes, out_nodes, act=nn.GELU(), norm='bn', post_norm=True):
    linear_unit_layers = OrderedDict()
    linear_unit_layers['linear'] = nn.Linear(in_nodes, out_nodes)

    if post_norm:
        if act is not None:
            linear_unit_layers['act'] = act

        if norm == 'bn':
            linear_unit_layers['norm'] = nn.BatchNorm1d(out_nodes)
        elif norm == 'ln':
            linear_unit_layers['norm'] = LayerNorm(out_nodes)
        else:
            linear_unit_layers['norm'] = nn.Identity()
    else:
        if norm == 'bn':
            linear_unit_layers['norm'] = nn.BatchNorm1d(out_nodes)
        elif norm == 'ln':
            linear_unit_layers['norm'] = LayerNorm(out_nodes)
        else:
            linear_unit_layers['norm'] = nn.Identity()

        if act is not None:
            linear_unit_layers['act'] = act

    return nn.Sequential(linear_unit_layers)


def mlp(input_dim, fc_dims, norms, acts, post_norm=True):
    fc_layers = OrderedDict()

    if len(fc_dims) > 1:
        fc_layers['fc0'] = linear_unit(input_dim, fc_dims[0], norm=norms[0], act=acts[0], post_norm=post_norm)
        input_dim = fc_dims[0]

        for i in range(1, len(fc_dims) - 1):
            fc_layers[f'fc{i}'] = linear_unit(
                input_dim, fc_dims[i], norm=norms[i], act=acts[i], post_norm=post_norm)
            input_dim = fc_dims[i]

    fc_layers[f'fc{len(fc_dims) - 1}'] = linear_unit(
        input_dim, fc_dims[-1], norm=norms[-1], act=acts[-1], post_norm=post_norm)

    return nn.Sequential(fc_layers)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520, weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-6 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        # noinspection PyTypeChecker
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, correction=0))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SPDConv2D(nn.Module):
    """ https://arxiv.org/abs/2208.03641, space-to-depth conv2d, no more strided convolutions or pooling """
    def __init__(self, in_chanls, out_chanls, stride=(2, 2), padding=(1, 0)):
        super().__init__()
        stride = stride if type(stride) == tuple else (stride, stride)
        padding = padding if type(padding) == tuple else (padding, padding)

        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=stride[0], p2=stride[1])
        self.conv2d_1x1 = nn.Conv2d(in_chanls * stride[0] * stride[1], out_chanls, 1, stride=1, padding=padding)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.conv2d_1x1(x)

        return x


class TransposeLayer(nn.Module):
    """ Transpose the channel dimension, [B, C, L] <-> [B, L, C] """
    def __init__(self):
        super(TransposeLayer, self).__init__()

    @staticmethod
    def forward(x):
        return x.transpose(1, 2)


class LayerNorm(nn.Module):
    def __init__(self, channels, rmsn=False, gn=False, groups=1, eps=1e-12, in_2d=False):
        super().__init__()

        if gn:
            self.ln = nn.GroupNorm(groups, channels, eps=eps)
        elif rmsn:
            self.ln = RMSLayerNorm(channels, eps=eps)
        else:
            self.ln = nn.LayerNorm(channels, eps=eps)

        self.gn = gn
        self.in_2d = in_2d  # 2d feature map per channel

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, channel, feat_dim, seq_len] if in_2d==True else [batch_size, seq_len, feat_dim]
        Returns:
            Tensor, [batch_size, channel, feat_dim, seq_len] if in_2d==True else [batch_size, seq_len, feat_dim]
        """

        if self.gn:
            if self.in_2d:
                x = self.ln(x)
            else:
                x = self.ln(x.transpose(1, 2)).transpose(1, 2)
        else:
            if self.in_2d:
                x = self.ln(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            else:
                x = self.ln(x)

        return x


class BlurPool(nn.Module):
    def __init__(self, channels, filter_size=3, stride=2, padding=1, pad_type='reflect', in_2d=False):
        super().__init__()

        assert isinstance(filter_size, int) and filter_size > 1, f'blur filter size {filter_size} should be an int > 1!'
        self.channels = channels
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.pad_type = pad_type
        self.in_2d = in_2d

        filter_c = torch.Tensor((np.poly1d([0.5, 0.5]) ** (filter_size - 1)).c)

        if in_2d:
            self.stride = stride if isinstance(stride, tuple) else (stride, stride)
            assert sum(self.stride) > 2, f'stride {stride} should be greater than 1 for at least one dimension!'

            self.padding = tuple(np.repeat(self.padding, 2))
            constant_c = torch.ones(filter_c.shape[0], 1) / filter_c.shape[0]

            if self.stride == (1, 2):
                self.register_buffer(
                    'filter_w', (constant_c * filter_c[None, :])[None, None, :, :].repeat(channels, 1, 1, 1))
            elif self.stride == (2, 1):
                self.register_buffer(
                    'filter_w', (filter_c[:, None] * constant_c.T)[None, None, :, :].repeat(channels, 1, 1, 1))
            else:
                self.register_buffer(
                    'filter_w', (filter_c[:, None] * filter_c[None, :])[None, None, :, :].repeat(channels, 1, 1, 1))
        else:
            assert isinstance(stride, int) and stride > 1, f'stride {stride} should be an int > 1 for 1D blur filter!'
            self.register_buffer('filter_w', filter_c[None, None, :].repeat((channels, 1, 1)))  # [C, 1, filter_size]

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, channels, ...]
        Returns:
            Tensor, [batch_size, channels, ...]
        """

        x = F.pad(x, self.padding, mode=self.pad_type)

        if self.in_2d:
            return F.conv2d(x, self.filter_w, stride=self.stride, groups=self.channels)

        return F.conv1d(x, self.filter_w, stride=self.stride, groups=self.channels)


def get_padding(kernel_size, stride=1, dilation=1):
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


class RMSLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        """
        Args:
            x: Tensor, [..., channels]
        Returns:
            Tensor, [..., channels]
        """
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class ResConnect(nn.Module):
    """ Residual connection """
    def __init__(self, module, module_factor=1., input_factor=1.):
        super(ResConnect, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x):
        return (self.module(x) * self.module_factor) + (x * self.input_factor)


class ResConnectAtt(nn.Module):
    """ Residual connection for attention """
    def __init__(self, module, module_factor=1., input_factor=1.):
        super(ResConnectAtt, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, x, pos):
        x_module, att, content_score, pos_score = self.module(x, pos)

        return (x_module * self.module_factor) + (x * self.input_factor), att, content_score, pos_score


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) """

    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with tensors of different dims, not just 2D CNNs
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, drop_prob=self.drop_prob, training=self.training)


class SEBlock(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        # noinspection PyTypeChecker
        self.se = nn.Sequential(OrderedDict([
            ('gap', nn.AdaptiveAvgPool1d(1)),
            ('fc1', conv1d_unit(input_dim, hidden_dim, kernel_size=1, act=nn.ReLU(), norm=None)),
            ('fc2', conv1d_unit(hidden_dim, input_dim, kernel_size=1, act=nn.Sigmoid(), norm=None))]))

    def forward(self, x):
        return self.se(x) * x


@torch.no_grad()
def dist_concat_all_gather(input_tensor):
    """
    Performs distributed concatenate using all_gather operation on the provided tensors
    *** Warning ***: dist.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(input_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, input_tensor, async_op=False)

    return torch.cat(tensors_gather, dim=0)


class GatherLayerFunc(Function):
    """ Gather tensors from all process, supporting backward propagation """
    @staticmethod
    def forward(ctx, x, **kwargs):
        ctx.save_for_backward(x)
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)

        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_output):
        (x,) = ctx.saved_tensors
        grad_input = torch.zeros_like(x)
        grad_input[:] = grad_output[dist.get_rank()]

        return grad_input


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversalFunc.apply(x, self.alpha)


class GradientReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, alpha, **kwargs):
        # ctx.save_for_backward(alpha)
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, *grad_output):
        # alpha = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output[0]
        return grad_input, None
