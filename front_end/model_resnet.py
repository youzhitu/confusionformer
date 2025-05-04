""" resnet """

import torch.nn as nn
from front_end.model_misc import conv2d_unit
from front_end.model_tdnn import TDNN


class ResNet(TDNN):
    """ ResNet34-ResBlock: 3-4-6-3, ResNet50-ResBlock3: 3-4-6-3, ResNet101-ResBlock3: 3-4-23-3 """
    def __init__(self, strides='1-2-2-2', n_res_blks='3-4-23-3', bottleneck3=True, name='resnet', **kwargs):
        self.strides = [int(stride) for stride in strides.split('-')]
        self.n_res_blks = [int(n_res_blk) for n_res_blk in n_res_blks.split('-')]
        self.bottleneck3 = True if self.n_res_blks[2] > 6 else bottleneck3
        super().__init__(name=name, **kwargs)

    def create_frame_level_layers(self, input_dim=1):
        return ResFrameLevelLayers(
            1, self.filters, self.kernel_sizes, self.strides, self.n_res_blks, self.feat_dim, self.bottleneck3)


class ResFrameLevelLayers(nn.Module):
    def __init__(self, input_dim, filters, kernel_sizes, strides, num_blocks, feat_dim=80, bottleneck3=False):
        super().__init__()

        self.bottleneck3 = bottleneck3
        self.frame_level_layers = nn.ModuleList()
        self.frame_level_layers.append(conv2d_unit(input_dim, filters[0], kernel_sizes[0], stride=1, padding=1))

        in_filters = [filters[0]] + [4 * f for f in filters[1: len(num_blocks)]] if bottleneck3 else \
            filters[:len(num_blocks)]

        for i in range(1, len(filters)):
            self.frame_level_layers.append(
                self._make_layer(in_filters[i - 1], filters[i], stride=strides[i - 1], num_block=num_blocks[i - 1]))

        stride = 1
        for stride_ in strides:
            stride *= stride_

        self.output_chnls = filters[-1] * 4 * (feat_dim // stride) if bottleneck3 else \
            filters[-1] * (feat_dim // stride)  # as pooling input dim

    def _make_layer(self, input_dim, output_dim, stride, num_block):
        layers = [ResBlock3(input_dim, output_dim, stride=stride) if self.bottleneck3 else
                  ResBlock(input_dim, output_dim, stride=stride)]

        for _ in range(num_block - 1):
            res_blk = ResBlock3(output_dim * 4, output_dim) if self.bottleneck3 else ResBlock(output_dim, output_dim)
            layers.append(res_blk)

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tensor, [batch_size, freq_dim, n_frames]
        Returns:
            x: Tensor, [batch_size, freq_dim, n_frames]
        """

        x = x.unsqueeze(1)

        for layer in self.frame_level_layers:
            x = layer(x)

        return x.view(x.size(0), -1, x.size(3))


class ResBlock(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv2d_unit(input_dim, output_dim, kernel_size, stride=stride, padding=1)
        self.conv2 = conv2d_unit(output_dim, output_dim, kernel_size, stride=1, padding=1, act=None)

        self.shortcut = nn.Sequential()

        if stride != 1 or input_dim != output_dim:
            self.shortcut = conv2d_unit(input_dim, output_dim, 1, stride=stride, padding=0, act=None)

        self.act = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.shortcut(x_skip)
        x = self.act(x)

        return x


class ResBlock3(nn.Module):
    def __init__(self, input_dim=64, output_dim=256, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = conv2d_unit(input_dim, output_dim, 1, stride=1, padding=0)
        self.conv2 = conv2d_unit(output_dim, output_dim, kernel_size, stride=stride, padding=1)
        self.conv3 = conv2d_unit(output_dim, output_dim * 4, 1, stride=1, padding=0, act=None)
        output_dim = output_dim * 4

        self.shortcut = nn.Sequential()

        if stride != 1 or input_dim != output_dim:
            self.shortcut = conv2d_unit(input_dim, output_dim, 1, stride=stride, padding=0, act=None)

        self.act = nn.ReLU()

    def forward(self, x):
        x_skip = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + self.shortcut(x_skip)
        x = self.act(x)

        return x
