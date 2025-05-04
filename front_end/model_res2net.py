""" ECAPA-TDNN """

import torch
import torch.nn as nn
from front_end.model_misc import conv1d_unit, SEBlock
from front_end.model_tdnn import TDNN


class Res2Net(TDNN):
    def __init__(self, scale=8, se=True, se_hidden_dim=128, name='res2net', **kwargs):
        self.scale = scale
        self.se = se
        self.se_hidden_dim = se_hidden_dim
        super().__init__(name=name, **kwargs)

    def create_frame_level_layers(self, input_dim=80):
        return Res2FrameLevelLayers(
            input_dim, self.filters, self.kernel_sizes, self.dilations, scale=self.scale, se=self.se,
            se_hidden_dim=self.se_hidden_dim)


class Res2FrameLevelLayers(nn.Module):
    def __init__(self, input_dim, filters, kernel_sizes, dilations, scale=8, se=True, se_hidden_dim=128):
        super().__init__()
        self.frame_level_layers = nn.ModuleList()
        self.frame_level_layers.append(
            conv1d_unit(input_dim, filters[0], kernel_sizes[0], padding='same', dilation=dilations[0]))

        for i in range(1, len(filters) - 1):
            self.frame_level_layers.append(
                Res2Block(
                    filters[i - 1], filters[i], kernel_sizes[i], dilation=dilations[i], scale=scale, se=se,
                    se_hidden_dim=se_hidden_dim))

        self.frame_level_layers.append(
            conv1d_unit(
                sum(filters[1:-1]), filters[-1], kernel_sizes[-1], padding='same', dilation=dilations[-1]))  # norm=None

        self.output_chnls = filters[-1]  # as pooling input dim

    def forward(self, x):
        x = self.frame_level_layers[0](x)
        x_res2blk0 = self.frame_level_layers[1](x)
        x_res2blk1 = self.frame_level_layers[2](x + x_res2blk0)
        x_res2blk2 = self.frame_level_layers[3](x + x_res2blk0 + x_res2blk1)
        x = self.frame_level_layers[4](torch.cat([x_res2blk0, x_res2blk1, x_res2blk2], dim=1))

        return x
        # return torch.flip(x, dims=[2])


class Res2Block(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, kernel_size=3, dilation=1, scale=8, se=True, se_hidden_dim=128):
        super().__init__()
        self.scale_width = output_dim // scale
        self.conv1d_unit0 = conv1d_unit(input_dim, self.scale_width * scale, kernel_size=1)

        self.conv1d_units_scale = nn.ModuleList()
        for _ in range(scale - 1):
            self.conv1d_units_scale.append(
                conv1d_unit(
                    self.scale_width, self.scale_width, kernel_size=kernel_size, padding='same', dilation=dilation))

        self.conv1d_unit1 = conv1d_unit(self.scale_width * scale, output_dim, kernel_size=1)
        self.se = SEBlock(output_dim, se_hidden_dim) if se else None

    def forward(self, x):
        x_direct = x
        x = self.conv1d_unit0(x)

        x_splits = torch.split(x, self.scale_width, 1)
        x_split = x_splits[0]
        x_split = self.conv1d_units_scale[0](x_split)
        x = x_split

        for i in range(1, len(self.conv1d_units_scale)):
            x_split = x_split + x_splits[i]
            x_split = self.conv1d_units_scale[i](x_split)
            x = torch.concat([x, x_split], 1)
        x = torch.concat([x, x_splits[-1]], 1)

        x = self.conv1d_unit1(x)

        if self.se is not None:
            x = self.se(x)

        return x_direct + x
