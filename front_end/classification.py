""" Loss layers """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLayer(nn.Module):
    def __init__(self, in_nodes=256, n_class=5000):
        super(SoftmaxLayer, self).__init__()
        self.layer = nn.Linear(in_nodes, n_class, bias=False)

    def forward(self, x):
        softmax_logits = self.layer(x)

        return softmax_logits, softmax_logits


class AMSoftmaxLayer(nn.Module):
    def __init__(self, in_nodes=256, n_class=5000, m=0.25, s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.randn(n_class, in_nodes), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, x, label):
        cos_theta = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1)).clamp(-1. + 1e-7, 1. - 1e-7)
        margin = torch.scatter(torch.zeros_like(cos_theta), 1, label.unsqueeze(1), self.m)
        amsoftmax_logits = self.s * (cos_theta - margin)

        return amsoftmax_logits, cos_theta  # return cos_theta as normalized softmax logits
