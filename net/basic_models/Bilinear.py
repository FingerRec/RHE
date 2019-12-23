#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-05 22:55
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : Bilinear.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import torch.nn as nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
class BilinearClassifyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BilinearClassifyBlock, self).__init__()
        self.compress = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.fc = torch.nn.Linear(
            in_features=out_channels * out_channels, out_features=out_channels, bias=True)
    def forward(self, x):
        x = self.compress(x)
        b, c, t, h, w = x.size()
        X = torch.reshape(x, (b, c, t * h * w))
        Y = torch.reshape(x, (b, c, t * h * w))
        res = torch.bmm(X, torch.transpose(Y, 1, 2)) / (t * h * w)
        assert res.size() == (b, c, c)
        res = torch.reshape(res, (b, c * c))

        res = torch.sqrt(res + 1e-5)
        res = torch.nn.functional.normalize(res)

        # Classification.
        res = self.fc(res)
        return res