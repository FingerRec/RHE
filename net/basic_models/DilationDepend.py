#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-07-17 22:21
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : DilationDepend.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

class DilatedDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DilatedDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1, dilation=1)
        self.middle_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1, dilation=2)
        self.local_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1, dilation=4)
    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x)
        middle_range_depen = self.middle_range_depen(spatial_pool_x)
        local_range_depen = self.local_range_depen(spatial_pool_x)
        return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2)