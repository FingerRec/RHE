#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-05 22:52
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : MultiDepend.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # close the warning


class MultiDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1)
        self.middle_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1)
        self.small_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1)
        self.local_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        # spatial_pool_x = torch.sigmoid(spatial_pool_x)　
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::(t-1),:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::math.ceil((t-1)/3),:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::math.ceil((t-1)/7),:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,::math.ceil((t-1)/15),:,:])
        '''
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::6,:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::4,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::2,:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,::1,:,:])
        '''
        return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2)


class MultiPyramidBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        # v1 all compress, v2 kernel_size = (2,1,1)
        super(MultiPyramidBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=(1,1,1))
        self.long_pooling = nn.AdaptiveMaxPool3d((1,1,1))
        self.middle_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=(2,1,1))
        self.middle_pooling = nn.AdaptiveMaxPool3d((2,1,1))
        self.small_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=(4,1,1))
        self.small_pooling = nn.AdaptiveMaxPool3d((4,1,1))
        self.local_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=(8,1,1))
        self.local_pooling = nn.AdaptiveMaxPool3d((8,1,1))

    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(self.long_pooling(spatial_pool_x))
        middle_range_depen = self.middle_range_depen(self.middle_pooling(spatial_pool_x))
        small_range_depen = self.small_range_depen(self.small_pooling(spatial_pool_x))
        local_range_depen = self.local_range_depen(self.local_pooling(spatial_pool_x))
        return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1,1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2)


def gen_index(total_len, size):
    '''

    :param total_len: origin length
    :param size: the length need to generate
    :return: index array
    '''
    index = random.randint(total_len // size, size=size)
    for i in range(size):
        index[i] += i * total_len // size
    #print(index)
    return index.tolist()


class RandomMultiDependBlock(nn.Module):
    '''
    random select from long distance -> short distance
    '''
    def __init__(self, in_channel, out_channel):
        super(RandomMultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=1)
        self.middle_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1)
        self.small_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1)
        self.local_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=(2,1,1), stride=1)

    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,gen_index(t, 1),:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,gen_index(t, 2),:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,gen_index(t, 4),:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,gen_index(t, 8),:,:])
        return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2)