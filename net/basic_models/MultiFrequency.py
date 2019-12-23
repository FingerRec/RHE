#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-27 21:19
     # @Author  : Awiny
     # @Site    :
     # @Project : amax-pytorch-i3d
     # @File    : MultiFrequency.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


class MultiFrequencyBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MultiFrequencyBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, dilation=(7,1,1))
        self.middle_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, dilation=(4,1,1))
        self.small_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, dilation=(2,1,1))
        self.local_range_depen = nn.Conv3d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x)
        middle_range_depen = self.middle_range_depen(spatial_pool_x)
        small_range_depen = self.small_range_depen(spatial_pool_x)
        local_range_depen = self.local_range_depen(spatial_pool_x)
        return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
               nn.AdaptiveMaxPool3d((1, 1, 1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2)

class StraightMultiFrequencyBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=8, frames=16):
        super(StraightMultiFrequencyBlock, self).__init__()

        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=in_channel//ratio, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=in_channel//ratio, out_channels=in_channel//ratio, kernel_size=(2,1,1), stride=1, dilation=(frames-1,1,1))
        self.middle_range_depen = nn.Conv3d(in_channels=in_channel//ratio, out_channels=in_channel//ratio, kernel_size=(2,1,1), stride=1, dilation=((frames-1)//2,1,1))
        self.small_range_depen = nn.Conv3d(in_channels=in_channel//ratio, out_channels=in_channel//ratio, kernel_size=(2,1,1), stride=1, dilation=((frames-1)//4,1,1))
        self.local_range_depen = nn.Conv3d(in_channels=in_channel//ratio, out_channels=in_channel//ratio, kernel_size=(2,1,1), stride=1, dilation=((frames-1)//7,1,1))
        self.channel_up = nn.Conv3d(in_channels=in_channel//ratio, out_channels=out_channel, kernel_size=1, stride=1)
        #self.batch3d = torch.nn.BatchNorm3d(in_channel//ratio)
        nn.init.constant_(self.channel_compress.weight, 0)
        nn.init.constant_(self.channel_compress.bias, 0)
        nn.init.constant_(self.channel_up.weight, 0)
        nn.init.constant_(self.channel_up.bias, 0)
        nn.init.constant_(self.long_range_depen.weight, 0)
        nn.init.constant_(self.long_range_depen.bias, 0)
        nn.init.constant_(self.middle_range_depen.weight, 0)
        nn.init.constant_(self.middle_range_depen.bias, 0)
        nn.init.constant_(self.small_range_depen.weight, 0)
        nn.init.constant_(self.small_range_depen.bias, 0)
        nn.init.constant_(self.local_range_depen.weight, 0)
        nn.init.constant_(self.local_range_depen.bias, 0)

    def forward(self, x):
        b, c, t, h, w = x.size()
        compress_x = self.channel_compress(x)
        #compress_x = self.batch3d(compress_x)
        #compress_x = F.relu(compress_x)
        long_range_depen = self.long_range_depen(compress_x)
        middle_range_depen = self.middle_range_depen(compress_x)
        small_range_depen = self.small_range_depen(compress_x)
        local_range_depen = self.local_range_depen(compress_x)
        combine = F.interpolate(long_range_depen, (t,h,w)) + F.interpolate(middle_range_depen, (t,h,w)) + F.interpolate(small_range_depen, (t,h,w)) + F.interpolate(local_range_depen, (t,h,w))
        #combine = F.relu(combine)
        output = self.channel_up(combine)
        #output = F.relu(output)
        return output + x


class TBTAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TBTAM, self).__init__()
        self.d_sample_1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.d_sample_2 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.u_sample_1 = nn.ConvTranspose3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.u_sample_2 = nn.ConvTranspose3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0))
        self.global_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1,1,1), stride=(1,1,1))
        self.sigmod_block = nn.Sigmoid()

    def forward(self, x):
        residual = x
        x = self.d_sample_1(x)
        skip_1 = x
        x = self.d_sample_2(x)
        x = self.global_conv(x)
        x = self.u_sample_1(x)
        x = F.interpolate(x, (skip_1.size()[2], skip_1.size()[3], skip_1.size()[4]), mode='trilinear')
        x += skip_1
        x = self.u_sample_2(x)
        x = self.sigmod_block(x)
        return (1+x)*residual
        #x = F.interpolate(x, (residual.size()[2], residual.size()[3], residual.size()[4]), mode='trilinear')
        #return x + residual


class TimeInceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=8):
        super(TimeInceptionBlock, self).__init__()

        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=in_channel // ratio, kernel_size=1,
                                          stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                          kernel_size=(2, 1, 1), stride=1, dilation=(7, 1, 1), groups=in_channel//ratio)
        self.middle_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                            kernel_size=(2, 1, 1), stride=1, dilation=(5, 1, 1), groups=in_channel//ratio)
        self.small_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                           kernel_size=(2, 1, 1), stride=1, dilation=(3,1,1), groups=in_channel//ratio)
        self.local_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                           kernel_size=(2, 1, 1), stride=1, dilation=1, groups=in_channel//ratio)
        self.channel_up = nn.Conv3d(in_channels=in_channel // ratio, out_channels=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        b, c, t, h, w = x.size()
        compress_x = self.channel_compress(x)
        long_range_depen = self.long_range_depen(compress_x)
        middle_range_depen = self.middle_range_depen(compress_x)
        small_range_depen = self.small_range_depen(compress_x)
        local_range_depen = self.local_range_depen(compress_x)
        combine = F.interpolate(long_range_depen, (t, h, w)) + \
                  F.interpolate(middle_range_depen,(t, h, w)) + \
                  F.interpolate(small_range_depen, (t, h, w)) + \
                  F.interpolate(local_range_depen, (t, h, w))
        output = self.channel_up(combine)
        return output + x

class TimeInception2Block(nn.Module):
    def __init__(self, in_channel, out_channel, ratio=1):
        super(TimeInception2Block, self).__init__()

        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=in_channel // ratio, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                          kernel_size=(7, 1, 1), stride=1, groups=in_channel//ratio)
        self.middle_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                            kernel_size=(5, 1, 1), stride=1, groups=in_channel//ratio)
        self.small_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                           kernel_size=(3, 1, 1), stride=1, groups=in_channel//ratio)
        self.local_range_depen = nn.Conv3d(in_channels=in_channel // ratio, out_channels=in_channel // ratio,
                                           kernel_size=(1, 1, 1), stride=1, groups=in_channel//ratio)
        self.channel_up = nn.Conv3d(in_channels=in_channel // ratio, out_channels=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        b, c, t, h, w = x.size()
        compress_x = self.channel_compress(x)
        long_range_depen = self.long_range_depen(compress_x)
        middle_range_depen = self.middle_range_depen(compress_x)
        small_range_depen = self.small_range_depen(compress_x)
        local_range_depen = self.local_range_depen(compress_x)
        combine = F.interpolate(long_range_depen, (t, h, w)) + \
                  F.interpolate(middle_range_depen,(t, h, w)) + \
                  F.interpolate(small_range_depen, (t, h, w)) + \
                  F.interpolate(local_range_depen, (t, h, w))
        output = self.channel_up(combine)
        return output + x
