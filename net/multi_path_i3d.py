#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-08 18:32
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : multi_path_i3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import scipy.io
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
#========================================================================================
#This network is designed to capture different range dependencies and cobine them.
#With dilated conv and downsample, i want to down the number of parameters.
#The network are divided into 3 parllel network. and across information between them.
#1:64frame input, 56 x 56 input, long range temporal dependencies, call s
#2:16frame input, 112x112, middle range temporal dependencies, call m
#3:4frame input, 224x224, shortest temporal dependencies, call l
#after these network, use tpp to combine them and put it into fc layer
#========================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
from math import exp
import os
import sys
from collections import OrderedDict
from net.basic_models.se_module import SELayer3D
# from models.self_attention import Self_Attn

class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init

class Unit3Dpy(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation='relu',
                 padding='SAME',
                 use_bias=False,
                 use_bn=True):
        super(Unit3Dpy, self).__init__()

        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == 'VALID':
            padding_shape = 0
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if padding == 'SAME':
            if not simplify_pad:
                self.pad = torch.nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        if activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
        return out


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 dilation=1,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                dilation=dilation,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)
            #here another implement is nn.BatchNorm3d(self.out_channels)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x

class TemporalPyramidPool3D_2(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(TemporalPyramidPool3D_2, self).__init__()
        self.out_side = out_side
        self.out_t = out_side[0] + out_side[1] + out_side[2]

    def forward(self, x):
        out = None
        for n in self.out_side:
            t_r, w_r, h_r = map(lambda s: math.ceil(s / n), x.size()[2:])  # Receptive Field Size
            s_t, s_w, s_h = map(lambda s: math.floor(s / n), x.size()[2:])  # Stride
            max_pool = nn.MaxPool3d(kernel_size=(t_r, w_r, h_r), stride=(s_t, s_w, s_h))
            y = max_pool(x)
            avg_pool = nn.AdaptiveAvgPool3d((y.size(2), 1, 1))
            y = avg_pool(y)
            # print(y.size())
            if out is None:
                out = y.view(y.size()[0], y.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], -1, 1, 1)), 2)
        return out

class TemporalPyramidPool3D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(TemporalPyramidPool3D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            avg_pool = nn.AdaptiveMaxPool3d((n, 1, 1))
            y = avg_pool(x)
            if out is None:
                out = y.view(y.size()[0], y.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], -1, 1, 1)), 2)
        return out

class SpatialPyramidPool3D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(SpatialPyramidPool3D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            max_pool = nn.AdaptiveMaxPool3d((1, n, n))
            y = max_pool(x)
            if out is None:
                out = y.view(y.size()[0], y.size()[1], 1, n*n, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], 1, n*n, 1)), 3)
        return out

class InplaceShift(torch.autograd.Function):
    # Special thanks to @raoyongming for the help to this function
    @staticmethod
    def forward(ctx, input, fold):
        # not support higher order gradient
        # input = input.detach_()
        ctx.fold_ = fold
        n, t, c, h, w = input.size()
        buffer = input.data.new(n, t, fold, h, w).zero_()
        buffer[:, :-1] = input.data[:, 1:, :fold]
        input.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, 1:] = input.data[:, :-1, fold: 2 * fold]
        input.data[:, :, fold: 2 * fold] = buffer
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.detach_()
        fold = ctx.fold_
        n, t, c, h, w = grad_output.size()
        buffer = grad_output.data.new(n, t, fold, h, w).zero_()
        buffer[:, 1:] = grad_output.data[:, :-1, :fold]
        grad_output.data[:, :, :fold] = buffer
        buffer.zero_()
        buffer[:, :-1] = grad_output.data[:, 1:, fold: 2 * fold]
        grad_output.data[:, :, fold: 2 * fold] = buffer
        return grad_output, None

class TemporalShuffle(nn.Module):
    def __init__(self, fold_div=8):
        super(TemporalShuffle, self).__init__()
        self.fold_div = fold_div

    def forward(self, x):
        b, t, c, h, w = x.size()
        fold = c // self.fold_div
        out = InplaceShift.apply(x, fold)
        return out.view(b, t, c, h, w)

class MultiDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel, concat=False, fc=False):
        super(MultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = Unit3D(in_channels=in_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='channel_compress')
        self.long_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='long_range_depen')
        self.middle_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='middle_range_depen')
        self.small_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='small_range_depen')
        self.local_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='local_range_depen')
        '''
        self.single_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='single_range_depen')
        '''
        self.concat = concat
        self.fc = fc
        if self.fc:
            self.fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(3 * out_channel, 128),
                nn.ReLU(),
                nn.Linear(128, out_channel),
            )
        #self.dropout_probality = 0.05
    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        #spatial_pool_x = nn.Dropout(self.dropout_probality)(spatial_pool_x)
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::(t-1),:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::(t-1)//2,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::(t-1)//4,:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,::(t-1)//7,:,:])
        #single_range_depen = self.single_range_depen(spatial_pool_x[:, :, ::1, :, :])
        '''
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::7,:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::4,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::2,:,:])
        local_range_depen = self.local_range_depen(spatial_pool_x[:,:,::1,:,:])
        '''
        if self.fc:
            out = torch.cat((nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2)), dim = 1)
            return self.fc_fusion(out)
        elif self.concat:
            return torch.cat((nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2), nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2)), dim = 1)
        else:
            return nn.AdaptiveMaxPool3d((1, 1, 1))(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
                   nn.AdaptiveMaxPool3d((1, 1, 1))(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
                   nn.AdaptiveMaxPool3d((1, 1, 1))(small_range_depen).squeeze(2).squeeze(2).squeeze(2) + \
                   nn.AdaptiveMaxPool3d((1, 1, 1))(local_range_depen).squeeze(2).squeeze(2).squeeze(2) #+ \
                   #nn.AdaptiveMaxPool3d((1, 1, 1))(single_range_depen).squeeze(2).squeeze(2).squeeze(2)
class DichMultiDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel, t):
        super(DichMultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
               kernel_size=(1, 1, 1),
               stride=(1, 1, 1),
               padding=0)
        self.multi_scale_depen = nn.ModuleList()
        while t/2 > 0:
            if t == 1:
                self.multi_scale_depen.append(nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                   kernel_size=(1, 1, 1), # 2 or 1?
                   stride=(1, 1, 1),
                   padding=0))
            else:
                self.multi_scale_depen.append(nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                   kernel_size=(2, 1, 1), # 2 or 1?
                   stride=(1, 1, 1),
                   padding=0))
            t = t // 2
    def forward(self, x):
        b, c, t, h, w = x.size() # 4 x 2048 x 1
        #print(x.size())
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        out = None
        stride = 1
        for depend in self.multi_scale_depen:
            if out is None:
                out = nn.AdaptiveMaxPool3d((1, 1, 1))(depend(spatial_pool_x[:,:,::stride,:,:])).squeeze(2).squeeze(2).squeeze(2)
            else:
                out += nn.AdaptiveMaxPool3d((1, 1, 1))(depend(spatial_pool_x[:,:,::stride,:,:])).squeeze(2).squeeze(2).squeeze(2)
            stride *= 2
            if stride == t:
                stride -= 1
        return out
class TemporalDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = Unit3D(in_channels=in_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='channel_compress')
        self.tpp = TemporalPyramidPool3D((1,2,4,8))
        self.temporal_conv = Unit3D(in_channels=out_channel, output_channels=out_channel,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv')
    def forward(self, x):
        b, c, t, h, w = x.size()
        compress = self.channel_compress(x)
        tpp = self.tpp(compress)
        out = self.temporal_conv(tpp)
        return out.view(b, out.size(1))

class HeavyMultiDependBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HeavyMultiDependBlock, self).__init__()
        self.out_channel = out_channel
        self.channel_compress = Unit3D(in_channels=in_channel, output_channels=out_channel,
               kernel_shape=[1, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='channel_compress')
        self.long_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='long_range_depen')
        self.middle_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='middle_range_depen')
        self.small_range_depen = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[2, 1, 1],
               stride=(1, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='small_range_depen')
        self.tpp_1 = TemporalPyramidPool3D((1,2,4))
        self.fusion_1 = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[7, 1, 1],
               stride=(7, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='long_range_depen')
        self.tpp_2 = TemporalPyramidPool3D((1,2,4))
        self.fusion_2 = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[7, 1, 1],
               stride=(7, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='middle_range_depen')
        self.tpp_3 = TemporalPyramidPool3D((1,2,4))
        self.fusion_3 = Unit3D(in_channels=out_channel, output_channels=out_channel,
               kernel_shape=[7, 1, 1],
               stride=(7, 1, 1),
               padding=0,
               activation_fn=None,
               use_batch_norm=False,
               use_bias=True,
               name='small_range_depen')
    def forward(self, x):
        b, c, t, h, w = x.size()
        spatial_pool_x = nn.AdaptiveAvgPool3d((t,1,1))(x)/2 + nn.AdaptiveMaxPool3d((t,1,1))(x)/2
        spatial_pool_x = self.channel_compress(spatial_pool_x)
        long_range_depen = self.long_range_depen(spatial_pool_x[:,:,::4,:,:])
        middle_range_depen = self.middle_range_depen(spatial_pool_x[:,:,::2,:,:])
        small_range_depen = self.small_range_depen(spatial_pool_x[:,:,::1,:,:])
        long_range_depen = self.tpp_1(long_range_depen)
        middle_range_depen = self.tpp_2(middle_range_depen)
        small_range_depen = self.tpp_3(small_range_depen)
        return self.fusion_1(long_range_depen).squeeze(2).squeeze(2).squeeze(2) + self.fusion_2(middle_range_depen).squeeze(2).squeeze(2).squeeze(2) + self.fusion_3(small_range_depen).squeeze(2).squeeze(2).squeeze(2)
        '''
        out = torch.cat((long_range_depen, middle_range_depen, small_range_depen), dim=2)
        out = self.tpp(out)
        out = self.fusion(out)
        return out
        '''
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        '''
        self.tba = Unit3D(in_channels=in_channels, output_channels=in_channels//16, kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_Temporal_1/Conv3d_0a_1x1')
        self.tbb = Unit3D(in_channels=in_channels//16, output_channels=in_channels//16, kernel_shape=[3, 1, 1], padding=0,
                          name=name + '/Branch_Temporal_2/Conv3d_0a_1x1')
        self.tbc = Unit3D(in_channels=in_channels//16, output_channels=out_channels[0]+out_channels[2]+out_channels[4]+out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_Temporal_3/Conv3d_0a_1x1')
        '''
        #self.temporal_shift = TemporalShuffle(fold_div=16)
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return  torch.cat([b0, b1, b2, b3], dim=1)
        """
        out = torch.cat([b0, b1, b2, b3], dim=1)
        b, c, t, h, w = x.size()
        if t > 16:
            ts_1 = self.temporal_shift(out)
            return out + ts_1
        else:
            '''
            tb0 = self.tba(x)
            tb1 = self.tbb(tb0)
            tb2 = self.tbc(tb1)
            '''
            return out
        """
class TemporalInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(TemporalInceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[1, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[1, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

class MultiPathI3d(nn.Module):
    def __init__(self, num_classes=400, spatial_squeeze=True, in_channels=3, dropout_prob=0.5):

        super(MultiPathI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self.logits = None

        self.Conv3d_1a_7x7 = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name='conv3d_1a_7_7')

        self.MaxPool3d_2a_3x3 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        self.Conv3d_2b_1x1 = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,name='Conv3d_2b_1x1')
        self.Conv3d_2c_3x3 = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1, name='Conv3d_2c_3x3')
        self.maxpool_1 = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)
        self.Mixed_3b = InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')
        self.Mixed_3c = InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')
        self.maxpool_2 = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)
        self.Mixed_4b = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')
        self.Mixed_4c = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')
        self.Mixed_4d = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')
        self.Mixed_4e = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')
        self.Mixed_4f = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')
        self.maxpool_3 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        self.Mixed_5b = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], 'Mixed_5b')
        self.Mixed_5c = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], 'Mixed_5c')
        '''
        self.CommonPath = nn.Sequential(OrderedDict([('Conv3d_2b_1x1',Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,name='Conv3d_2b_1x1')),
                                                     ('Conv3d_2c_3x3',Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,name='Conv3d_2c_3x3')),
                                                     ('maxpool_1',MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),padding=0)),
                                                     ('Mixed_3b', InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')),
                                                     ('Mixed_3c', InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')),
                                                     ('maxpool_2',MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_4b', InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')),
                                                     ('Mixed_4c', InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')),
                                                     ('Mixed_4d', InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')),
                                                     ('Mixed_4e', InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')),
                                                     ('Mixed_4f', InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')),
                                                     ('maxpool_3',MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_5b', InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],'Mixed_5b')),
                                                     ('Mixed_5c', InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],'Mixed_5c'))
                                                     ]))
        '''
        '''
        #small image size, 
        self.SamllPath = nn.Sequential(OrderedDict([('Conv3d_2b_1x1',Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,name='Conv3d_2b_1x1')),
                                                     ('Conv3d_2c_3x3',Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,name='Conv3d_2c_3x3')),
                                                     ('maxpool_1',MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),padding=0)),
                                                     ('Mixed_3b', InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')),
                                                     ('Mixed_3c', InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')),
                                                     ('maxpool_2',MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_4b', InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')),
                                                     ('Mixed_4c', InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')),
                                                     ('Mixed_4d', InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')),
                                                     ('Mixed_4e', InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')),
                                                     ('Mixed_4f', InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')),
                                                     ('maxpool_3',MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_5b', InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],'Mixed_5b')),
                                                     ('Mixed_5c', InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],'Mixed_5c'))
                                                     ]))
        #middle image size
        self.MiddlePath = nn.Sequential(OrderedDict([('Conv3d_2b_1x1',Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,name='Conv3d_2b_1x1')),
                                                     ('Conv3d_2c_3x3',Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,name='Conv3d_2c_3x3')),
                                                     ('maxpool_1',MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),padding=0)),
                                                     ('Mixed_3b', InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')),
                                                     ('Mixed_3c', InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')),
                                                     ('maxpool_2',MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_4b', InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')),
                                                     ('Mixed_4c', InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')),
                                                     ('Mixed_4d', InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')),
                                                     ('Mixed_4e', InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')),
                                                     ('Mixed_4f', InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')),
                                                     ('maxpool_3',MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_5c', InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],'Mixed_5c'))
                                                     ]))
        self.LargePath = nn.Sequential(OrderedDict([('Conv3d_2b_1x1',Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,name='Conv3d_2b_1x1')),
                                                     ('Conv3d_2c_3x3',Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,name='Conv3d_2c_3x3')),
                                                     ('maxpool_1',MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),padding=0)),
                                                     ('Mixed_3b', InceptionModule(192, [64, 96, 128, 16, 32, 32], 'Mixed_3b')),
                                                     ('Mixed_3c', InceptionModule(256, [128, 128, 192, 32, 96, 64], 'Mixed_3c')),
                                                     ('maxpool_2',MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_4b', InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 'Mixed_4b')),
                                                     ('Mixed_4c', InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 'Mixed_4c')),
                                                     ('Mixed_4d', InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 'Mixed_4d')),
                                                     ('Mixed_4e', InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 'Mixed_4e')),
                                                     ('Mixed_4f', InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 'Mixed_4f')),
                                                     ('maxpool_3',MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),padding=0)),
                                                     ('Mixed_5b', InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],'Mixed_5b')),
                                                     ('Mixed_5c', InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],'Mixed_5c'))
                                                     ]))
        '''
        '''
        self.MiddleMotionPath = nn.Sequential(OrderedDict([('down_channel', Unit3D(in_channels=832, output_channels=64, kernel_shape=[1, 1, 1],
                                                             padding=0, name='down_channel_1')),
                                                     ('spatial_conv_1', Unit3D(in_channels=64, output_channels=64,
                                                                              kernel_shape=[1, 3, 3],stride=(1,1,3), padding=0,
                                                                              name='spatial_conv_1')),
                                                     ('spatial_conv_2', Unit3D(in_channels=64, output_channels=64,
                                                                            kernel_shape=[1,3,3],stride=(1,2,2), padding=0,
                                                                            name='spatial_conv_2')),
                                                   ('up_channel', Unit3D(in_channels=64, output_channels=832,
                                                                           kernel_shape=[1, 1, 1],
                                                                           padding=0, name='up_channel_1'))
                                                    ]))
        '''
        '''
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        '''
        '''
        self.path_s_up_logits = Unit3D(in_channels=480, output_channels=1024,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             use_batch_norm=True,
                             use_bias=True,
                             name='path_l_de_logits')
        self.path_m_up_logits = Unit3D(in_channels=832, output_channels=1024,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             use_batch_norm=True,
                             use_bias=True,
                             name='path_l_de_logits')
        '''
        '''
        self.path_s_de_logits = Unit3D(in_channels=480, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15,1,1),
                             padding=0,
                             activation_fn=F.relu,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_l_de_logits')
        self.path_m_de_logits = Unit3D(in_channels=832, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15,1,1),
                             padding=0,
                             activation_fn=F.relu,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_l_de_logits')
        self.path_l_de_logits = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15,1,1),
                             padding=0,
                             activation_fn=F.relu,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_l_de_logits')
        '''
        '''
        self.path_sp_l_de_logits = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[21, 1, 1],
                             stride=(21,1,1),
                             padding=0,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_l_de_logits')
        '''
        '''
        self.all_conv = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[5, 1, 1],
                             stride=(5, 1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_l_de_logits')
        '''
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_probality = dropout_prob
        #self.channel_attention = SELayer3D(1024, reduction=16)
        #self.channel_attention_2 = SELayer3D(1024, reduction=16)
        #self.tpp_s = TemporalPyramidPool3D_2(out_side=(1, 2, 4))
        #self.tpp_m = TemporalPyramidPool3D(out_side=(1, 2, 4))
        #self.spp_l = SpatialPyramidPool3D(out_side=(1,2,4))
        #self.softmax = torch.nn.Softmax(dim=1)
        #self.fc_out = nn.Linear(400, self._num_classes, bias=True)
        #self.three_fc = nn.Linear(1200, self._num_classes, bias=True)
        #self.tpp_l = TemporalPyramidPool3D(out_side=(1, 2, 4))
        #========================three path networks======================================
        #here we not use latteral connection first
        #=================================================================================
        """
        #first do temporal compress to compress channel, then use another path
        """
        '''
        self.Middle_Temporal_Compress = Unit3D(in_channels=832, output_channels=832//8, kernel_shape=[1, 1, 1],
                                               padding=0, name='middle_down_channel')
        self.Middle_Temporal_Mixed_5b = TemporalInceptionModule(256 // 8 + 320 // 8 + 128 // 8 + 128 // 8,
                                                         [256 // 8, 160 // 8, 320 // 8, 32 // 8, 128 // 8, 128 // 8],
                                                         'Temporal_Mixed_5b')
        self.Middle_Temporal_Mixed_5c = TemporalInceptionModule(256 // 8 + 320 // 8 + 128 // 8 + 128 // 8,
                                                         [384 // 8, 192 // 8, 384 // 8, 48 // 8, 128 // 8, 128 // 8],
                                                         'Temporal_Mixed_5c')
        self.Middle_Temporal_Up = Unit3D(in_channels=128, output_channels=1024,
                                               kernel_shape=[1, 1, 1],
                                               stride=(1,1,1),
                                               padding=0, name='middle_up_channel')
        '''
        #self.Middle_One_Step = Unit3D(in_channels=832, output_channels=400, kernel_shape=[3,3,3],stride=(2,2,2), padding=0, name='middle_one_step')
        '''
        self.Middle_Temporal_Compress = Unit3D(in_channels=832, output_channels=832 // 8, kernel_shape=[1, 1, 1],
                                               padding=0, name='middle_down_channel')
        '''
        '''
        self.Small_Temporal_Up = Unit3D(in_channels=480, output_channels=1024,
                                               kernel_shape=[1, 1, 1],
                                               padding=0, name='small_up_channel')
        self.Small_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4))
        self.Small_Pool = MaxPool3dSamePadding(kernel_size=[4, 4, 4], stride=(4, 4, 4), padding=0)

        self.Middle_Temporal_Up = Unit3D(in_channels=832, output_channels=1024,
                                               kernel_shape=[1, 1, 1],
                                               padding=0, name='middle_up_channel')
        self.Middle_Pool = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)
        self.Middle_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4))
        self.temporal_compress = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[21, 1, 1],
                             stride=(21, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='middle_temporal_conv')
        #self.temporal_compress = torch.nn.Linear(1200, 400)
        '''
        '''
        self.Middle_Tpp = TemporalPyramidPool3D(out_side=(16,))
        self.Latter_Tpp = TemporalPyramidPool3D(out_side=(8,))
        self.Temporal_Conv = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[24, 1, 1],
                             stride=(24, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='middle_temporal_conv')
        self.Latter_Temporal_Compress = Unit3D(in_channels=1024, output_channels=400, kernel_shape=[1, 1, 1],
                                               padding=0, name='middle_down_channel')
        '''
        '''
        self.Middle_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.Middle_Conv = Unit3D(in_channels=128, output_channels=128,
                             kernel_shape=[15, 1, 1],
                             stride=(15, 1,1),
                              activation_fn=None,
                              use_batch_norm=False,
                             padding=0,
                             use_bias=True,
                             name='middle_temporal_conv')
        '''
        '''
        self.Lateral_Smooth = Unit3D(in_channels=400, output_channels=400, kernel_shape=[3, 3, 3],
                                     stride=(1,1,1),
                                               padding=0, name='lateral_smooth')
        self.Latter_Temporal_Compress = Unit3D(in_channels=1024, output_channels=400, kernel_shape=[1, 1, 1],
                                               padding=0, name='latter_down_channel')
        self.Latter_Lateral_Compress = Unit3D(in_channels=400+400, output_channels=400, kernel_shape=[1, 1, 1],
                                              activation_fn=None,
                                              use_batch_norm=False,
                                              padding=0, name='lateral_down_channel')
        '''
        #=========================================Only Temporal Path================================================
        '''
        self.Latter_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.Latter_Conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv')
        '''
        '''
        self.MiddleMotionPath = nn.Sequential(
            OrderedDict([('middle_down_channel', Unit3D(in_channels=832, output_channels=832//8, kernel_shape=[1, 1, 1],
                                                 padding=0, name='middle_down_channel')),
                         ('middle_spatial_conv_1', TemporalInceptionModule(256 // 8 + 320 // 8 + 128 // 8 + 128 // 8,
                                                         [256 // 8, 160 // 8, 320 // 8, 32 // 8, 128 // 8, 128 // 8],
                                                         'Temporal_Mixed_5b')),
                         ('middle_spatial_conv_2', TemporalInceptionModule(256 // 8 + 320 // 8 + 128 // 8 + 128 // 8,
                                                         [384 // 8, 192 // 8, 384 // 8, 48 // 8, 128 // 8, 128 // 8],
                                                         'Temporal_Mixed_5c')),
                         ('middle_up_channel', Unit3D(in_channels=128, output_channels=400,
                                               kernel_shape=[1, 1, 1],
                                               padding=0, name='middle_up_channel')),
                         ('middle_temporal_pooling', TemporalPyramidPool3D(out_side=(1, 2, 4, 8))),
                         ('middle_temporal_conv', Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15, 1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='middle_temporal_conv'))
                         ]))
        self.LatterMotionPath = nn.Sequential(
            OrderedDict([('latter_temporal_pooling', TemporalPyramidPool3D(out_side=(1, 2, 4, 8))),
                         ('latter_temporal_conv', Unit3D(in_channels=1024, output_channels=400,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv'))
            ])
        )
        '''
        #=======================================Two Branch Net=======================================
        '''
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self.Latter_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.Latter_Conv = Unit3D(in_channels=1024, output_channels=400,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv')
        self.fc_out = nn.Linear(400, self._num_classes, bias=True)
        '''
        #========================RCNN lat layers and smooth layers======================================
        '''
        # Top layer
        self.RCNN_toplayer = nn.Conv3d(1024, 400, kernel_size=1, stride=1, padding=0)  # reduce channel

        # Smooth layers
        #self.RCNN_smooth1 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth1 = nn.Conv3d(400, 400, kernel_size=3, stride=1, padding=1)
        self.RCNN_smooth2 = nn.Conv3d(400, 400, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        #self.RCNN_latlayer1 = nn.Conv3d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer1 = nn.Conv3d(832, 400, kernel_size=1, stride=1, padding=0)
        self.RCNN_latlayer2 = nn.Conv3d(480, 400, kernel_size=1, stride=1, padding=0)
        self.RCNN_tpp_1 = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.RCNN_tpp_2 = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.RCNN_tpp_3 = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.RCNN_compress_1 = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15, 1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='RCNN_compress_1')
        self.RCNN_compress_2 = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15, 1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='RCNN_compress_2')
        self.RCNN_compress_3 = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[15, 1, 1],
                             stride=(15, 1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='RCNN_compress_3')
        self.RCNN_compress = Unit3D(in_channels=400*3, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1, 1, 1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='RCNN_compress')
        '''
        #====================================learnable parameters====================================
        #self.spatial_weight = nn.Parameter(torch.FloatTensor([0.5]))
        #self.temporal_weight = nn.Parameter(torch.FloatTensor([0.5]))
        #self.small_weight = nn.Parameter(torch.FloatTensor([0.3333333]))
        #self.middle_weight = nn.Parameter(torch.FloatTensor([0.333333]))
        #self.large_weight = nn.Parameter(torch.FloatTensor([0.333333]))
        #self.lateral_s_l = nn.Parameter(torch.FloatTensor([0.5]))
        #self.lateral_m_l = nn.Parameter(torch.FloatTensor([0.5]))

        #====================================Multi Stride Temporal Convolutional========================
        '''
        self.long_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.middle_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.small_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        '''
        #====================================All Stride Temporal Convolutional========================
        '''
        self.long_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.middle_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.small_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.local_range_conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.local_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        '''
        #=====================================Temporal Shift========================================
        '''
        self.middle_temporal_shift  = TemporalShuffle()
        self.small_temporal_shift = TemporalShuffle()
        self.latter_temporal_shift = TemporalShuffle()
        '''
        #=======================================Multi Stride Multi Path Network======================
        '''
        self.s_long_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.s_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.s_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.s_middle_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.s_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.s_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.s_small_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.s_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.s_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.m_long_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.m_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.m_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.m_middle_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.m_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.m_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.m_small_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.m_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.m_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.l_long_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.l_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.l_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.l_middle_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.l_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.l_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.l_small_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.l_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.l_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        '''
        #=======================================TPP + Multi Stride Multi Path Network==========================
        '''
        self.s_long_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.s_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.s_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.s_middle_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.s_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.s_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.s_small_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.s_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.s_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.m_long_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.m_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.m_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.m_middle_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.m_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.m_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.m_small_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.m_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.m_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.l_long_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.l_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.l_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.l_middle_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.l_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.l_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.l_small_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.l_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.l_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.Latter_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.Latter_Conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv')
        '''
        #=================================Multi Stride Multi Path Compress Network======================
        self.s_depend = MultiDependBlock(480, self._num_classes, concat=False, fc=False)
        self.m_depend = MultiDependBlock(832, self._num_classes, concat=False, fc=False)
        self.l_depend = MultiDependBlock(1024, self._num_classes, concat=False, fc=False)
        #self.s_depend= DichMultiDependBlock(480, num_classes, 8)
        #self.m_depend = DichMultiDependBlock(832, num_classes, 4)
        #self.l_depend = DichMultiDependBlock(1024, num_classes, 2)
        #self.main_depend = MultiDependBlock(1024, self._num_classes, concat=False, fc=False)
        #self.l_s_compress = nn.Conv3d(in_channels=1024, out_channels=480,kernel_size=1,stride=1)
        #self.l_m_compress = nn.Conv3d(in_channels=1024, out_channels=832, kernel_size=1, stride=1)
        self.concat = False
        self.fc_fusion = False
        if self.concat:
            self.fc = nn.Linear(self._num_classes*9, self._num_classes)
        '''
        self.Latter_Tpp = TemporalPyramidPool3D(out_side=(1, 2, 4, 8))
        self.Latter_Conv = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[15, 1, 1],
                                                stride=(15, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='latter_temporal_conv')
        '''
        #=================================Multi Stride Dense Information Network========================
        '''
        #self.depend_64_1 = MultiDependBlock(64, self._num_classes)
        #self.depend_64_2 = MultiDependBlock(64, self._num_classes)
        #self.depend_192 = MultiDependBlock(192, self._num_classes)
        #self.depend_256 = MultiDependBlock(256, self._num_classes)
        #self.depend_480 = MultiDependBlock(480, self._num_classes)
        #self.depend_512_1 = MultiDependBlock(512, self._num_classes)
        #self.depend_512_2 = MultiDependBlock(512, self._num_classes)
        #self.depend_512_3 = MultiDependBlock(512, self._num_classes)
        self.depend_528 = MultiDependBlock(528, self._num_classes)
        self.depend_832_1 = MultiDependBlock(832, self._num_classes)
        self.depend_832_2 = MultiDependBlock(832, self._num_classes)
        self.depend_1024 = MultiDependBlock(1024, self._num_classes)
        '''
        #======================================Multi TPP path===================================
        '''
        self.s_depend = TemporalDependBlock(480, self._num_classes)
        self.m_depend = TemporalDependBlock(832, self._num_classes)
        self.l_depend = TemporalDependBlock(1024, self._num_classes)
        '''
        #=================================Multi Stride Multi Path Heavy Network======================
        '''
        self.s_depend = HeavyMultiDependBlock(480, self._num_classes)
        self.m_depend = HeavyMultiDependBlock(832, self._num_classes)
        self.l_depend = HeavyMultiDependBlock(1024, self._num_classes)
        '''
        # =======================================Multi Stride Multi Path Concat Network======================
        '''
        self.s_long_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.s_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.s_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.s_middle_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.s_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.s_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.s_small_range_compress = Unit3D(in_channels=480, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.s_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.s_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.m_long_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.m_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.m_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.m_middle_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.m_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.m_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.m_small_range_compress = Unit3D(in_channels=832, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.m_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.m_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.l_long_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_compress')
        self.l_long_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='long_range_conv')
        self.l_long_range_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.l_middle_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_compress')
        self.l_middle_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='middle_range_conv')
        self.l_middle_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.l_small_range_compress = Unit3D(in_channels=1024, output_channels=self._num_classes,
                                                kernel_shape=[1, 1, 1],
                                                stride=(1, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_compress')
        self.l_small_range_conv = Unit3D(in_channels=self._num_classes, output_channels=self._num_classes,
                                                kernel_shape=[2, 1, 1],
                                                stride=(2, 1, 1),
                                                padding=0,
                                                activation_fn=None,
                                                use_batch_norm=False,
                                                use_bias=True,
                                                name='small_range_conv')
        self.l_small_range_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_out = nn.Linear(self._num_classes * 3, self._num_classes)
        '''
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, T, H, W = y.size()
        return F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=True)/2 + y/2
        #return F.upsample(x, size=(T, H, W), mode='trilinear') + y

    def constrain(self, x):
        alpha = 0.2
        beta = 5
        return 1/(beta+exp(-x)) + alpha
    #==============================================Multi Stride Temporal Poling=======================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        long_range = x[:,:,::7,:,:]
        long_range = self.long_range_conv(long_range)
        long_range = self.long_range_pool(long_range)
        middle_range = x[:, :, ::3, :, :]
        middle_range = self.middle_range_conv(middle_range)
        middle_range = self.middle_range_pool(middle_range)
        small_range = x[:, :, ::2, :, :]
        small_range = self.small_range_conv(small_range)
        small_range = self.small_range_pool(small_range)
        fc_out = long_range + middle_range + small_range
        fc_out = fc_out.squeeze(2).squeeze(2).squeeze(2)
        return fc_out
    '''
    #==============================================All Stride Temporal Poling=======================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        long_range = x[:,:,::7,:,:]
        long_range = nn.Dropout(self.dropout_probality)(long_range)
        long_range = self.long_range_conv(long_range)
        long_range = self.long_range_pool(long_range)
        middle_range = x[:, :, ::3, :, :]
        middle_range = nn.Dropout(self.dropout_probality)(middle_range)
        middle_range = self.middle_range_conv(middle_range)
        middle_range = self.middle_range_pool(middle_range)
        small_range = x[:, :, ::2, :, :]
        small_range = nn.Dropout(self.dropout_probality)(small_range)
        small_range = self.small_range_conv(small_range)
        small_range = self.small_range_pool(small_range)
        local_range = x[:, :, ::1, :, :]
        local_range = nn.Dropout(self.dropout_probality)(local_range)
        local_range = self.small_range_conv(local_range)
        local_range = self.small_range_pool(local_range)
        fc_out = long_range + middle_range + small_range + local_range
        fc_out = fc_out.squeeze(2).squeeze(2).squeeze(2)
        return fc_out
    '''
    # ==============================================Multi Stride Multi Path Pooling=======================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        path_s = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        s_long_range = path_s[:,:,::7,:,:]
        s_long_range = nn.Dropout(self.dropout_probality)(s_long_range)
        s_long_range = self.s_long_range_compress(s_long_range)
        s_long_range = self.s_long_range_conv(s_long_range)
        s_long_range = self.s_long_range_pool(s_long_range)
        s_middle_range = path_s[:, :, ::4, :, :]
        s_middle_range = nn.Dropout(self.dropout_probality)(s_middle_range)
        s_middle_range = self.s_middle_range_compress(s_middle_range)
        s_middle_range = self.s_middle_range_conv(s_middle_range)
        s_middle_range = self.s_middle_range_pool(s_middle_range)
        s_small_range = path_s[:, :, ::2, :, :]
        s_small_range = nn.Dropout(self.dropout_probality)(s_small_range)
        s_small_range = self.s_small_range_compress(s_small_range)
        s_small_range = self.s_small_range_conv(s_small_range)
        s_small_range = self.s_small_range_pool(s_small_range)
        s_fc_out = s_long_range + s_middle_range + s_small_range
        s_fc_out = s_fc_out.squeeze(2).squeeze(2).squeeze(2)
        m_long_range = path_m[:,:,::7,:,:]
        m_long_range = nn.Dropout(self.dropout_probality)(m_long_range)
        m_long_range = self.m_long_range_compress(m_long_range)
        m_long_range = self.m_long_range_conv(m_long_range)
        m_long_range = self.m_long_range_pool(m_long_range)
        m_middle_range = path_m[:, :, ::4, :, :]
        m_middle_range = nn.Dropout(self.dropout_probality)(m_middle_range)
        m_middle_range = self.m_middle_range_compress(m_middle_range)
        m_middle_range = self.m_middle_range_conv(m_middle_range)
        m_middle_range = self.m_middle_range_pool(m_middle_range)
        m_small_range = path_m[:, :, ::2, :, :]
        m_small_range = nn.Dropout(self.dropout_probality)(m_small_range)
        m_small_range = self.m_small_range_compress(m_small_range)
        m_small_range = self.m_small_range_conv(m_small_range)
        m_small_range = self.m_small_range_pool(m_small_range)
        m_fc_out = m_long_range + m_middle_range + m_small_range
        m_fc_out = m_fc_out.squeeze(2).squeeze(2).squeeze(2)
        l_long_range = path_l[:,:,::7,:,:]
        l_long_range = nn.Dropout(self.dropout_probality)(l_long_range)
        l_long_range = self.l_long_range_compress(l_long_range)
        l_long_range = self.l_long_range_conv(l_long_range)
        l_long_range = self.l_long_range_pool(l_long_range)
        l_middle_range = path_l[:, :, ::3, :, :]
        l_middle_range = nn.Dropout(self.dropout_probality)(l_middle_range)
        l_middle_range = self.l_middle_range_compress(l_middle_range)
        l_middle_range = self.l_middle_range_conv(l_middle_range)
        l_middle_range = self.l_middle_range_pool(l_middle_range)
        l_small_range = path_l[:, :, ::2, :, :]
        l_small_range = nn.Dropout(self.dropout_probality)(l_small_range)
        l_small_range = self.l_small_range_compress(l_small_range)
        l_small_range = self.l_small_range_conv(l_small_range)
        l_small_range = self.l_small_range_pool(l_small_range)
        l_fc_out = l_long_range + l_middle_range + l_small_range
        l_fc_out = l_fc_out.squeeze(2).squeeze(2).squeeze(2)
        return l_fc_out + m_fc_out + s_fc_out
    '''
    #=====================================TPP + Multi Stride Multi Path Pooling===========================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        path_s = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        s_long_range = path_s[:, :, ::7, :, :]
        s_long_range = nn.Dropout(self.dropout_probality)(s_long_range)
        s_long_range = self.s_long_range_compress(s_long_range)
        s_long_range = self.s_long_range_conv(s_long_range)
        s_long_range = self.s_long_range_pool(s_long_range)
        s_middle_range = path_s[:, :, ::4, :, :]
        s_middle_range = nn.Dropout(self.dropout_probality)(s_middle_range)
        s_middle_range = self.s_middle_range_compress(s_middle_range)
        s_middle_range = self.s_middle_range_conv(s_middle_range)
        s_middle_range = self.s_middle_range_pool(s_middle_range)
        s_small_range = path_s[:, :, ::2, :, :]
        s_small_range = nn.Dropout(self.dropout_probality)(s_small_range)
        s_small_range = self.s_small_range_compress(s_small_range)
        s_small_range = self.s_small_range_conv(s_small_range)
        s_small_range = self.s_small_range_pool(s_small_range)
        s_fc_out = s_long_range + s_middle_range + s_small_range
        s_fc_out = s_fc_out.squeeze(2).squeeze(2).squeeze(2)
        m_long_range = path_m[:, :, ::7, :, :]
        m_long_range = nn.Dropout(self.dropout_probality)(m_long_range)
        m_long_range = self.m_long_range_compress(m_long_range)
        m_long_range = self.m_long_range_conv(m_long_range)
        m_long_range = self.m_long_range_pool(m_long_range)
        m_middle_range = path_m[:, :, ::4, :, :]
        m_middle_range = nn.Dropout(self.dropout_probality)(m_middle_range)
        m_middle_range = self.m_middle_range_compress(m_middle_range)
        m_middle_range = self.m_middle_range_conv(m_middle_range)
        m_middle_range = self.m_middle_range_pool(m_middle_range)
        m_small_range = path_m[:, :, ::2, :, :]
        m_small_range = nn.Dropout(self.dropout_probality)(m_small_range)
        m_small_range = self.m_small_range_compress(m_small_range)
        m_small_range = self.m_small_range_conv(m_small_range)
        m_small_range = self.m_small_range_pool(m_small_range)
        m_fc_out = m_long_range + m_middle_range + m_small_range
        m_fc_out = m_fc_out.squeeze(2).squeeze(2).squeeze(2)
        l_long_range = path_l[:, :, ::7, :, :]
        l_long_range = nn.Dropout(self.dropout_probality)(l_long_range)
        l_long_range = self.l_long_range_compress(l_long_range)
        l_long_range = self.l_long_range_conv(l_long_range)
        l_long_range = self.l_long_range_pool(l_long_range)
        l_middle_range = path_l[:, :, ::3, :, :]
        l_middle_range = nn.Dropout(self.dropout_probality)(l_middle_range)
        l_middle_range = self.l_middle_range_compress(l_middle_range)
        l_middle_range = self.l_middle_range_conv(l_middle_range)
        l_middle_range = self.l_middle_range_pool(l_middle_range)
        l_small_range = path_l[:, :, ::2, :, :]
        l_small_range = nn.Dropout(self.dropout_probality)(l_small_range)
        l_small_range = self.l_small_range_compress(l_small_range)
        l_small_range = self.l_small_range_conv(l_small_range)
        l_small_range = self.l_small_range_pool(l_small_range)
        l_fc_out = l_long_range + l_middle_range + l_small_range
        l_fc_out = l_fc_out.squeeze(2).squeeze(2).squeeze(2)
        latter_tpp = self.Latter_Tpp(x)
        latter_tpp = self.Latter_Conv(latter_tpp)
        latter_tpp = latter_tpp.squeeze(2).squeeze(2).squeeze(2)
        return l_fc_out + m_fc_out + s_fc_out + latter_tpp
    '''
    # ==============================================Multi Stride Multi Path Concat Pooling=======================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        path_s = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        s_long_range = path_s[:,:,::7,:,:]
        s_long_range = nn.Dropout(self.dropout_probality)(s_long_range)
        s_long_range = self.s_long_range_compress(s_long_range)
        s_long_range = self.s_long_range_conv(s_long_range)
        s_long_range = self.s_long_range_pool(s_long_range)
        s_middle_range = path_s[:, :, ::4, :, :]
        s_middle_range = nn.Dropout(self.dropout_probality)(s_middle_range)
        s_middle_range = self.s_middle_range_compress(s_middle_range)
        s_middle_range = self.s_middle_range_conv(s_middle_range)
        s_middle_range = self.s_middle_range_pool(s_middle_range)
        s_small_range = path_s[:, :, ::2, :, :]
        s_small_range = nn.Dropout(self.dropout_probality)(s_small_range)
        s_small_range = self.s_small_range_compress(s_small_range)
        s_small_range = self.s_small_range_conv(s_small_range)
        s_small_range = self.s_small_range_pool(s_small_range)
        s_fc_out = s_long_range + s_middle_range + s_small_range
        s_fc_out = s_fc_out.squeeze(2).squeeze(2).squeeze(2)
        m_long_range = path_m[:,:,::7,:,:]
        m_long_range = nn.Dropout(self.dropout_probality)(m_long_range)
        m_long_range = self.m_long_range_compress(m_long_range)
        m_long_range = self.m_long_range_conv(m_long_range)
        m_long_range = self.m_long_range_pool(m_long_range)
        m_middle_range = path_m[:, :, ::4, :, :]
        m_middle_range = nn.Dropout(self.dropout_probality)(m_middle_range)
        m_middle_range = self.m_middle_range_compress(m_middle_range)
        m_middle_range = self.m_middle_range_conv(m_middle_range)
        m_middle_range = self.m_middle_range_pool(m_middle_range)
        m_small_range = path_m[:, :, ::2, :, :]
        m_small_range = nn.Dropout(self.dropout_probality)(m_small_range)
        m_small_range = self.m_small_range_compress(m_small_range)
        m_small_range = self.m_small_range_conv(m_small_range)
        m_small_range = self.m_small_range_pool(m_small_range)
        m_fc_out = m_long_range + m_middle_range + m_small_range
        m_fc_out = m_fc_out.squeeze(2).squeeze(2).squeeze(2)
        l_long_range = path_l[:,:,::7,:,:]
        l_long_range = nn.Dropout(self.dropout_probality)(l_long_range)
        l_long_range = self.l_long_range_compress(l_long_range)
        l_long_range = self.l_long_range_conv(l_long_range)
        l_long_range = self.l_long_range_pool(l_long_range)
        l_middle_range = path_l[:, :, ::4, :, :]
        l_middle_range = nn.Dropout(self.dropout_probality)(l_middle_range)
        l_middle_range = self.l_middle_range_compress(l_middle_range)
        l_middle_range = self.l_middle_range_conv(l_middle_range)
        l_middle_range = self.l_middle_range_pool(l_middle_range)
        l_small_range = path_l[:, :, ::2, :, :]
        l_small_range = nn.Dropout(self.dropout_probality)(l_small_range)
        l_small_range = self.l_small_range_compress(l_small_range)
        l_small_range = self.l_small_range_conv(l_small_range)
        l_small_range = self.l_small_range_pool(l_small_range)
        l_fc_out = l_long_range + l_middle_range + l_small_range
        l_fc_out = l_fc_out.squeeze(2).squeeze(2).squeeze(2)
        fc_out = torch.cat((s_fc_out, m_fc_out, l_fc_out), dim=1)
        fc_out = self.fc_out(fc_out)
        return fc_out
    '''
    # ========================================Two Path Networks========================================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)#4x480x32x28x28 here temporal downsample, if can select here directly to capture small motion? or use conv layer to capture?
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)#4x832x16x14x14 here temporal downsample, if can select here directly to capture middle motion? or use conv layer to capture?
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        main_path = self.dropout(self.avg_pool(x))
        main_path = self.logits(main_path)
        if self._spatial_squeeze:
            main_path = main_path.squeeze(3).squeeze(3) # remove dim whose size is 1
        spatial_path = torch.mean(main_path, 2)
        path_l = self.Latter_Tpp(x)
        path_l = nn.Dropout(self.dropout_probality)(path_l)
        path_l = self.Latter_Conv(path_l)
        temporal_path = path_l.squeeze(3).squeeze(3).squeeze(2)
        logits_out = nn.Dropout(self.dropout_probality)(spatial_path / 2 + temporal_path / 2)
        fc_out = self.fc_out(logits_out)
        return fc_out

        #path_l = self.channel_attention(path_l)
    '''
    #=============================================Only Pooling Network=============================================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)  # 4x480x32x28x28 here temporal downsample, if can select here directly to capture small motion? or use conv layer to capture?
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        # for path_1
        path_l = x  # 4x1024x8x7x7
        path_l = self.Latter_Tpp(path_l)
        path_l = nn.Dropout(self.dropout_probality)(path_l)
        path_l = self.Latter_Conv(path_l)
        path_l = path_l.squeeze(3).squeeze(3).squeeze(2)

        fc_out = path_l
        return fc_out
    '''
    #==============================================Multi Stride Multi Path Compress Version===================================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        path_s = x
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.maxpool_2(x)
        path_m = x
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_l = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        main_path = x
        #path_s = self._upsample_add(self.l_s_compress(path_l), path_s)
        #path_m = self._upsample_add(self.l_m_compress(path_l), path_m)
        #temporal_path = self.Latter_Tpp(x)
        #temporal_path = nn.Dropout(0.3)(temporal_path)
        #temporal_path = self.Latter_Conv(temporal_path).squeeze(2).squeeze(2).squeeze(2)
        if self.concat:
            out =  torch.cat((self.s_depend(path_s), self.m_depend(path_m), self.l_depend(path_l)), dim=1)
            return self.fc(out) #+ temporal_path
        else:
            return self.s_depend(path_s) + self.m_depend(path_m) + self.l_depend(path_l) + self.main_depend(main_path) #+ temporal_path
            #return self.s_depend(path_s) + self.m_depend(path_m)  + temporal_path
    '''
    # ==========================================Heavy Multi Stride Multi Path Version===================================
    '''
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.maxpool_2(x)
        path_s = x
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        return self.s_depend(path_s) + self.m_depend(path_m) + self.l_depend(path_l)
    '''
    #========================================Multi Loss(for Multi path)==============================

    def forward(self, x):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        path_s = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        #path_s = self._upsample_add(self.l_s_compress(path_l), path_s)
        #path_m = self._upsample_add(self.l_m_compress(path_l), path_m)
        #temporal_path = self.Latter_Tpp(x)
        #temporal_path = self.Latter_Conv(temporal_path).squeeze(2).squeeze(2).squeeze(2)
        path_s = self.s_depend(path_s)
        path_m = self.m_depend(path_m)
        path_l = self.l_depend(path_l)
        #main_path = self.main_depend(x)
        main_path = path_m + path_l + path_s
        if self.concat:
            out =  torch.cat((self.s_depend(path_s), self.m_depend(path_m), self.l_depend(path_l)), dim=1)
            return self.fc(out) #+ temporal_path
        else:
            #return main_path, path_m, path_l
            return main_path, path_s, path_m, path_l
            #return main_path + path_l + path_m + path_s, path_s, path_m, path_l
            #return self.s_depend(path_s) + self.m_depend(path_m)  + temporal_path

    #=====================================Dense Multi Loss(for Multi path)==============================
    '''
    def forward(self, x):
        x = self.Conv3d_1a_7x7(x) # 5 x 64 x 32 x 112 x 112
        #x_64_1 = x
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x) # 5 x 64 x 32 x 56 x 56
        #x_64_2 = x
        x = self.Conv3d_2c_3x3(x) # 5 x 192 x 32 x 56 x 56
        #x_192 = x
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x) # 5 x 256 x 32 x 28 x 28
        #x_256 =x
        x = self.Mixed_3c(x)
        x_480 = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x) # 5 x 512 x 16 x 14 x 14
        #x_512_1 = x
        x = self.Mixed_4c(x) # 5 x 512 x 16 x 14 x 14
        #x_512_2 = x
        x = self.Mixed_4d(x) # 5 x 512 x 16 x 14 x 14
        x_512_3 = x
        x = self.Mixed_4e(x) # 5 x 528 x 16 x 14 x 14
        x_528 = x
        x = self.Mixed_4f(x) # 832
        x_832_1 = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x) # 5 x 832 x 16 x 14 x 14
        x_832_2 = x
        x = self.Mixed_5c(x) # 1024
        #x_64_1 = self.depend_64_1(x_64_1)
        #x_64_2 = self.depend_64_2(x_64_2)
        #x_192 = self.depend_192(x_192)
        #x_256 = self.depend_256(x_256)  # 480
        #x_480 = self.depend_480(x_480)
        #x_512_1 = self.depend_512_1(x_512_1)
        #x_512_2 = self.depend_512_2(x_512_2)
        #x_512_3 = self.depend_512_3(x_512_3)
        x_528 = self.depend_528(x_528)
        x_832_1 = self.depend_832_1(x_832_1)
        x_832_2 = self.depend_832_2(x_832_2)
        x_1024 = self.depend_1024(x)
        #main_path = 0.01*x_64_1 + 0.01*x_64_2 + 0.03*x_192 + 0.03*x_256 + 0.1*x_480 + 0.1*x_512_1 + 0.2*x_512_2 + 0.2*x_512_3 + 0.2*x_528 + 0.25*x_832_1 + 0.25*x_832_2 + 0.5*x_1024
        #return main_path, [x_64_1, x_64_2, x_192, x_256, x_480, x_512_1, x_512_2, x_512_3, x_528, x_832_1, x_832_2, x_1024]
        #main_path = x_512_3 + x_480 + x_528 + x_1024
        main_path = x_832_1 + x_832_2 + x_528 + x_1024
        return main_path, [x_528, x_832_1, x_832_2, x_1024]
    '''
    #========================================Multi Loss(for Multi TPP path)==============================
    '''
    def forward(self, x):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        path_s = x
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)
        path_m = x
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        path_l = x
        path_s = self.s_depend(path_s)
        path_m = self.m_depend(path_m)
        path_l = self.l_depend(path_l)
        main_path = path_s + path_m + path_l
        return main_path, path_s, path_m, path_l
    '''
    #==================================================Others==================================================
    """
    def forward(self, x, alpha):
        x = self.Conv3d_1a_7x7(x)
        x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x)
        x = self.Conv3d_2c_3x3(x)
        x = self.maxpool_1(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)#4x480x32x28x28 here temporal downsample, if can select here directly to capture small motion? or use conv layer to capture?
        #path_s = x
        #x = self.small_temporal_shift(x)/2 + x/2
        x = self.maxpool_2(x)
        x = self.Mixed_4b(x)
        x = self.Mixed_4c(x)
        x = self.Mixed_4d(x)
        x = self.Mixed_4e(x)
        x = self.Mixed_4f(x)#4x832x16x14x14 here temporal downsample, if can select here directly to capture middle motion? or use conv layer to capture?
        #path_m = x
        #x = self.middle_temporal_shift(x)/2 + x/2
        #path_m = self.MiddleMotionPath(path_m)#3x832x16x1x1
        #print(path_m.size())
        x = self.maxpool_3(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        #for path_1
        path_l = x # 4x1024x8x7x7
        path_l = path_l/2
        #===========================================RCNN============================
        '''
        p3 = self.RCNN_toplayer(path_l)
        p2 = self._upsample_add(p3, self.RCNN_latlayer1(path_m))
        p2 = self.RCNN_smooth2(p2)
        p1 = self._upsample_add(p2, self.RCNN_latlayer2(path_s))
        p1 = self.RCNN_smooth1(p1)
        RCNN_path_1 = self.RCNN_tpp_1(p3)
        RCNN_path_2 = self.RCNN_tpp_2(p2)
        RCNN_path_3 = self.RCNN_tpp_3(p1)
        RCNN_path_1 = self.RCNN_compress_1(RCNN_path_1)
        RCNN_path_2 = self.RCNN_compress_2(RCNN_path_2)
        RCNN_path_3 = self.RCNN_compress_3(RCNN_path_3)
        #rpn_feature_maps = [p1, p2, p3] # 256 x 3
        #rpn_feature_maps = torch.cat((p1, p2, p3), dim=1)  # 1x400x21?
        #RCNN_path = self.RCNN_tpp(rpn_feature_maps)
        #RCNN_path = self.RCNN_compress(RCNN_path)
        #========================way1: mean combine========================
        RCNN_path = RCNN_path_1/3 + RCNN_path_2/3 + RCNN_path_3/3
        RCNN_path = RCNN_path.squeeze(3).squeeze(3).squeeze(2)
        #========================way2: concat combine========================
        #RCNN_path = torch.cat((RCNN_path_1, RCNN_path_2, RCNN_path_3), dim=1)
        #RCNN_path = self.RCNN_compress(RCNN_path)
        #RCNN_path = RCNN_path.squeeze(3).squeeze(3).squeeze(2)
        #then use temporal pooling
        '''
        #==========================================main path============================
        '''
        main_path = self.dropout(self.avg_pool(x))
        main_path = self.logits(main_path)
        if self._spatial_squeeze:
            main_path = main_path.squeeze(3).squeeze(3) # remove dim whose size is 1
        spatial_path = torch.mean(main_path, 2)
        #path_l = self.channel_attention(path_l)
        '''
        #==============================add for multi path==============================
        '''
        path_m = self.MiddleMotionPath(path_m)
        path_m = path_m.squeeze(3).squeeze(3).squeeze(2)
        path_l = self.LatterMotionPath(path_l)
        path_l = path_l.squeeze(3).squeeze(3).squeeze(2)
        '''
        '''
        path_m = self.Middle_Temporal_Compress(path_m)
        path_m = self.Middle_Temporal_Mixed_5b(path_m)
        path_m = self.Middle_Temporal_Mixed_5c(path_m)
        '''
        '''
        lateral_path_m = self.Middle_Pool(path_m)
        lateral_path_m = self.Middle_Temporal_Up(lateral_path_m)
        lateral_path_m = self.Middle_Tpp(lateral_path_m)
        #lateral_path_m = self.Middle_One_Step(path_m)
        #lateral_path_m = self.Middle_Temporal_Compress(path_m)
        lateral_path_s = self.Small_Pool(path_s)
        lateral_path_s = self.Small_Temporal_Up(lateral_path_s)
        lateral_path_s = self.Small_Tpp(lateral_path_s)
        path_l = self.Latter_Tpp(path_l)
        temporal_path = torch.cat((lateral_path_s/3, lateral_path_m/3, path_l/3), dim=2)
        temporal_path = self.temporal_compress(temporal_path)
        temporal_path = temporal_path.squeeze(3).squeeze(3).squeeze(2)
        '''
        '''
        lateral_path_m = self.Middle_Temporal_Up(path_m)
        lateral_path_m = self.Middle_Pool(lateral_path_m)
        lateral_path_m = self.Middle_Tpp(lateral_path_m)
        '''
        #print(lateral_path_m)
        '''
        path_s = self.Small_Pool(path_s)
        path_s = self.Small_Tpp(path_s)
        path_s = self.path_s_de_logits(path_s)
        path_s = path_s.squeeze(3).squeeze(3).squeeze(2)
        path_m = self.Middle_Pool(path_m)
        path_m = self.Middle_Tpp(path_m)
        path_m = self.path_m_de_logits(path_m)
        path_m = path_m.squeeze(3).squeeze(3).squeeze(2)
        path_l = self.Latter_Tpp(path_l)
        path_l = self.path_l_de_logits(path_l)
        path_l = path_l.squeeze(3).squeeze(3).squeeze(2)
        temporal_branch = torch.cat((path_s, path_m, path_l), dim=1)
        temporal_path = self.temporal_compress(temporal_branch)
        '''

        '''
        path_l = self.Latter_Temporal_Compress(path_l)
        lateral_connect = lateral_path_m + path_l
        path_l = self.Lateral_Smooth(lateral_connect)
        '''
        #lateral_connect = torch.cat((lateral_path_m, path_l), dim=1)
        #lateral_connect = torch.cat((self._upsample_add(lateral_path_m[:, :, ::2, :, :], path_l), path_l), dim=1)
        #path_l = self.Latter_Lateral_Compress(lateral_connect)
        #lateral_connect = self._upsample_add(lateral_path_m[:, :, ::2, :, :], path_l)
        #path_l = self.Latter_Tpp(path_l)*1/3 + lateral_path_m*1/3 + lateral_path_s*1/3
        #path_l = self.Latter_Tpp(path_l) * 1/3 + lateral_path_m * 1/3 + lateral_path_s * 1/3
        path_l = self.Latter_Tpp(path_l)
        path_l = nn.Dropout(self.dropout_probality)(path_l)
        path_l = self.Latter_Conv(path_l)
        path_l = path_l.squeeze(3).squeeze(3).squeeze(2)
        #temporal_path = path_l
        #temporal_path = torch.cat((path_m, path_l), dim=2)
        #temporal_path = self.Temporal_Conv(temporal_path)
        #temporal_path = temporal_path.squeeze(3).squeeze(3).squeeze(2)

        #concat_path = torch.cat((spatial_path, path_l, path_m), dim=1)
        #concat_path = spatial_path*alpha + temporal_path*(1-alpha)
        #concat_path = spatial_path * (0.5*torch.sigmoid(self.spatial_weight)+0.2) + temporal_path * (0.5*torch.sigmoid(self.temporal_weight) + 0.2)
        '''
        if random.random() < 0.01:
            print("spatial:{}temporal:{}".format(self.spatial_weight.data.cpu().numpy(),
                                                 self.temporal_weight.data.cpu().numpy()))
        '''
        '''
        if random.random() < 0.01:
            print("l:{}m:{}s:{}spatial:{}temporal:{}".format(self.large_weight.data.cpu().numpy(),
                                                             self.middle_weight.data.cpu().numpy(),
                                                             self.small_weight.data.cpu().numpy(),
                                                             self.spatial_weight.data.cpu().numpy(),
                                                             self.temporal_weight.data.cpu().numpy()))
        '''
        '''
        path_l = self.tpp_l(path_l)
        path_l = self.path_l_de_logits(path_l) # 1 x 400 x 7 ?
        path_l = path_l.squeeze(3).squeeze(3).squeeze(2)
        path_s = self.tpp_s(path_s)
        path_s = self.path_s_de_logits(path_s) # 1 x 400 x 7 ?
        path_s = path_s.squeeze(3).squeeze(3).squeeze(2)
        path_m = self.tpp_m(path_m)
        path_m = self.path_m_de_logits(path_m) # 1 x 400 x 7 ?
        path_m = path_m.squeeze(3).squeeze(3).squeeze(2)
        three_path = torch.cat((path_s, path_m, path_l),dim=1)
        '''
        '''
        path_l = self.tpp_l(path_l)
        path_s = self.tpp_s(path_s)
        path_m = self.tpp_m(path_m)
        three_path = torch.cat((path_s, path_m, path_l), dim=2)#1x400x21?
        three_path = self.path_sp_l_de_logits(three_path)
        #print(three_path)
        three_path = three_path.squeeze(2).squeeze(2).squeeze(2)
        '''
        #path_s_l = self.spp_l(x)
        #path_s_l = self.path_sp_l_de_logits(path_s_l)#3x400x1x1x1
        #path_s_l = path_s_l.squeeze(4).squeeze(2).squeeze(2)
        #concat_path = torch.cat((main_path.view(x.size(0), 400, 1, 1, 1), path_s, path_m, path_l, path_s_l), dim=2)
        #all_path = self.all_conv(concat_path).squeeze(3).squeeze(3).squeeze(2)
        #==============================fusion temporal and spatial==============================
        #logits = nn.Dropout(self.dropout_probality)(main_path/4+path_l/4+path_s/4+path_m/4)
        #logits = nn.Dropout(self.dropout_probality)(main_path)
        #three_path = self.three_fc(three_path)
        #logits = nn.Dropout(self.dropout_probality)(main_path/3+path_l/3+path_m/3)
        #logits = nn.Dropout(self.dropout_probality)(main_path / 4 + path_l / 4 + path_s / 4 + path_m / 4)
        #logits = nn.Dropout(self.dropout_probality)(all_path)
        #logits = nn.Dropout(self.dropout_probality)(main_path + RCNN_path)
        #logits = nn.Dropout(self.dropout_probality)(concat_path)
        #fc_out = self.fc_out(logits)
        fc_out = path_l
        return fc_out
    """