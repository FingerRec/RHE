#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-10 11:09
     # @Author  : Awiny
     # @Site    :
     # @Project : remote_pytorch_i3d
     # @File    : fl_i3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-10 10:16
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : fl_i3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
#from net.basic_models.se_module import SELayer3D, SELayer3DPooling, SELayer3DPoolingWay2, SELayer3DPoolingWay3, SELayer3DPoolingWay4, SELayer3DConv1, SELayer3DPoolingWay3_light
#from net.basic_models.temporal_no_pooling import TemporalNoPoolingLayer, TemporalConvLayer, TemporalCompressConvLayer
#from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from torch.autograd import Variable

#===========================================================================================
#This file is implement for support full length video input, such as three segments
#===========================================================================================
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


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 groups=1,
                 dilation=(1,1,1),
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
                                padding=0,
                                dilation=dilation,
                                groups=groups,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

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
        #print(x.size())
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


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
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)

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

class TemporalPyramidSimplifyPool3D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self, out_side):
        super(TemporalPyramidSimplifyPool3D, self).__init__()
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
class TemporalPyramidSimplify2Pool3D(nn.Module):
    """
    max pool + avg pool
    """

    def __init__(self, out_side):
        super(TemporalPyramidSimplify2Pool3D, self).__init__()
        self.out_side = out_side

    def forward(self, x):
        out = None
        for n in self.out_side:
            avg_pool = nn.AdaptiveAvgPool3d((n, 1, 1))
            max_pool = nn.AdaptiveMaxPool3d((n, 1, 1))
            y = avg_pool(x) + max_pool(x)
            if out is None:
                out = y.view(y.size()[0], y.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], -1, 1, 1)), 2)
        return out

class TemporalMultiStridePool3D(nn.Module):
    """
    Args:
        out_side (tuple): Length of side in the pooling results of each pyramid layer.

    Inputs:
        - `input`: the input Tensor to invert ([batch, channel, width, height])
    """

    def __init__(self):
        super(TemporalMultiStridePool3D, self).__init__()

    def forward(self, x):
        out = None
        for n in range(1, x.size(2) + 1):
            avg_pool = nn.AdaptiveMaxPool3d((n, 1, 1))
            y = avg_pool(x)
            if out is None:
                out = y.view(y.size()[0], y.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, y.view(y.size()[0], y.size()[1], -1, 1, 1)), 2)
        return out

class TemporalMultipyPool3D(nn.Module):
    """
    Args:
        spatial pooling + temporal multipy pooling
    """

    def __init__(self, out_side):
        super(TemporalMultipyPool3D, self).__init__()
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
            #temp = all ones tensor
            temp = torch.ones((y.size())).cuda()
            #temp = y[:, :, 0:t_r, :, :]
            for i in range(n):
                for j in range(y.size()[2]//n * i, y.size()[2]//n * (i+1)):
                    temp[:, :, i, :, :] *=  y[:, :, j, :, :] #elementise multipy
            if out is None:
                out = temp.view(temp.size()[0], temp.size()[1], -1, 1, 1)
            else:
                out = torch.cat((out, temp.view(temp.size()[0], temp.size()[1], -1, 1, 1)), 2)
        #here the value of out may be extremmly small/big, need sigmoid?
        #out = F.sigmoid(out)
        return out

class TemporalBilinearPool3D(nn.Module):
    """
    Args:
       input: 1024 x 8 x 7 x 7
       output: 1024x 7 x 1 x 1
       Method1: each channel bilinear?
       Method2: all channel bilineari?
    """

    def __init__(self):
        super(TemporalBilinearPool3D, self).__init__()

    def forward(self, x):
        x = torch.sigmoid(x)
        out = None
        #or do pooling here first, become 1024 * 8
        #max_pool = nn.AvgPool3d(kernel_size=(x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        input_size = 1024 * 8 * 7 * 7
        output_size = 1024 * 7 * 1 * 1
        #mcb = CompactBilinearPooling(input_size, input_size, output_size).cuda()
        #y = max_pool(x).view(x.size(0), -1)
        # b x (8x7x7) x 1024 * b x 1024 x (8x7x7) = b x 8x7x7 x 8x7x7 = b x 392 x 392
        y = x.view(x.size(0), x.size(1), -1) # b x 1024 x 392
        y1 = torch.transpose(y, 2, 1) # b x 392 x 1024
        y = torch.bmm(y, y1)/(8*7*7) # 1024 x1024
        '''
        #print(y.size())
        y = y.view(x.size(0), 8*7*7, 8*7*7, 1, 1)
        #print(y.size())
        y = nn.AdaptiveAvgPool3d((1, 1, 1))(y)
        y = F.relu(y)
        #out = mcb(y, y)
        #print(out)
        '''
        y = y.view(x.size(0), -1)
        out = torch.sign(y)*torch.sqrt(y+1e-8)
        out = F.normalize(out)
        #print(out.size())
        out = F.sigmoid(out)
        return out

class CompactBilinearPool3D(nn.Module):
    """
    Args:
       input: 1024 x 8 x 7 x 7
       output: 1024x 7 x 1 x 1
       first use bilinear pooling and then tpp
    """

    def __init__(self):
        super(CompactBilinearPool3D, self).__init__()
        self.tpp = TemporalPyramidPool3D((1,2,4))
    def forward(self, x):
        out = None

        x = torch.relu(x)
        #y = nn.AdaptiveAvgPool3d((1, 1, 1))(y)
        mcb = CompactBilinearPooling(1024, 1024, 1024).cuda()
        out = mcb(x.permute(0,2,3,4,1), x.permute(0,2,3,4,1))
        out = out.permute(0,4,1,2,3)
        #print(out.size())
        #out = out.view(x.size(0), -1)
        #out = nn.AdaptiveMaxPool3d((1,1,1))(out).view(x.size(0), -1)
        #out = torch.sqrt(out+1e-8)
        out = F.normalize(out)
        #out = F.relu(out)
        #print(out)
        out = self.tpp(out)
        #print(out)
        return out

class FullLengthI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_prob=0.5):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(FullLengthI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return
        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.max_pool = nn.MaxPool3d(kernel_size=[8, 1, 1],
                                     stride=(8, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout_probality = dropout_prob
        '''
        self.main_conv1 = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=384 + 384 + 128 + 128,
                             kernel_shape=[1, 7, 7],
                             stride=(1,7,7),
                             padding=0,
                             activation_fn=F.relu,
                             use_batch_norm=None,
                             use_bias=True,
                             name='main_conv1')
        self.main_conv2 = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=384 + 384 + 128 + 128,
                             kernel_shape=[1, 3, 3],
                             stride=(1,2,2),
                             padding=0,
                             activation_fn=F.relu,
                             use_batch_norm=None,
                             use_bias=True,
                             name='main_conv2')
        self.main_conv3 = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=384 + 384 + 128 + 128,
                             kernel_shape=[1, 3, 3],
                             stride=(1,2,2),
                             padding=0,
                             activation_fn=F.relu,
                             use_batch_norm=None,
                             use_bias=True,
                             name='main_conv3')
        '''
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=None,
                             use_bias=True,
                             name='logits')

        self.softmax = torch.nn.Softmax(dim=1)
        if self._num_classes != 400:
            self.fc_out = nn.Linear(400, self._num_classes, bias=True)
        self.build()
        #self.tpp_layer = TemporalPyramidPool3D(out_side=(1, 2, 4))
        self.tpp_layer = TemporalPyramidSimplifyPool3D(out_side=(1, 2, 4, 8, 16))
        #self.mspp_layer = TemporalMultiStridePool3D()
        #self.tpp_layer_2pool = TemporalPyramidSimplify2Pool3D(out_side=(1, 2, 4, 8))

        #=========================for tpp pooling=========================

        self.path2_de_logits_custom = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[31, 1, 1],
                             stride=(31,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')

        '''
        self.path2_de_logits_1 = Unit3D(in_channels=1024, output_channels=64,
                             kernel_shape=[7, 1, 1],
                             stride=(7,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')
        self.path2_de_logits_2 = Unit3D(in_channels=64, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')
        self.path2_de_logits_3 = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[7, 1, 1],
                             stride=(7,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             groups=400,
                             name='path_2_de_logits')
        '''
        '''
        #add one layer lstm
        self.rnn= nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True)
        self.lstm_logits = Unit3D(in_channels=512, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='lstm_logits')
        '''
        '''
        self.path_sp_l_de_logits = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[1, 21, 1],
                             stride=(1,21,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_l_de_logits')
        '''
        if self._final_endpoint == end_point: return

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def replace_dropout(self, dropout_radio):
        self.dropout = nn.Dropout(dropout_radio)

        if self._num_classes != 400:
            self.logits_dropout = nn.Dropout(dropout_radio)

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
    #========================================================================================================
    #method1: we concat predict together, and then use our model
    #method2: for every segment, we process it idependly, and then acerage them
    #========================================================================================================
    def forward(self, x):
        #print(x.size()) # b x num_segmtns x c x t x h x w
        #print(x.size())
        clips_feature = None #x:batchx3x3x64x224x224
        for i in range(x.size(1)):
            clip = x[:,i, :, :, :, :] #batch_sizex3x64x224x224
            #print(clip.size()) #
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    clip = self._modules[end_point](clip)
            if clips_feature is None:
                clips_feature = clip
            else:
                clips_feature = torch.cat((clips_feature, clip), dim = 2)
        #print(clips_feature.size())
        # b x 1024 x 8 x 7 x 7 -> b x 1024 x 24 x 7 x 7
        #==============main_path=============================
        #1024 x [8xsegments] x 7 x 7 -> 1024 x 23 x 1 x 1 -> 400 x 1
        #may be transform -> 7 x 1 x1 can converigence fast
        #pool_feature = clips_feature[:, :, :8, :, :]
        #pool_feature = nn.Dropout(self.dropout_probality)(self.avg_pool(pool_feature))
        #
        pool_feature = nn.Dropout(self.dropout_probality)(nn.AdaptiveAvgPool3d((7, 1, 1))(clips_feature))
        pool_feature = self.logits(pool_feature)
        if self._spatial_squeeze:
            logits = pool_feature.squeeze(3).squeeze(3)
        logits = torch.mean(logits, 2)
        #======================tpp============================
        #1024 x [8 x segments] x 7 x 7 -> 1024 x 15 x 1 x 1(tpp) -> 400 x 1 x 1(path_2_de_logits) -> 400 x 1
        path_2 = clips_feature
        path_2 = self.tpp_layer(path_2)
        path_2 = self.path2_de_logits_custom(path_2)
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)

        if self._num_classes != 400:
            logits_out = nn.Dropout(self.dropout_probality)(logits / 2 + path_2 / 2)
            fc_out = self.fc_out(logits_out)
            return fc_out
        else:
            return logits

    '''
    def forward(self, x):
        clips_feature = None #x:batchx3x3x64x224x224
        out = None
        for i in range(x.size(1)):
            clip = x[:,i, :, :, :, :] #batch_sizex3x64x224x224
            #print(clip.size()) #
            for end_point in self.VALID_ENDPOINTS:
                if end_point in self.end_points:
                    clip = self._modules[end_point](clip)
            pool_feature = nn.Dropout(self.dropout_probality)(nn.AdaptiveAvgPool3d((7, 1, 1))(clip))
            pool_feature = self.logits(pool_feature)
            if self._spatial_squeeze:
                logits = pool_feature.squeeze(3).squeeze(3)
            logits = torch.mean(logits, 2)
            path_2 = clip
            path_2 = self.tpp_layer(path_2)
            path_2 = self.path2_de_logits(path_2)
            path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
            if self._num_classes != 400:
                logits_out = nn.Dropout(self.dropout_probality)(logits / 2 + path_2 / 2)
                fc_out = self.fc_out(logits_out)
            else:
                fc_out = logits
            if out is None:
                out = fc_out / x.size(1)
            else:
                out += fc_out / x.size(1)
        return out
    '''
    #
    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def get_fine_tuning_parameters(model):
    ft_module_names = []
    # ft_module_names.append('Mixed_5b')
    # ft_module_names.append('Mixed_5c')
    ft_module_names.append('fc_out')
    ft_module_names.append('logits')
    ft_module_names.append('path2_de_logits_custom')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0})

    return parameters

def transfer_learning_stragety(model, lr):
    ft_module_names = []
    ft_module_names.append('fc_out')
    ft_module_names.append('logits')
    ft_module_names.append('path2_de_logits_custom')
    parameters = []
    l = 0
    len = 0
    for _, _ in model.named_parameters():
        len += 1
    print(len)
    for k, v in model.named_parameters():
        l = l + 1
        beta_l = math.sin(l / len * math.pi / 2) * lr
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v, 'lr' : lr})
                break
        else:
            parameters.append({'params': v, 'lr': beta_l})

    return parameters
