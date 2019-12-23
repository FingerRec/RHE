import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
from net.basic_models.se_module import SELayer3D, SELayer3DPooling, SELayer3DPoolingWay2, SELayer3DPoolingWay3, SELayer3DPoolingWay4, SELayer3DConv1, SELayer3DPoolingWay3_light
from net.basic_models.temporal_no_pooling import TemporalNoPoolingLayer, TemporalConvLayer, TemporalCompressConvLayer
#from compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from net.basic_models.attention import TemporalAttention
from torch.autograd import Variable
from collections import OrderedDict
from net.basic_models.TPP import SkipPooling

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
            self.bn = nn.BatchNorm3d(self._output_channels)

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


'''
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


class Unit3D(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=F.relu,
                 padding=0,
                 use_bias=False,
                 use_batch_norm=True,
                 name='unit3dpy',
                ):
        super(Unit3D, self).__init__()

        self.padding = 'SAME'
        self.name = name
        self.use_bn = use_batch_norm
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_shape, stride)
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
                    kernel_shape,
                    stride=stride,
                    bias=use_bias)
            else:
                self.conv3d = torch.nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_shape,
                    stride=stride,
                    padding=pad_size,
                    bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_shape,
                padding=padding_shape,
                stride=stride,
                bias=use_bias)
        else:
            raise ValueError(
                'padding should be in [VALID|SAME] but got {}'.format(padding))

        if self.use_bn:
            self.batch3d = torch.nn.BatchNorm3d(out_channels)

        self.activation = activation_fn

    def forward(self, inp):
        if self.padding == 'SAME' and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = torch.nn.functional.relu(out)
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

class TemporalShuffle(nn.Module):
    """
    for this module, random shuffle temporal dim, we want to find if the temporal information is important
    """
    def __init__(self):
        super(TemporalShuffle, self).__init__()

    def forward(self, x):
        """
        random shuffle temporal dim
        :param x: b x c x t x h x w
        :return: out: b x c x t' x h x w
        """
        t = x.size(2)
        idx = torch.randperm(t)
        out = x[:,:,idx,:,:]
        return out

class InceptionI3d(nn.Module):
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
        #'ChannelAttention_1',
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

        super(InceptionI3d, self).__init__()
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
        #self.end_points[end_point] = TemporalNoPoolingLayer(64, kernel=[1, 3, 3], stride=(1, 2, 2),
        #                                                    side=(1))
        #self.end_points[end_point] = SELayer3DPoolingWay2(64, kernel=[1, 3, 3], stride=(1, 2, 2),
        #                                                  frame=32, side=[1])
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
        #self.end_points[end_point] = TemporalNoPoolingLayer(192, kernel=[1, 3, 3], stride=(1, 2, 2),
        #                                                    side=(1))
        #self.end_points[end_point] = SELayer3DPoolingWay2(192, kernel=[1, 3, 3], stride=(1, 2, 2),
        #                                                  frame=32, side=[1])
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        # ====================================Add Some Model To I3d
        """
        end_point = 'attention_1'
        self.end_points[end_point] = Self_Attn(480, 'relu')
        if self._final_endpoint == end_point: return
        """
        # =======================================
        end_point = 'MaxPool3d_4a_3x3'
        #self.end_points[end_point] = nn.Sequential(
        #    OrderedDict([
        #        ('temporal_attention', TemporalAttention(temporal_size=32)),
        #        ('max_pool', MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)),
        #                ]))

        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        #self.end_points[end_point] = TemporalNoPoolingLayer(128 + 192 + 96 + 64, kernel=[3, 3, 3], stride=(2, 2, 2),
        #                                                    side=(1, 2))
        #self.end_points[end_point] = SELayer3DPoolingWay2(128 + 192 + 96 + 64, kernel=[3, 3, 3], stride=(2, 2, 2),
        #                                                  frame=32, side=(1, 3))
        #self.end_points[end_point] = SELayer3DPoolingWay3(MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0),
        #                                                 128 + 192 + 96 + 64, kernel=[3, 3, 3], stride=(2, 2, 2),
        #                                                  frame=32, side=(1, 2, 4, 8))
        #self.end_points[end_point] = SELayer3DPoolingWay3_light(
        #    MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0),
        #    128 + 192 + 96 + 64, kernel=[3, 3, 3], stride=(2, 2, 2),
        #    frame=32, side=(8, 16, 32))
        #self.end_points[end_point] = SELayer3DPoolingWay4(MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0),
        #                                                  128 + 192 + 96 + 64, kernel=[3, 3, 3], stride=(2, 2, 2),
        #                                                  frame=32, side=(1, 2, 4, 8))
        #self.end_points[end_point] = SELayer3DConv1(MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),padding=0),
        #                                                  128 + 192 + 96 + 64, kernel=[3, 3, 3], stride=(2, 2, 2),
        #                                                  frame=32, side=(32, 16, 8))
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
        #self.end_points[end_point] = nn.Sequential(OrderedDict([('temporal_attention', TemporalAttention(temporal_size=16)), ('max_pool', MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        #                                                  padding=0))]))
        #main_pooling = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        #                                                  padding=0)
        #self.end_points[end_point] = SELayer3DPoolingWay2(256 + 320 + 128 + 128, kernel=[2,2,2], stride=(2,2,2), frame=16, side=(1,2))
        #self.end_points[end_point] = TemporalNoPoolingLayer(256 + 320 + 128 + 128, kernel=[2,2,2], stride=(2,2,2), side=(1,2))
        #self.end_points[end_point] = TemporalConvLayer(256 + 320 + 128 + 128, kernel=[2,2,2], stride=(2,2,2), side=(1,2))
        #self.end_points[end_point] = TemporalCompressConvLayer(256 + 320 + 128 + 128, kernel=[2,2,2], stride=(2,2,2), side=(1,2))
        #self.end_points[end_point] = SELayer3DPoolingWay3(MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        #                                                  padding=0), 256 + 320 + 128 + 128, kernel=[2, 2, 2], stride=(2, 2, 2),
        #                                                 frame=16, side=(1, 2, 4))
        #self.end_points[end_point] = SELayer3DPoolingWay3_light(MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        #                                                  padding=0), 256 + 320 + 128 + 128, kernel=[2, 2, 2], stride=(2, 2, 2),
        #                                                  frame=16, side=(4, 8, 16))
        #self.end_points[end_point] = SELayer3DPoolingWay4(MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
        #                                                  padding=0), 256 + 320 + 128 + 128, kernel=[2, 2, 2], stride=(2, 2, 2),
        #                                                  frame=16, side=(1, 2, 4, 8))
        #self.end_points[end_point] = SELayer3DConv1(MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),padding=0),
        #                                            256 + 320 + 128 + 128, kernel=[2, 2, 2], stride=(2, 2, 2), frame=16, side=(4, 8, 16))
        if self._final_endpoint == end_point: return
        '''
        end_point = 'ChannelAttention_1'
        self.end_points[end_point] = SELayer3D(256 + 320 + 128 + 128, 16)
        if self._final_endpoint == end_point: return
        '''

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
        self.logits_custom = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=None,
                             use_bias=True,
                             name='logits')

        self.softmax = torch.nn.Softmax(dim=1)
        #if self._num_classes != 400:
        #    self.fc_out = nn.Linear(self._num_classes, self._num_classes, bias=True)
        self.build()
        #self.compact_fc = nn.Linear(1024, self._num_classes)
        #torch.nn.init.kaiming_normal_(self.compact_fc.weight.data)
        #self.blinear_fc = nn.Linear(1024**2, self._num_classes)
        #self.channel_attention = SELayer3D(1024, reduction=16)
        #self.channel_attention_2 = SELayer3D(1024, reduction=16)
        #self.tpp_layer = TemporalPyramidSimplifyPool3D(out_side=(1,)) # for global pooling
        #self.tpp_layer = TemporalPyramidSimplifyPool3D(out_side=(4,)) # for local pooling
        #self.tpp_layer = TemporalPyramidSimplifyPool3D(out_side=(8,)) # for single pooling
        #self.tpp_layer = TemporalPyramidSimplifyPool3D(out_side=(1, 2, 4))
        self.tpp_layer = TemporalPyramidSimplifyPool3D(out_side=(1, 2, 4, 8))
        #self.tpp_layer = SkipPooling(out_side=(1, 2, 3, 7))
        #.mspp_layer = TemporalMultiStridePool3D()
        #self.tpp_layer_2pool = TemporalPyramidSimplify2Pool3D(out_side=(1, 2, 4, 8))
        #self.tpp_layer = TemporalMultipyPool3D(out_side=(1, 2, 4))
        #self.tpp_layer = TemporalBilinearPool3D()
        #self.spp_l = SpatialPyramidPool3D(out_side=(1,2,4))
        #self.temporal_shuffle = TemporalShuffle()
        #self.tpp_layer =  CompactBilinearPool3D()
        #=========================add for multi stride pooling================
        '''
        self.channel_compress = Unit3D(in_channels=1024, output_channels=64,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             use_batch_norm=False,
                             use_bias=True,
                             name='channel_compress')
        self.channel_decompress = Unit3D(in_channels=64, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             use_batch_norm=False,
                             use_bias=True,
                             name='channel_decompress')
        self.low_channel_conv = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[36, 1, 1],
                             stride=(36,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='low_channel_conv')
        '''
        #=========================for multi stride pooling================
        '''
        self.path2_ch_compress = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[1, 1, 1],
                             stride=(1,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')
        self.path2_tp_logits = Unit3D(in_channels=400, output_channels=400,
                             kernel_shape=[36, 1, 1],
                             stride=(36,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')
        '''
        #=========================for tpp pooling=========================
        '''
        self.path2_de_logits = Unit3D(in_channels=1024, output_channels=400,
                             kernel_shape=[4, 1, 1],
                             stride=(4,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')
        '''
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
        #================================================LSTM================================
        #add one layer lstm
        #self.rnn= nn.LSTM(input_size=1024, hidden_size=400, num_layers=2, batch_first=False)
        '''
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
        #================================================Temporal Attention================================
        #self.temporal_attention = TemporalAttention(temporal_size=8)
        self.path2_de_logits = Unit3D(in_channels=1024, output_channels=self._num_classes,
                             kernel_shape=[15, 1, 1],
                             stride=(15,1,1),
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='path_2_de_logits')
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

    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        #path_2 = self.channel_attention_2(x)
        path_2 = x
        #======================tpp============================
        '''
        #path_2 = self.temporal_shuffle(path_2)
        path_2 = self.tpp_layer(path_2)
        path_2 = self.path2_de_logits(path_2) # 1 x 400 x 7 ?
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
        #first do temporal shuffle
        '''
        '''
        path_2 = self.temporal_shuffle(path_2)
        path_2 = self.tpp_layer_2pool(path_2)
        path_2 = self.path2_de_logits(path_2)
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
        '''
        #======================multi_stide pp=========================
        '''
        path_2 = self.mspp_layer(path_2)
        path_2 = self.path2_ch_compress(path_2)
        path_2 = self.path2_tp_logits(path_2)
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
        '''
        #======================channel compress + tpp=================
        '''
        path_2 = self.mspp_layer(path_2)
        path_2 = self.channel_compress(path_2) # 1024 -> 64 here, then conv over 64 channel, then back to 400 channel
        path_2 = self.channel_decompress(path_2)
        path_2 = self.low_channel_conv(path_2)
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
        '''
        #==========================CNN + LSTM===================
        '''
        #path_2 = nn.AdaptiveAvgPool3d((path_2.size()[2], 1, 1))(path_2)
        #path_2 = path_2.squeeze(3).squeeze(3)
        path_2 = path_2.view(path_2.size()[0], path_2.size()[1], -1)
        hidden = None
        path_2 = torch.transpose(path_2, 1, 2)
        #print(path_2.size())
        path_2, hidden = self.rnn(path_2, hidden)
        #weight = Variable(torch.Tensor(range(path_2.shape[0])) / sum(range(path_2.shape[0]))).view(-1, 1).repeat(1,path_2.shape[1])
        # print(weight)
        #path_2 = torch.mul(path_2, weight)
        path_2 = torch.mean(path_2, dim=1).squeeze(1)  # 1x101?
        #print(path_2.size())
        #path_2 = path_2[:,0,:]
        #path_2 = path_2.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        #path_2 = self.lstm_logits(path_2)
        #path_2 = path_2.squeeze(2).squeeze(2).squeeze(2)
        #print(path_2)
        '''
        #==============================Three Path==============================
        '''
        path_2 = self.tpp_layer(path_2)
        path_2 = self.path2_de_logits_1(path_2) # 1 x 400 x 7 ?
        path_2 = self.path2_de_logits_2(path_2)
        path_2 = self.path2_de_logits_3(path_2)
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
        '''
        #=============================SPP==============================
        '''
        path_s_l = self.spp_l(x)
        path_s_l = self.path_sp_l_de_logits(path_s_l)
        path_s_l = path_s_l.squeeze(4).squeeze(2).squeeze(2)
        #pool_feature = self.channel_attention(pool_feature)
        '''
        #=============================temporal attention==============================
        #path_2 = self.temporal_attention(path_2)
        path_2 = self.tpp_layer(path_2)
        path_2 = self.path2_de_logits(path_2) # 1 x 400 x 7 ?
        path_2 = path_2.squeeze(3).squeeze(3).squeeze(2)
        #print("???")
        #================origin================================
        pool_feature = nn.Dropout(self.dropout_probality)(self.avg_pool(x))
        x = self.logits_custom(pool_feature)
        if self._spatial_squeeze:
            out_logits = x.squeeze(3).squeeze(3)
        logits = torch.mean(out_logits, 2)
        #print(logits)
        #================multi conv==========================
        '''
        #temporal pooling first, and then use conv to get final result
        pool_feature = self.max_pool(x)
        x = self.main_conv1(pool_feature)#4x1024x1x4x4
        #print(x.size())
        #x = self.main_conv2(x)
        #x = self.main_conv3(x)
        x = self.logits(x)
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        logits = torch.mean(logits, 2)
        '''
        if self._num_classes != 400:
            #logits_out = nn.Dropout(self.dropout_probality)(logits / 2 + path_2 / 2)
            #logits_out = nn.Dropout(self.dropout_probality)(path_s_l / 3 + path_2 / 3 + logits / 3)
            #logits_out = nn.Dropout(self.dropout_probality)(logits * path_2)
            #print(logits_out)
            #logits_out = F.sigmoid(logits_out)
            #print(logits_out)
            #logits_out = nn.Dropout(self.dropout_probality)(logits)
            #logits_out = nn.Dropout(self.dropout_probality)(path_2)
            #fc_out = self.fc_out(logits_out) + self.blinear_fc(path_2)
            #logits_out = nn.Dropout(self.dropout_probality)(path_2)
            fc_out = logits/2+path_2/2
            #fc_out = self.fc_out(logits_out)
            #print(path_2)
            #fc_out = self.blinear_fc(path_2)
            return fc_out
        else:
            return logits

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)


def get_fine_tuning_parameters(model):
    ft_module_names = []
    ft_module_names.append('fc_out')
    ft_module_names.append('logits')
    ft_module_names.append('path2_de_logits')

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
    ft_module_names.append('path2_de_logits')
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
