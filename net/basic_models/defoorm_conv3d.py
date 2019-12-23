#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-26 21:35
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : defoorm_conv3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _triple
import math

from torch.autograd import Function, Variable
class ConvOffset3d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 channel_per_group=1,
                 bias=True,
                 groups=1):
        super(ConvOffset3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.channel_per_group = channel_per_group
        self.group = groups

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.weight = nn.Parameter(torch.cuda.FloatTensor(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal(self.weight.data, mode='fan_out')
        nn.init.constant(self.weight.data, 1)
        if bias:
            self.bias = nn.Parameter(torch.cuda.FloatTensor(out_channels))
            nn.init.uniform(self.bias.data, -0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, offset):
        # return ConvOffset3dFunction(self.stride, self.padding, self.channel_per_group)(input, offset, self.weight)
        return ConvOffset3dFunction.apply(input, offset, self.weight, self.bias, self.stride, self.padding,
                                          self.dilation,
                                          self.channel_per_group, self.group)

class ConvOffset3dFunction(Function):
    # def __init__(ctx, stride, padding, channel_per_group):
    #     super(ConvOffset3dFunction, ctx).__init__()
    #     ctx.stride = stride
    #     ctx.padding = padding
    #     ctx.channel_per_group = channel_per_group
    #     ctx.savedtensors = ()

    @staticmethod
    def forward(ctx, inputs, offset, weight, bias=None, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                channel_per_group=1,
                group=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.channel_per_group = channel_per_group
        ctx.group = group
        ctx.save_for_backward(inputs, offset, weight, bias)

        output_size = [int((inputs.size()[i + 2] + 2 * ctx.padding[i] - weight.size()[i + 2]) / ctx.stride[i] + 1)
                       for i in range(3)]

        output = inputs.new(inputs.size(0), weight.size(0), output_size[0], output_size[1], output_size[2], ).zero_()

        ctx.columns = inputs.new(inputs.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                                 output_size[0] * output_size[1] * output_size[2]).zero_()

        deform_conv3d_op.deform_conv_forward_cuda(
            inputs, weight, offset, ctx.columns, output,
            ctx.padding[0], ctx.padding[1], ctx.padding[2],
            ctx.stride[0], ctx.stride[1], ctx.stride[2],
            ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
            ctx.channel_per_group, ctx.group)

        if bias is not None:
            output += bias.view((1, -1, 1, 1, 1)).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, offset, weight, bias = ctx.saved_variables

        grad_input = grad_offset = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = inputs.data.new(inputs.size()).zero_()
            grad_offset = offset.data.new(offset.size()).zero_()

            deform_conv3d_op.deform_conv_backward_input_offset_cuda(
                inputs.data, weight.data, offset.data, grad_output.data, ctx.columns, grad_input, grad_offset,
                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                ctx.channel_per_group, ctx.group)

        if ctx.needs_input_grad[2]:
            grad_weight = weight.data.new(weight.size()).zero_()

            deform_conv3d_op.deform_conv_backward_weight_cuda(
                inputs.data, offset.data, grad_output.data, ctx.columns, grad_weight,
                ctx.padding[0], ctx.padding[1], ctx.padding[2],
                ctx.stride[0], ctx.stride[1], ctx.stride[2],
                ctx.dilation[0], ctx.dilation[1], ctx.dilation[2],
                ctx.channel_per_group, ctx.group)

        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).sum(1).sum(1).sum(1)

        return Variable(grad_input), Variable(grad_offset), Variable(
            grad_weight), grad_bias, None, None, None, None, None