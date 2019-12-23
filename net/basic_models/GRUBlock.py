#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=False)
        self.embedding = nn.Linear(nHidden, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class GRUBlock(nn.Module):
    def __init__(self, in_channel, out_channel, nh=256):
        super(GRUBlock, self).__init__()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(in_channel, nh, nh),
            BidirectionalLSTM(nh, nh, out_channel))

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = nn.AdaptiveMaxPool3d((t, 1, 1))(x).squeeze(3).squeeze(3)
        x = x.permute(2, 0, 1)  # [t, b, c]
        # rnn features
        output = self.rnn(x)
        output = output.permute(1, 2, 0)
        # print(output.size())
        return output.mean(2)
        # return output[:, :, t-1]

    # =============================2 layer GRU===================
    # def __init__(self, in_channel, out_channel):
    #     super(GRUBlock, self).__init__()
    #     self.down_channel = nn.Conv1d(in_channel, out_channel, 1, 1)
    #     self.rnn = nn.GRU(out_channel, out_channel, 2, bidirectional=True)
    #     self.out_channel = out_channel
    #
    # def forward(self, x):
    #     b, c, t, h, w = x.size()
    #     x = nn.AdaptiveMaxPool3d((t, 1, 1))(x).squeeze(3).squeeze(3)
    #     x = self.down_channel(x)
    #     x = x.transpose(0, 1).transpose(0, 2)
    #     h0 = torch.randn(4, b, self.out_channel).cuda()
    #     output, hn = self.rnn(x, h0)
    #     output = output.transpose(0, 1).transpose(1, 2)
    #     print(output.size())
    #     print(hn.size())
    #     return hn


    # ==========================================GRU Cell=========================
    # def __init__(self, in_channel, out_channel):
    #     super(GRUBlock, self).__init__()
    #     self.down_channel = nn.Conv3d(in_channel, out_channel, 1, 1)
    #     self.rnn = nn.GRUCell(out_channel, out_channel)
    #     # self.rnn2 = nn.GRUCell(out_channel, out_channel)
    #     self.out_channel = out_channel
    #
    # def forward(self, x):
    #     outputs = []
    #     outputs2 = []
    #     x = self.down_channel(x)
    #     b, c, t, h, w = x.size()
    #     hx = torch.zeros(b, self.out_channel).cuda()
    #     # hx2 = torch.zeros(b, self.out_channel).cuda()
    #     x = nn.AdaptiveMaxPool3d((t, 1, 1))(x).squeeze(3).squeeze(3)
    #     for i in range(t):
    #         hx = self.rnn(x[:, :, i], hx)
    #         outputs.append(hx)
    #     return outputs[t-1]
    #     # for i in range(t):
    #     #     hx2 = self.rnn2(outputs[i], hx2)
    #     #     outputs2.append(hx2)
    #     # return sum(outputs2)/t

class GRUCombineMultiDepend(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GRUCombineMultiDepend, self).__init__()
        self.down_channel = nn.Conv1d(in_channel, out_channel, 1, 1)
        self.rnn = nn.GRU(out_channel, out_channel, 2, bidirectional=False)
        self.out_channel = out_channel
        self.channel_compress = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.long_range_depen = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=1)
        self.middle_range_depen = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=2, stride=1)
        self.small_range_depen = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=2,
                                           stride=1)
        self.local_range_depen = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = nn.AdaptiveMaxPool3d((t, 1, 1))(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        x = self.down_channel(x)
        x = F.relu(x)
        x = x.transpose(0, 1).transpose(0, 2)
        h0 = torch.randn(2, b, self.out_channel).cuda()
        output, hn = self.rnn(x, h0)
        output = output.transpose(0, 1).transpose(1, 2)
        long_range_depen = self.long_range_depen(output[:, :, ::(t - 1)])
        middle_range_depen = self.middle_range_depen(output[:, :, ::math.ceil((t - 1) / 2)])
        small_range_depen = self.small_range_depen(output[:, :, ::math.ceil((t-1)/4)])
        local_range_depen = self.local_range_depen(output[:, :, ::math.ceil((t - 1) / 7)])
        return nn.AdaptiveMaxPool1d((1))(long_range_depen).squeeze(2) + \
               nn.AdaptiveMaxPool1d((1))(middle_range_depen).squeeze(2) + \
               nn.AdaptiveMaxPool1d((1))(small_range_depen).squeeze(2) + \
               nn.AdaptiveMaxPool1d((1))(local_range_depen).squeeze(2)