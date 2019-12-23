#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-26 21:57
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : coarse_to_fine_resnet.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import net.resnet_3d.resnet as resnet

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

#=======================================Coarse To Fine Action Recognition=======================================
#1. for a long sequence, use a light network to detect information, such as choose 32 from 128
#
#
#===============================================================================================================

class CoarseToFine3D(nn.Module):
    def __init__(self, num_classes):
        super(CoarseToFine3D, self).__init__()
        self.coarse_net = resnet.CoarseI3Res50(num_classes=num_classes)
        self.fine_net = resnet.i3_res50(num_classes=num_classes)

    def forward(self, x):
        b, c, t, h, w = x.size()
        coarse_feature = self.coarse_net(x) #
        index = 1
        y = self.fine_net(coarse_feature)
        return y
