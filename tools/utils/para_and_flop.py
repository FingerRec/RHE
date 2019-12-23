#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-09 10:32
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : para_and_flop.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close
sys.path.append('.')
#========================================================================
#this module is used to calculate net's para and flop
#========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from net.multi_path_i3d import MultiPathI3d
from net.multi_path_i3d_pt import I3D
from net.i3d_1fc import InceptionI3d, transfer_learning_stragety
from torchvision.models import resnet50
from thop import profile
os.environ["CUDA_VISIBLE_DEVICES"]='2'

def main():
    #params is correct
    '''
    model = MultiPathI3d(num_classes=51, in_channels=3)
    model = model.cuda()
    summary(model, (3, 64, 224, 224))
    '''
    #flops is correct #pytorch 1.1
    #device = torch.device("cuda:2")
    model = I3D(num_classes=51, modality='rgb')
    #model = InceptionI3d(num_classes=51, in_channels=3)
    model = model.cuda()
    #model = model.to(device)
    flops, params = profile(model, input_size=(1, 3, 64, 224, 224))
    print(params/1024/1024, flops/1024/1024/1024)
if __name__ == '__main__':
    main()