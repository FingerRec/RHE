import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
from net.basic_models.spatio_temporal_conv import SpatioTemporalConv
from net.basic_models.se_module import *
from net.basic_models.non_local import *

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.N = 8
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels//self.N, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//self.N, output_channels//self.N, 3, stride, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//self.N, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)
        nn.init.xavier_uniform(self.conv4.weight.data)
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class ResidualBlock3D(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.input_channels = input_channels
        self.N = 8
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(input_channels, output_channels//self.N, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm3d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(output_channels//self.N, output_channels//self.N, (1, 3, 3), (1,stride,stride), padding = (0,1,1), bias = False)
        self.bn3 = nn.BatchNorm3d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(output_channels//self.N, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv3d(input_channels, output_channels , 1, stride, bias = False)
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)
        nn.init.xavier_uniform(self.conv4.weight.data)
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out


class ResidualBlock2plus1D(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock2plus1D, self).__init__()
        self.input_channels = input_channels
        self.N = 4
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = SpatioTemporalConv(input_channels, output_channels//self.N, 1, stride=1)
        #self.conv1 = nn.Conv2d(input_channels, output_channels//self.N, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm3d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = nn.Conv2d(output_channels//self.N, output_channels//self.N, 3, stride, padding = 1, bias = False)
        self.conv2 = SpatioTemporalConv(output_channels//self.N, output_channels//self.N, 3, padding = 1,stride=stride)
        self.bn3 = nn.BatchNorm3d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = SpatioTemporalConv(output_channels//self.N, output_channels, 1, stride=1)
        #self.conv3 = nn.Conv2d(output_channels//self.N, output_channels, 1, 1, bias = False)
        #self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        self.conv4 = SpatioTemporalConv(input_channels, output_channels, 1, stride = stride)
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class ResidualBlockBottleneck(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlockBottleneck, self).__init__()
        self.input_channels = input_channels
        self.N = 8
        self.output_channels = output_channels
        self.stride = stride
        self.bn1 = nn.BatchNorm3d(input_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(input_channels, output_channels//self.N, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm3d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(output_channels//self.N, output_channels//self.N, (1,3,3), stride, padding = (0,2,2), bias = False, dilation = 2)
        self.bn3 = nn.BatchNorm3d(output_channels//self.N)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(output_channels//self.N, output_channels, 1, 1, bias = False)
        self.conv4 = nn.Conv3d(input_channels, output_channels , 1, stride, bias = False)
        nn.init.xavier_uniform(self.conv1.weight.data)
        nn.init.xavier_uniform(self.conv2.weight.data)
        nn.init.xavier_uniform(self.conv3.weight.data)
        nn.init.xavier_uniform(self.conv4.weight.data)
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out1 = self.relu(out)
        out = self.conv1(out1)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out1)
        out += residual
        return out

class ResidualBlock3DAttention(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock3DAttention, self).__init__()
        self.channel_attention_1 = SELayer3D_modify(input_channels)
    def forward(self, x):
        residual = x
        out = self.channel_attention_1(x) 
        out += residual
        return out
