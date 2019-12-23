#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-06-25 15:24
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : SimilarityMetric.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import torch
import torch.nn as nn


def L2Confusion(features):
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')
    batch_left = features[:int(0.5*batch_size)]
    batch_right = features[int(0.5*batch_size):]
    loss = torch.norm((batch_left - batch_right).abs(),2, 1).sum() / float(batch_size)
    return loss


def EntropicConfusion(features):
    batch_size = features.size(0)
    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)


#======a loss should be here, for the same class, the distance should be not too away
def PairwiseLoss(out_logits, target_var):
    intra_lambda_coeff = 1e-2
    inter_lambda_coeff = 1e-2
    intra_loss = 0
    inter_loss = 0
    b, c = out_logits.size()
    pdist = nn.PairwiseDistance(p=2)
    #loss = torch.norm((out_logits - out_logits).abs(), 2, 1).sum() / float(b)
    for i in range(b):
        for j in range(b//2, b):
            if i == j:
                continue
            if target_var[i] == target_var[j]:
                #print("intra")
                intra_loss += pdist(out_logits[i].view(1, c), out_logits[j].view(1, c))
            else:
                inter_loss += pdist(out_logits[i].view(1, c), out_logits[j].view(1, c))
    # print(intra_loss, inter_loss)
    pairwiseLoss = intra_lambda_coeff * intra_loss + inter_lambda_coeff * inter_loss
    return pairwiseLoss.mean()


def MeanLoss(logits, mean, targets):
    b,c = logits.size()
    intra_lambda_coeff = 1e-1
    inter_lambda_coeff = 1e-1
    intra_loss = 0
    inter_loss = 0
    # print(mean)
    pdist = nn.PairwiseDistance(p=2)

    for i in range(b):
        intra_loss += intra_lambda_coeff * pdist(logits[i].view(1, c), mean[:, int(targets[i].item())])

    for i in range(b):
        for j in range(i+1, b):
            if targets[i] == targets[j]:
                continue
            else:
                inter_loss += inter_lambda_coeff * nn.functional.cosine_similarity(logits[i].view(1, c), logits[j].view(1, c))
    if inter_loss != 0:
        return intra_loss.mean() + inter_loss.mean()
    else:
        return intra_loss.mean()


def DiversityLoss(features):
    """
    for feature in temporal, insure each feature are not same, long distance, the feature map should not same bigger,
    :param features:
    :return:
    """
    #print(features[0,:,0,:,:].size())
    intra_lambda_coeff = 1e-2 # this should be gradually increase
    diversity_loss = 0
    pdist = nn.PairwiseDistance(p=2)
    b, c, t, h, w = features.size()
    for i in range(b):
        for j in range(t-1):
            diversity_loss += nn.functional.cosine_similarity(features[i,:,j,:,:].contiguous().view(1,c*h*w), features[i,:,j+1,:,:].contiguous().view(1,c*h*w))
            #diversity_loss += pdist(features[i,:,j,:,:].contiguous().view(1,c*h*w), features[i,:,j+1,:,:].contiguous().view(1,c*h*w))
    diversity_loss *= intra_lambda_coeff
    #print(diversity_loss)
    return diversity_loss.mean()


def HeatmapDiversityLoss(features):
    """
    for feature in temporal, the distance larger, the similarity should be smaller
    :return:
    """
    #print(features[0,:,0,:,:].size())
    intra_lambda_coeff = 1e-4 # this should be gradually increase
    diversity_loss = 0
    pdist = nn.PairwiseDistance(p=2)
    b, c, t, h, w = features.size()
    for i in range(b):
        for j in range(t-1):
            for k in range(j+1, t-1):
                #diversity_loss += nn.functional.cosine_similarity(features[i,:,j,:,:].contiguous().view(1,c*h*w), features[i,:,j+1,:,:].contiguous().view(1,c*h*w))
                diversity_loss += (k-j) * pdist(features[i,:,j,:,:].contiguous().view(1,c*h*w), features[i,:,k,:,:].contiguous().view(1,c*h*w))
    diversity_loss *= intra_lambda_coeff
    #print(diversity_loss)
    return diversity_loss.mean()