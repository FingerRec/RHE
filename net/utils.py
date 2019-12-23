#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-22 22:26
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : utils.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

def transfer_learning_stragety(model, lr):
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
