#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-06 12:44
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : change_kinetics_path.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

out = []
count = 0
with open('data/kinetics/kinetics_video_trainlist.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        path = line.strip().split(' ')[0]
        label = line.strip().split(' ')[1]
        new_line = path.split('/')[5]  + '/' + path.split('/')[6] + '/' + path.split('/')[7]  + ' ' + label + '\n'
        out.append(new_line)

with open('data/kinetics/ssd_kinetics_video_trainlist.txt', 'a') as f:
    f.writelines(out)