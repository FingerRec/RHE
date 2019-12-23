#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-15 21:41
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : change_something_path.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

out = []
count = 0
with open('data/something_something_v1/test_videofolder.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        path = line.strip().split(' ')[0]
        #label = line.strip().split(' ')[2]
        frame_num = line.strip().split(' ')[1]
        #new_line = path + ' ' + str(int(frame_num)-1) + ' ' + label + ' ' + '\n'
        new_line = path + ' ' + str(int(frame_num) - 1) + ' ' + '\n'
        out.append(new_line)

with open('data/something_something_v1/flow_test_videofolder.txt', 'a') as f:
    f.writelines(out)