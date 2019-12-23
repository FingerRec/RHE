#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-23 20:50
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : generate_mini_kinetics.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
#=========================train============================================
'''
with open('data/kinetics_video_trainlist.txt','r') as f:
    older_lines = f.readlines()

out = []
count = 0
with open('data/train_ytid_list.txt','r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        for old_line in older_lines:
            temp = old_line.strip().split(' ')[0]
            if line in temp:
                count +=1
                out.append(old_line)
        if count % 100 == 0 and count != 0:
            print(count)
            
with open('data/mini_kinetics_video_trainlist.txt', 'a') as f:
    f.writelines(out)
'''
with open('data/kinetics_video_vallist.txt', 'r') as f:
    older_lines = f.readlines()

out = []
count = 0
with open('data/val_ytid_list.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        for old_line in older_lines:
            temp = old_line.strip().split(' ')[0]
            if line in temp:
                count += 1
                out.append(old_line)
        if count % 100 == 0 and count != 0:
            print(count)

with open('data/mini_kinetics_video_vallist.txt', 'a') as f:
    f.writelines(out)