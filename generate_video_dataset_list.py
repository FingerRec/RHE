#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-15 13:16
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : generate_video_dataset_list.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import cv2

cap = cv2.VideoCapture("video.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )