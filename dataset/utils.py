#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-15 16:30
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
import cv2
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        #print("could not open: ", video_path)
        return -1
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    return length, width, height


def drop_bound():
    img = Image.open('/media/awiny/9224670c-39f0-4964-8a14-7478ec574a6b/dataset/3D_hmdb51/push/Big_Bebe_walking_and_pushing_her_lion_in_a_box!_push_f_cm_np1_ri_bad_0/image_00005.jpg'
)
    img.show()
    new_img = img.load()
    print(img.size)
    seed = [img.size[0]//2, img.size[1]//2]
    w_1, w_2, h_1, h_2 = 0, img.size[0], 0, img.size[1]
    for i in range(img.size[0]//2-1):
        if cmp(new_img[seed[0] - i, seed[1]], (0, 0, 0)) is 0:
            w_1 = seed[0] - i
            break
    for i in range(img.size[0]//2-1):
        if cmp(new_img[seed[0] + i, seed[1]], (0, 0, 0)) is 0:
            w_2 = seed[0] + i
            break
    for i in range(img.size[1]//2-1):
        if cmp(new_img[seed[0], seed[1]-i], (0, 0, 0)) is 0:
            h_1 = seed[1]-i
            break
    for i in range(img.size[1]//2-1):
        if cmp(new_img[seed[0], seed[1]+i], (0, 0, 0)) is 0:
            h_2 = seed[1]+i
            break
    print(w_1, w_2, h_1, h_2)
    if w_1 or w_2 or h_1 or h_2:
        img2 = img.crop((w_1, h_1, w_2, h_2))
        img2.show()
    else:
        return img

# drop_bound()