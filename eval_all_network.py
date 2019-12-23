#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-05 14:47
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : eval_all_network.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import argparse
import sys
import numpy as np
sys.path.append('.')

from tools.utils.video_funcs import default_aggregation_func
from tools.utils.metrics import mean_class_accuracy
#------------------------------------------------------------------------------------
#Usage: python eval_scores.py RGB_SCORE_FILE FLOW_SCORE_FILE --score_weights 1 1.5
#
#------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--multi_crop', nargs='+', type=int, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()

num_classes = 51
score_npz_files = [np.load(x) for x in args.score_files]
print(len(score_npz_files))

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

score_list = [x['scores'][:, 0] for x in score_npz_files]
label_list = [x['labels'] for x in score_npz_files]
agg_score_list = [] # 3783 x 1?
#Notice: here label list may be not same, (due to shuffle)
count = 0

for score_vec in score_list:
    #x:10 x 1 x 101
    agg_score_vec = [default_aggregation_func(np.reshape(x,(1,1,num_classes)), normalization=False) for x in score_vec]
    agg_score_list.append(np.array(agg_score_vec))
    count = count + 1

final_scores = np.zeros_like(agg_score_list[0])
for i, agg_score in enumerate(agg_score_list):
    final_scores += agg_score * score_weights[i]
acc = mean_class_accuracy(final_scores, label_list[0])
#print 'Final accuracy {:02f}%'.format(acc * 100)
sys.exit(acc * 100)