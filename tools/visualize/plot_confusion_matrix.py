#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-04-28 22:25
     # @Author  : Awiny
     # @Site    :
     # @Project : amax-pytorch-i3d
     # @File    : plot_confusion_matrix.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning


def plot_confuse_matrix(matrix, classes,normalize=True,title=None,cmap=plt.cm.Blues):
    """

    :param matrix:
    :param classes:
    :param normalize:
    :param title:
    :param cmap:
    :return:
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix on hmdb51'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix on hmdb51")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=80, fontsize=5, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), fontsize=5)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] > 0.1 and i != j:
                ax.text(j, i, format("({},{}):".format(classes[i],classes[j])) + format(cm[i, j], fmt),
                        ha="center", va="center", fontsize=5,
                        color="black")
            '''
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            '''
    # fig.tight_layout()
    return ax


def get_action_index():
    action_label = []
    with open('../../data/hmdb51/hmdb51_classInd.txt') as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    for line in content:
        label, action = line.split(' ')
        action_label.append(action)
    return action_label


def plot_matrix_test():
    classes = get_action_index()
    confuse_matrix = np.load("../../test_output/hmdb51/43.57_hmdb51_confusion.npy")
    plot_confuse_matrix(confuse_matrix, classes)
    plt.tight_layout()
    plt.savefig("hmdb51_confusion_test_2.png", dpi=1080)
    #plt.show()


if __name__ == '__main__':
    plot_matrix_test()
