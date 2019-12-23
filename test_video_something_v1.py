#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-05-21 09:56
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : test_video_something_v1.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
import argparse
import time

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from net.multi_path_i3d import MultiPathI3d

from dataset.ucf101_dataset import I3dDataSet

# from net import TSN
import videotransforms
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--root', type=str, default="")
parser.add_argument('--dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'something_something_v1'])
parser.add_argument('--mode', type=str, choices=['rgb', 'flow'])
parser.add_argument('--test_list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str, default="mpi3d", choices=['mpi3d_pt', 'mpi3d'])
parser.add_argument('--save_scores', type=str, default="test_output/save_scores")
parser.add_argument('--test_clips', type=int, default=10)
parser.add_argument('--clip_size', type=int, default=64)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

# ******************************************************************************************
# here we follow the test procedure report as Non-Local Neural Network
# 1.sample 10 clips in temporal
# 2.for each clip, img is 256 x 320 we clip 3 256 x 256 and compress them togeher
# 3.input is 256 x 256
# ******************************************************************************************
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_transform(model_dict, pretrain_dict):
    '''

    :return:
    '''
    weight_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(weight_dict)
    return model_dict

def plot_confuse_matrix(matrix, classes,
                        normalize=True,
                        title=None,
                        cmap=plt.cm.Blues
                        ):
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
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    return ax


def get_action_index():
    action_label = []
    with open('data/classInd.txt') as f:
        content = f.readlines()
        content = [x.strip('\r\n') for x in content]
    f.close()
    for line in content:
        label, action = line.split(' ')
        action_label.append(action)
    return action_label


def plot_matrix_test():
    classes = get_action_index()
    confuse_matrix = np.load("test_output/ucf101_confusion.npy")
    plot_confuse_matrix(confuse_matrix, classes)
    plt.show()


def main():
    if args.dataset == 'ucf101':
        num_class = 101
        data_length = 300 #the video frames to load
        image_tmpl = "frame{:06d}.jpg"
        dense_sample = False
    elif args.dataset == 'hmdb51':
        num_class = 51
        data_length = 200
        image_tmpl = "img_{:05d}.jpg"
        dense_sample = False
    elif args.dataset == 'kinetics':
        num_class = 400
        data_length = 300
        image_tmpl = "frame{:06d}.jpg"
        dense_sample = False
    elif args.dataset == 'something_something_v1':
        num_class = 174
        data_length = 200
        image_tmpl = "{:05d}.jpg"
        dense_sample = False
    else:
        raise ValueError('Unknown dataset ' + args.dataset)
    test_transforms = transforms.Compose([videotransforms.VideoCrop(224)]) # 256 or 224?
    #test_transforms = transforms.Compose([videotransforms.CornerCrop(224)])
    if args.mode == 'rgb':
        print("construct {}".format(args.arch))
        if args.arch == 'mpi3d':
            net = MultiPathI3d(num_classes=num_class, in_channels=3, dropout_prob=args.dropout)
        elif args.arch == 'mpi3d_pt':
            from net.multi_path_i3d_pt import I3D
            net = I3D(num_class,modality='rgb',dropout_prob=0)
        else:
            Exception("not implement error")
    elif args.mode == 'flow':
        print("construct {}".format(args.arch))
        if args.arch == 'mpi3d':
            net = MultiPathI3d(num_classes=num_class, in_channels=2, dropout_prob=args.dropout)
        elif args.arch == 'mpi3d_pt':
            from net.multi_path_i3d_pt import I3D
            net = I3D(num_class,modality='flow',dropout_prob=0)
        else:
            Exception("not implement error")
    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    if args.arch == 'mpi3d':
        model_dict = net.state_dict()
        model_dict = weight_transform(model_dict, base_dict)
        net.load_state_dict(model_dict)
    else:
        net.load_state_dict(base_dict)

    val_dataset = I3dDataSet(args.root, args.test_list, num_segments=1,
                             new_length=data_length,
                             modality=args.mode,
                             test_mode=True,
                             dataset=args.dataset,
                             image_tmpl=image_tmpl if args.mode in ["rgb", 'RGB',
                                                                    "RGBDiff"] else args.flow_prefix + "{}_{:05d}.jpg",
                             random_shift=False,
                             transform=test_transforms,
                             full_video = True)
    data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.batch_size,
                                              pin_memory=True)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()

    data_gen = enumerate(data_loader)
    total_num = len(data_loader.dataset)  # 3783
    max_num = len(data_loader.dataset)

    print("total test num", total_num)

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def plot_point(a, b):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(a, b, 'o')
        plt.title("point figure")
        plt.show()
    """
    def eval_video(video_data):
        '''
        average 10 clips, do it later
        '''
        i, datas, label = video_data
        # data length is 250, get 20 clips and get the average result
        output = None
        # print(len(datas))  # 1 x 3 x 250 x 224 x 224
        with torch.no_grad():
            for data in datas:
                if len(data.size()) == 4:
                    data = torch.unsqueeze(data, 0)
                for i in range(args.test_clips):
                    # print(data.size())
                    index = random.randint(1, data_length - args.clip_size)
                    #print(index)
                    clip_data = data[:, :, index : index + args.clip_size, :, :]
                    input_var = torch.autograd.Variable(clip_data)
                    if output is None:
                        main_path, _, _, _ = net(input_var, alpha=0.5)
                        output = main_path.data.cpu().numpy().copy() / args.test_clips * 5
                    else:
                        main_path, _, _, _ = net(input_var, alpha=0.5)
                        output += main_path.data.cpu().numpy().copy() / args.test_clips * 5
        return output, label
    """


    def eval_one_video(video_data):
        '''
        average 10 clips, do it later
        '''
        _, datas, label = video_data
        # data length is 250, get 20 clips and get the average result
        output = None
        with torch.no_grad():
            for data in datas:#1 x 3 x 200 x 256 x 256
                if len(data.size()) == 4:
                    data = torch.unsqueeze(data, 0)
                    coeff = 1
                else:
                    #coeff = 5 # for corner clip
                    coeff = 3 # for full video windows
                for i in range(args.test_clips):
                    index = i * (data_length - args.clip_size)//args.test_clips
                    if dense_sample:
                        clip_data = data[:, :, 0:data_length, :, :]
                    else:
                        clip_data = data[:, :, index: index + args.clip_size, :, :]
                    input_var = torch.autograd.Variable(clip_data)
                    main_path, path_s, path_m, path_l = net(input_var)
                    one_output = main_path + path_m + path_l
                    #one_output = main_path + 1/2*path_m + 1/3*path_l + 1/4*path_s
                    #one_output = torch.max(main_path, path_m)
                    #one_output = torch.max(one_output, path_l)
                    #one_output = torch.max(one_output, path_s)
                    if output is None:
                        output = one_output.data.cpu().numpy().copy() / args.test_clips / coeff
                    else:
                        output += one_output.data.cpu().numpy().copy() / args.test_clips / coeff
                        #output = max(one_output.data.cpu().numpy().copy().all() / args.test_clips / coeff, output.all())  # max
        return output, label

    output = []
    print('*' * 50)
    data_gen = enumerate(data_loader)
    proc_start_time = time.time()
    for i, (data, label) in data_gen:
        rst = eval_one_video((i, data, label))
        output.append(list(rst))
        cnt_time = time.time() - proc_start_time
        if i % 10 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i + 1,
                                                                            total_num,
                                                                            float(cnt_time) / (i + 1)))
            if i > 300:
                video_pred = [np.argmax(x[0]) for x in output]
                video_labels = [x[1] for x in output]
                cf = confusion_matrix(video_labels, video_pred).astype(float)
                cls_cnt = cf.sum(axis=1)
                cls_hit = np.diag(cf)
                cls_acc = cls_hit / cls_cnt
                print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    # =====output: every video's num and every video's label
    # =====x[0]:softmax value x[1]:label
    video_pred = [np.argmax(x[0]) for x in output]
    np.save("test_output/" + args.dataset + '/' + args.mode + '_' + str(np.mean(cls_acc)*100)[:5] + '_' + "video_pred.npy", video_pred)
    video_labels = [x[1] for x in output]
    np.save("test_output/" + args.dataset + '/' + args.mode + '_' + str(np.mean(cls_acc)*100)[:5] + '_' + "video_labels.npy", video_labels)
    cf = confusion_matrix(video_labels, video_pred).astype(float)
    np.save("test_output/" + args.dataset + '/' + args.mode + '_' + str(np.mean(cls_acc)*100)[:5] + '_' + args.dataset + "_confusion.npy", cf)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt
    print(cls_acc)
    print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))
    if args.save_scores is not None:
        # reorder before saving
        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e: i for i, e in enumerate(sorted(name_list))}
        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
        try:
            np.savez('test_output/' + args.dataset + '/' + args.mode + '_' + str(np.mean(cls_acc)*100)[:5] + '_' + args.dataset + '_' + args.mode + '_' + args.save_scores, scores=reorder_output, labels=reorder_label)
        except TypeError:
            np.savez('test_output/temp', scores=reorder_output, labels=reorder_label)

if __name__ == '__main__':
    main()
    #plot_matrix_test()
