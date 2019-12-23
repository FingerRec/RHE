#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-21 15:19
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : train_charades_i3d.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str, default='/home/wjp/Desktop/disk2_6T/DataSet/charades/Charades_v1_rgb/')
parser.add_argument('-video_split', type=str, default='dataset/charades.json')
parser.add_argument('-lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--arch',default='i3d', type=str, choices=['i3d', 'mpi3d', 'nli3d', 'mfnet', 'fli3d'])
parser.add_argument('--lr_steps', default=[10, 20, 25, 30, 35, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('-batch_size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--gpus', type=str, default="0",
                    help="define gpu id")
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np

from net.charades_i3d import CharadesI3d

from dataset.charades_dataset import Charades as Dataset
import tools.utils.charades_checkpoints as checkpoints
from utils import *
import tools.utils.map as map
import time


best_mAP = 0

def main():
    global best_mAP
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    train_dataset = Dataset(args.video_split, 'training', args.root, args.mode, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16,
                                             pin_memory=True)

    val_dataset = Dataset(args.video_split, 'testing', args.root, args.mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=16,
                                                 pin_memory=True)

    val_video = Dataset(args.video_split, 'val_video', args.root, args.mode, test_transforms)
    val_video_dataloader = torch.utils.data.DataLoader(val_video, batch_size=args.batch_size*4, shuffle=False, num_workers=16,
                                                 pin_memory=True)
    num_classes = 157
    # setup the model
    if args.mode == 'flow':
        i3d = CharadesI3d(num_classes, in_channels=2)
        i3d.load_state_dict(torch.load('pretrained_models/flow_imagenet.pt'))
    else:
        i3d = CharadesI3d(num_classes, in_channels=3)
        pretrain_dict = torch.load('pretrained_models/rgb_imagenet.pt')
        model_dict = i3d.state_dict()
        model_dict = weight_transform(model_dict, pretrain_dict)
        i3d.load_state_dict(model_dict)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    optimizer = optim.SGD(i3d.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0000001)
    criterion = F.binary_cross_entropy_with_logits
    model = i3d
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)
        #now we juse use a clip to test our code, need to modify later
        cls_train_loss, loc_train_loss = train(train_dataloader, model, criterion, optimizer, epoch)
        cls_val_loss, loc_val_loss = validate(val_dataloader, model, criterion, epoch)
        mAP = validate_video(val_video_dataloader, model, epoch)
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        scores = {'cls_train_loss': cls_train_loss, 'loc_train_loss': loc_train_loss, 'cls_val_loss': cls_val_loss, 'loc_val_loss': loc_val_loss, 'mAP': mAP}
        checkpoints.save(epoch, args, model, optimizer, is_best, scores)

def weight_transform(model_dict, pretrain_dict):
    '''

    :return:
    '''
    weight_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(weight_dict)
    return model_dict

def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(train_loader, model, criterion, optimizer, epoch):
    tot_loss = AverageMeter()
    batch_time = AverageMeter()
    tot_loc_loss = AverageMeter()
    tot_cls_loss = AverageMeter()
    optimizer.zero_grad()
    model.train()
    # Iterate over data.
    end = time.time()
    for i, (vid, inputs, labels)  in enumerate(train_loader):
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())
        per_frame_logits = model(inputs)
        #per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
        per_frame_logits = nn.functional.interpolate(per_frame_logits, t, mode='linear', align_corners=True)
        loc_loss = criterion(per_frame_logits, labels)
        tot_loc_loss.update(loc_loss.item(), inputs.size(0))
        # compute classification loss (with max-pooling along time B x C x T)
        cls_loss = criterion(torch.max(per_frame_logits, dim=2)[0],
                                                      torch.max(labels, dim=2)[0])
        tot_cls_loss.update(cls_loss.item(), inputs.size(0))
        loss = (0.5 * loc_loss + 0.5 * cls_loss)
        tot_loss.update(loss.item(), inputs.size(0))
        loss.backward()
        batch_time.update(time.time()-end)
        end = time.time()
        optimizer.step()
        optimizer.zero_grad()
        if i % args.print_freq == 0:
            print('Epoch: [{epoch}] \t'
                  'Train: [{0}/{1} ({2}/{3})]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'ClsLoss {tot_cls_loss.val:.4f} ({tot_cls_loss.avg:.4f})\t'
                  'LocLoss {tot_loc_loss.val:.4f} ({tot_loc_loss.avg:.4f})\t'
                  'TotalLoss {tot_loss.val:.4f} ({tot_loss.avg:.4f})'.format(
                i, len(train_loader), i*args.batch_size, int(len(train_loader) * args.batch_size),
                batch_time=batch_time, tot_cls_loss=tot_cls_loss, tot_loc_loss=tot_loc_loss, epoch=epoch,
            tot_loss=tot_loss))

    return cls_loss.item(), loc_loss.item()

def validate(train_loader, model, criterion, epoch):
    tot_loss = AverageMeter()
    batch_time = AverageMeter()
    tot_loc_loss = AverageMeter()
    tot_cls_loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i,(vid, inputs, labels) in enumerate(train_loader):
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)
            labels = Variable(labels.cuda())
            per_frame_logits = model(inputs)
            #per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
            per_frame_logits = nn.functional.interpolate(per_frame_logits, t, mode='linear', align_corners=True)
            # compute localization loss
            loc_loss = criterion(per_frame_logits, labels)
            tot_loc_loss.update(loc_loss.item(), inputs.size(0))
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = criterion(torch.max(per_frame_logits, dim=2)[0],
                                                          torch.max(labels, dim=2)[0])
            tot_cls_loss.update(cls_loss.item(), inputs.size(0))
            loss = (0.5 * loc_loss + 0.5 * cls_loss)
            tot_loss.update(loss.item(), inputs.size(0))

            if i % args.print_freq == 0:
                print('Epoch: [{epoch}] '
                      'Test: [{0}/{1} ({2}/{3})]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'ClsLoss {tot_cls_loss.val:.4f} ({tot_cls_loss.avg:.4f})\t'
                      'LocLoss {tot_loc_loss.val:.4f} ({tot_loc_loss.avg:.4f})\t'
                      'TotalLoss {tot_loss.val:.4f} ({tot_loss.avg:.4f})'.format(
                    i, len(train_loader), i*args.batch_size,int(len(train_loader) * args.batch_size),
                    batch_time=batch_time, tot_cls_loss=tot_cls_loss, tot_loc_loss=tot_loc_loss, epoch=epoch,
                tot_loss=tot_loss))
    return cls_loss.item(), loc_loss.item()

def validate_video(validata_loader, model, epoch):
    """ Run video-level validation on the Charades test set"""
    #all video should use here, first just use a clip
    batch_time = AverageMeter()
    outputs = []
    gts = []
    ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # loader: 1863 labels, each is a video, in a video have serveral actions data: image_paths: 45950 targets:45950, each has 159 action vector, such as [0, 0, 0, 0, ...., 1, 1,1 ,0 ...0]
    # ids:45950, each is str, such as 'YSKX3'
    # input: 25 x 3 x 224 x 224 target: 25 x 157 meta:(list) 0 - 24, each is str like 'YSKX3'
    # actions may be overlap, for example. lbels{dict}: 'YSKX3': 0-['c077', 12.1, 18.0], 1-['c079', 11.8, 17.3]
    # our target is 157 x 64
    with torch.no_grad():
        for i, (vid, inputs, target) in enumerate(validata_loader):
            target = Variable(target.cuda())
            input_var = torch.autograd.Variable(inputs.cuda())
            output = model(input_var)
            output_video = torch.max(output, dim=2)[0]
            outputs.append(output_video.data.cpu().numpy())
            gts.append(target)  # list: each elements is a 159 vector
            ids.append(vid)  # ids:video id, list elements: 'YSKX3'
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{epoch}] Test_video: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                    i, len(validata_loader), batch_time=batch_time, epoch=epoch))
    # mAP, _, ap = map.map(np.vstack(outputs), np.vstack(gts))
    # outputs:1838 x 157(157 float) gts: 1838 x 157 gts is multi-label vector, such as [0, 0, 0, 0, 0, ...., 1, 1, 1, 0]
    mAP, _, ap = map.charades_map(np.vstack(outputs), np.vstack(gts))
    print(ap)  # mean ap is sum(ap) / 157
    print(' * mAP {:.3f}'.format(mAP))
    #submission_file(
    #    ids, outputs, 'test_output/charades/epoch_{:03d}.txt'.format(epoch + 1))
    return mAP


if __name__ == '__main__':
    main()
