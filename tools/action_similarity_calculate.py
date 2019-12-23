#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-15 09:40
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : action_similarity_calculate.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import os
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
from sklearn.metrics.pairwise import euclidean_distances
from net.i3d_2fc import InceptionI3d
from torchvision import transforms
import videotransforms
from torch.autograd import Variable
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='0'

#===============================Step1. Construct I3D Model and dataloader ==================
def constuct_i3d():
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('pretrained_models/rgb_imagenet.pt'))
    i3d.cuda()
    return i3d

#===============================Step2. Calculate Kinetics feature matrix A================
def kinetics_feature_extrac(model):
    """
    each row represent a class,
    :return: a numpy array with [400 x feature_size]
    """
    from dataset.kinetics_dataset import I3dDataSet
    video_transform = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = I3dDataSet("", 'data/kinetics_rgb_4.txt', num_segments=1,
                         new_length=64,
                         modality='rgb',
                         dataset='kinetics',
                         image_tmpl="img_{:05d}.jpg",
                         transform=video_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    feature = np.zeros([400, 1024])
    each_class_num = np.ones(400)
    save_dir = "features/"
    count = 0
    model.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for data in dataloader:
        count += 1
        _, rest = divmod(count, 100)
        if rest == 0:
            print(count)
        # get the inputs 1 x 3 x 64 x 224 x 224 labels 38(1)
        inputs, labels = data
        #print(labels)
        each_class_num[labels] += 1
        inputs = Variable(inputs.cuda(), volatile=True)
        features = model.extract_features(inputs)  # 1 x 1024 x 7 x 1 x 1
        save_feature = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()  # 7 x 1 x 1 x 1024
        save_feature = np.reshape(np.mean(save_feature, axis=0), [1024])
        feature[labels] += save_feature
    for i in range(400):
        feature[i] = feature[i] / each_class_num[i]
    np.save(os.path.join(save_dir, "kinetics_action_similarity_mean_feature" + '.npy'), feature)
    return feature

def kinetics_video_feature_extrac(model):
    """
    each row represent a class,
    :return: a numpy array with [400 x feature_size]
    """
    from dataset.dataset import VideoDataSet
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    val_data_set = VideoDataSet("/home/manager/disk1_6T/Share_Folder/wjp",
                            "data/kinetics_video_trainlist.txt",
                            data_set='kinetics',
                            new_length=64,
                            test_mode=False,
                            modality='rgb',
                            random_shift=False,
                            transform=test_transforms
                                )
    dataloader = torch.utils.data.DataLoader(val_data_set, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    feature = np.zeros([400, 1024])
    each_class_num = np.ones(400)
    save_dir = "features/"
    count = 0
    model.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for data in dataloader:
        count += 1
        _, rest = divmod(count, 100)
        if rest == 0:
            print(count)
        # get the inputs 1 x 3 x 64 x 224 x 224 labels 38(1)
        inputs, labels = data
        batch_size = inputs.size(0)
        inputs = Variable(inputs.cuda(), volatile=True)
        features = model.extract_features(inputs)  # 1 x 1024 x 7 x 1 x 1
        for i in range(batch_size):
            #print(labels)
            save_feature = features[i].squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()  # 7 x 1 x 1 x 1024
            save_feature = np.reshape(np.mean(save_feature, axis=0), [1024])
            feature[labels] += save_feature[i]
            each_class_num[labels[i]] += 1
    for i in range(400):
        feature[i] = feature[i] / each_class_num[i]
    np.save(os.path.join(save_dir, "kinetics_action_similarity_mean_feature" + '.npy'), feature)
    return feature
#===============================Step3. Calculate UCF101 feature matrix  B=================
def ucf101_feature_extract(model):
    """
    each row represent a class,
    :return: a numpy array with [101 x feature_size]
    """
    from dataset.ucf101_dataset import I3dDataSet
    video_transform = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = I3dDataSet("", 'data/ucf101_rgb_train_split_1.txt', num_segments=1,
                         new_length=64,
                         modality='rgb',
                         dataset='ucf101',
                         image_tmpl="frame{:06d}.jpg",
                         transform=video_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    feature = np.zeros([101, 1024])
    each_class_num = np.ones(101)
    save_dir = "features/"
    count = 0
    model.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for data in dataloader:
        count += 1
        _, rest = divmod(count, 100)
        if rest == 0:
            print(count)
        # get the inputs 1 x 3 x 64 x 224 x 224 labels 38(1)
        inputs, labels = data
        each_class_num[labels] += 1
        inputs = Variable(inputs.cuda(), volatile=True)
        features = model.extract_features(inputs)  # 1 x 1024 x 7 x 1 x 1
        save_feature = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy()  # 7 x 1 x 1 x 1024
        save_feature = np.reshape(np.mean(save_feature, axis=0), [1024])
        feature[labels] += save_feature
    for i in range(101):
        feature[i] = feature[i] / each_class_num[i]
    np.save(os.path.join(save_dir, "ucf101_action_similarity_mean_feature" + '.npy'), feature)
    return feature
#===============================Step4. Calculate Eucildean distance ======================
def eucildean_distance(A, B):
    return euclidean_distances(A, B)
#===============================Step5. Calculate Earth Move distance =====================
def earth_move_distance(A, B):
    return euclidean_distances(A, B)

def construct_simliar_list(E, output_path):
    """
    the input is a matrix with n * n
    :param E:
    :param output_path:
    :return:
    """
    print(E)
    similar_list = list()
    for i in range(101):
        record = "{} \n".format(np.argsort(E[i])[1])
        similar_list.extend(record)
    with open(output_path, 'w') as f:
        f.writelines(similar_list)

if __name__ == '__main__':
    #model = constuct_i3d()
    #A = kinetics_video_feature_extrac(model)
    #B = ucf101_feature_extract(model)
    A = np.load("features/kinetics_action_similarity_mean_feature" + '.npy')
    A = A / 255
    B = np.load("features/ucf101_action_similarity_mean_feature" + '.npy')
    C = eucildean_distance(A, B)
    print(C)
    result = np.mean(C, 1)
    print(np.argsort(result))
    #D = earth_move_distance(A, B)
    #E = eucildean_distance(B, B)
    #output_path = "data/ucf101_triplet_similar_rgb.txt"
    #construct_simliar_list(E, output_path)