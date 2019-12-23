#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-11 20:26
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : knn_retraining.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os
import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import videotransforms
import numpy as np
#from pytorch_i3d import InceptionI3d
#from dataset.ucf101_dataset import I3dDataSet
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import scale

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
os.environ["CUDA_VISIBLE_DEVICES"]='2'

'''
#===========================================Step1: Load I3D Model==============================
# setup the model
video_transform = transforms.Compose([videotransforms.CenterCrop(224)])
i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('pretrained_models/rgb_imagenet.pt'))
i3d.cuda()
#===========================================Step2: Feature Extractor And Get Mean==============
# the easyest way is to use dataloader with shuffle = False
#2.1 Load DataSet
dataset = 'hmdb51'
if dataset == 'ucf101':
    num_class = 101
    data_length = 64
    image_tmpl = "frame{:06d}.jpg"
elif dataset == 'hmdb51':
    num_class = 51
    data_length = 64
    image_tmpl = "img_{:05d}.jpg"
elif dataset == 'kinetics':
    num_class = 400
    data_length = 64
    image_tmpl = "img_{:05d}.jpg"
else:
    raise ValueError('Unknown dataset ' + dataset)

dataset = I3dDataSet("", 'data/hmdb51_rgb_train_split_1.txt', num_segments=1,
                     new_length=64,
                     modality='rgb',
                     dataset=dataset,
                     image_tmpl=image_tmpl,
                     transform=video_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

val_dataset = I3dDataSet("", 'data/hmdb51_rgb_val_split_1.txt', num_segments=1,
                         new_length=64,
                         modality='rgb',
                         dataset=dataset,
                         image_tmpl=image_tmpl,
                         random_shift=False,
                         transform=video_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

dataloaders = {'train': dataloader, 'val': val_dataloader}

#2.2 extract features
feature = np.zeros([num_class, 1024])
each_class_num = np.ones(num_class)
save_dir = "features/"
count = 0
for phase in ['train', 'val']:
    i3d.train(False)  # Set model to evaluate mode

    # Iterate over data.
    for data in dataloaders[phase]:
        count += 1
        _, rest = divmod(count, 100)
        if rest == 0:
            print(count)
        # get the inputs 1 x 3 x 64 x 224 x 224 labels 38(1)
        inputs, labels = data
        each_class_num[labels] += 1
        #if os.path.exists(os.path.join(save_dir, "feature" + '.npy')):
        #    continue

        b, c, t, h, w = inputs.shape
        # wrap them in Variable
        inputs = Variable(inputs.cuda(), volatile=True)
        features = i3d.extract_features(inputs) # 1 x 1024 x 7 x 1 x 1
        save_feature = features.squeeze(0).permute(1, 2, 3, 0).data.cpu().numpy() # 7 x 1 x 1 x 1024
        save_feature = np.reshape(np.mean(save_feature, axis=0), [1024])
        feature[labels] += save_feature
for i in range(num_class):
    feature[i] = feature[i] / each_class_num[i]
np.save(os.path.join(save_dir, "hmdb51_mean_feature" + '.npy'), feature)
'''

#===========================================Step3: Use Kmeans/KNN to Cluster==========================
num_class=51
feature = np.load("features/hmdb51_mean_feature.npy")
#1000维数据维度过高，会引入欧式距离inflated问题，用PCA降维可以缓解
reduced_data = PCA(n_components=2).fit_transform(feature)
k_means = KMeans(init='k-means++', n_clusters=10, n_init=1000)
k_means.fit(reduced_data)
for i in range(num_class):
    if k_means.labels_[i] == 2:
        print(i+1)

print(k_means.labels_)
np.save('features/hmdb51_class_cluster_label.npy', k_means.labels_)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = k_means.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = k_means.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the HMDB51 dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#===========================================Step4: Change Training Stragety====================
#each train time, random select one video, and put videos together in one area
#construct a txt file(each batch is 20 and be formed of simliar action)
#loss is two part, smiliar loss and random loss

class_cluster_label = np.load('features/hmdb51_class_cluster_label.npy')
# return index
list_file = 'data/hmdb51_rgb_train_split_1.txt'
video_list = [(x.strip().split(' ')) for x in open(list_file)]
cluster = [[],[],[],[],[], [], [], [], [], []] #10 kmeans
videos_num = len(video_list)
for i in range(videos_num):
    index = class_cluster_label[int(video_list[i][2])]
    cluster[index].append(video_list[i])
cluster_list = cluster[0]
for i in range(len(cluster) - 1):
    cluster_list = np.concatenate((cluster_list, cluster[i + 1]), axis=0)

# written file
cluster_file = "data/hmdb51_rgb_train_cluster_split_1.txt"
rgb_list = list()
for i in range(len(cluster_list)):
    rgb_list.append('{} {} {}\n'.format(cluster_list[i][0], cluster_list[i][1], cluster_list[i][2]))
with open(cluster_file, 'w') as f:
    f.writelines(rgb_list)
