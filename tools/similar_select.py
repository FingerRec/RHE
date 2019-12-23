#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-12 20:17
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : similar_select.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import os
import numpy as np

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
dataset = I3dDataSet("", 'data/ucf101_rgb_train_split_1.txt', num_segments=1,
                     new_length=64,
                     modality='rgb',
                     dataset='ucf101',
                     image_tmpl="frame{:06d}.jpg",
                     transform=video_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

val_dataset = I3dDataSet("", 'data/ucf101_rgb_val_split_1.txt', num_segments=1,
                         new_length=64,
                         modality='rgb',
                         dataset='ucf101',
                         image_tmpl="frame{:06d}.jpg",
                         random_shift=False,
                         transform=video_transform)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

dataloaders = {'train': dataloader, 'val': val_dataloader}

#2.2 load label and extract features
# origin_action
ucf_action_label = [x.strip().split()[1] for x in open('data/classInd.txt')]
kinetics_action_label = list(json.loads(open('data/classids.json').read()).keys())
kinetics_count = np.zeros(400)
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
        #if os.path.exists(os.path.join(save_dir, "feature" + '.npy')):
        #    continue

        b, c, t, h, w = inputs.shape
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        output = i3d(inputs)
        probality = np.max(output.data.cpu().numpy())
        index = int(np.argmax(output.data.cpu().numpy()))
        if probality > 0.6:
            kinetics_count[index] += 1
        print("ucf label: {}, kinetics label: {} with prob {}".format(ucf_action_label[int(labels.cpu().numpy())], kinetics_action_label[index], probality))
np.save("data/ucf101_to_kinetics_count.npy", kinetics_count)
'''
#========================================Step3 load kinetics_count numpy and select sub classes
kinetics_count = np.load("data/ucf101_to_kinetics_count.npy")
index = kinetics_count.argsort()[-101:][::-1]
print(index)

#========================================Step4 construct train examples for kinetics and retrain serveral epoches

list_file = 'data/kinetics_rgb_train_list.txt'
video_list = [(x.strip().split(' ')) for x in open(list_file)]
video_num = len(video_list)
retrain_rgb_list = list()
retrain_file = 'data/kinetics_rgb_retrain_list.txt'
count = 0
for i in range(video_num):
    if int(video_list[i][2]) in index:
        count += 1
        retrain_rgb_list.append('{} {} {}\n'.format(video_list[i][0], video_list[i][1], video_list[i][2]))
print("all training num is", count)
with open(retrain_file, 'w') as f:
    f.writelines(retrain_rgb_list)
