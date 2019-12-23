#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-03-15 21:19
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : triplet_dataset.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import cv2
import torch
import random
from videotransforms import transform_data


# from config import opt
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    # return torch.from_numpy(pic)
    return torch.from_numpy(pic.transpose([3, 0, 1, 2])).type(torch.FloatTensor)


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TripletDataSet(data.Dataset):
    def __init__(self, root_path, list_file, similiar_list_file, dataset='ucf101',
                 new_length=64, modality='rgb',
                 image_tmpl='img_{:06d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.simiar_list_file = similiar_list_file
        self.new_length = new_length
        self.modality = modality
        self.dataset = dataset
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        if self.test_mode:
            self.test_frames = 250
        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()  # get video list

    def _load_image(self, directory, idx):
        if self.modality == 'rgb':
            img = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
            return [img]

    def _triplet_list(self, video_list):
        action_record = {}
        for i in range(101):
            action_record[str(i)] = list()
        for i in range(len(video_list)):
            action_record[str(video_list[i].label)].append(i)
        return action_record

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        self.parse_dict = self._triplet_list(self.video_list)
        self.simiar_list = [x.strip().split(' ') for x in open(self.simiar_list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if not self.test_mode:
            index = random.randint(1, max(record.num_frames - self.new_length, 0) + 1)
        else:
            index = 5
        return index

    def __getitem__(self, index):
        """
        if test mode:
        :param index:
        :return:
        """
        def _get_data(_record, _data_augment=True):
            segment_indices = self._sample_indices(record)
            _data, _label = self.get(record, segment_indices, data_augment=_data_augment)
            _data = 2 * (_data / 255) - 1
            _data = video_to_tensor(_data)
            return _data, _label
        record = self.video_list[index]
        if not self.test_mode:
            data_ai, label = _get_data(record, _data_augment=True)

            one_class_action = self.parse_dict[str(record.label)]
            while 1==1:
                index_2 = one_class_action[randint(0, len(one_class_action))]
                record_2 = self.video_list[index_2]
                if record_2.path != record.path:
                    break
            data_ap, _ = _get_data(record_2, _data_augment=True)
            nearest_class_label = self.simiar_list[record.label]
            nearest_class_action = self.parse_dict[nearest_class_label[0]]
            index_3 = nearest_class_action[randint(0, len(nearest_class_action))]
            record_3 = self.video_list[index_3]
            data_ni, label_np = _get_data(record_3, _data_augment=True)

            return data_ai, data_ap, data_ni, label, label_np
        else:
            return _get_data(record, _data_augment=False)

    def get(self, record, indices, data_augment=False, side_length=224, is_numpy=True):
        images = list()
        p = int(indices)
        for i in range(self.new_length):
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
            if p < record.num_frames:
                p += 1
            else:
                p = 1
        images = transform_data(images, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        if is_numpy:
            frames_up = []
            if self.modality == 'rgb':
                for i, img in enumerate(images):
                    frames_up.append(np.asarray(img))
            elif self.tag == 'flow':
                for i in range(0, len(images), 2):
                    # it is used to combine frame into 2 channels
                    tmp = np.stack([np.asarray(images[i]), np.asarray(images[i + 1])], axis=2)
                    frames_up.append(tmp)
            return np.stack(frames_up), record.label
        return images, record.label

    def __len__(self):
        return len(self.video_list)
