#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
     # @Time    : 2019-07-17 19:56
     # @Author  : Awiny
     # @Site    :
     # @Project : pytorch_i3d
     # @File    : deformable_conv_dataset.py
     # @Software: PyCharm
     # @Github  : https://github.com/FingerRec
     # @Blog    : http://fingerrec.github.io
"""
import scipy.io
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #close the warning
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


class I3dDataSet(data.Dataset):
    def __init__(self, root_path, list_file, dataset='ucf101',
                 num_segments=1, new_length=64, stride=1, modality='rgb',
                 image_tmpl='img_{:06d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, full_video=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.stride = stride
        self.modality = modality
        self.dataset = dataset
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.full_video = full_video
        if self.test_mode:
            self.test_frames = 250
        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()  # get video list


    #据据idx 返回2张图, v和u
    def _load_image(self, directory, idx):
        directory = self.root_path + directory
        if self.modality == 'rgb' or self.modality == 'RGBDiff' or self.modality == 'RGB':
            #print(os.path.join(directory, self.image_tmpl.format(idx)))
            new_directory = directory[9:]
            #print(new_directory)
            img = Image.open(os.path.join(new_directory, self.image_tmpl.format(idx))).convert('RGB')
            if self.dataset == 'something_something_v1':
                width, height = img.size
                #print(width, height)
                img = img.resize((int(256/height*width),256), Image.BILINEAR)
                return [img]
            else:
                return [img]
            #return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'flow':
            #here may be a bug, we don;t load u and v ?
            #there is the reason why ucf101 optical flow is low
            #ucf101_flow_val_split list is wrong
            if self.dataset == 'ucf101':
                u_img_path = directory + '/frame'+ str(idx).zfill(6) + '.jpg'
                v_img_path = directory + '/frame'+ str(idx).zfill(6) + '.jpg'
                x_img = Image.open(u_img_path).convert('L')
                y_img = Image.open(v_img_path).convert('L')
                #x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
                #y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            else:
                x_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
                y_img = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')
            return [x_img, y_img]


    def preprocess(img):
        """change RGB [0,1] valued image to BGR [0,255]"""
        out = np.copy(img) * 255
        out = out[:, :, [2, 1, 0]]
        return out

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        # video list?

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        # new_length = 1 ?
        # num_frames : frame num
        #
        index = random.randint(1, max(record.num_frames - self.new_length * self.stride, 0) + 1)
        return index  # ? return array,because rangint is 0 -> num-1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length * self.stride + 1) / 1
            offsets = np.array(
                random.sample(range(0, record.num_frames - self.new_length * self.stride + 1), self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length * self.stride - 1:
            offsets = np.sort(
                random.sample(range(0, record.num_frames - self.new_length * self.stride + 1), self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]  # video name?
        if not self.test_mode:
            segment_indices = self._sample_indices(record)
            data, label = self.get(record, segment_indices, data_augment=True)
            data = 2 * (data / 255) - 1
            data = self.transform(data)
            # print(len(data))
            if type(data) == list and len(data) > 1:
                new_data = list()
                for one_sample in data:
                    new_data.append(video_to_tensor(one_sample))
            else:
                new_data = video_to_tensor(data)
            # print(new_data.size())
        else:
            segment_indices = self._get_test_indices(record)
            data, label = self.get(record, segment_indices, data_augment=False)
            data = 2 * (data / 255) - 1
            data = self.transform(data)
            if type(data) == list and len(data) > 1:
                new_data = list()
                for one_sample in data:
                    new_data.append(video_to_tensor(one_sample))
            else:
                new_data = video_to_tensor(data)
        return new_data, label

    def frames_padding(self, imgs, data_length):
        """
        if video's len is short than 64, do padding
        first method: just padding the same
        second method: according to optical flow, estimate the imgs
        three mthod: use numpy interpolation may be error?
        :param imgs:
        :return:
        """
        t, h, w, c = imgs.shape
        pdding_num = data_length - len(imgs)
        padding_imgs = np.resize(imgs, (data_length, h, w, c))
        return padding_imgs

    def get(self, record, indices, data_augment=False, side_length=224, is_numpy=True):
        images = list()
        p = int(indices)
        if not self.full_video:
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames - self.stride + 1:
                    p += self.stride
                else:
                    p = 1
        else:
            if record.num_frames < self.new_length:
                for i in range(record.num_frames):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.stride + 1:
                        p += self.stride
                    else:
                        p = 1
            else:
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames - self.stride + 1:
                        p += self.stride
                    else:
                        p = 1
        # images = transform_data(images, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        if is_numpy:
            frames_up = []
            if self.modality == 'rgb':
                for i, img in enumerate(images):
                    frames_up.append(np.asarray(img))
            elif self.modality == 'flow':
                for i in range(0, len(images), 2):
                    # it is used to combine frame into 2 channels
                    tmp = np.stack([np.asarray(images[i]), np.asarray(images[i + 1])], axis=2)
                    frames_up.append(tmp)
            images = np.stack(frames_up)

            if self.full_video:
                if record.num_frames < self.new_length:
                    images = self.frames_padding(images, self.new_length)
        return images, record.label

    def get_test(self, record, indices):
        '''
        get num_segments data
        '''
        # print(indices)
        all_images = []
        count = 0
        for seg_ind in indices:
            images = []
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                # print(seg_imgs)
                images.append(seg_imgs)
                if p < record.num_frames - self.stride + 1:
                    p += self.stride
                else:
                    p = 1
            all_images.append(images)
            count = count + 1
        process_data = np.asarray(all_images, dtype=np.float32)
        # print(process_data.shape)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
