# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import cv2
import torch
from .on_the_fly_decode import _load_action_frame_nums_to_4darray
from videotransforms import video_frames_resize
import dataset.utils as utils

# from config import opt


def images_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class VideostreamError(BaseException):
    pass


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class VideoDataSet(data.Dataset):
    """
    3d based dataset
    random select one video and decode on the fly,
    return an array of decoded images
    the input txt file should be in format
    video_path label
    """
    def __init__(self, root, list_file, data_set='ucf101',
                 new_length=64, modality='rgb',
                 transform=None,
                 random_shift=True, test_mode=False):

        self.root = root
        self.list_file = list_file
        self.new_length = new_length
        self.modality = modality
        self.dataset = data_set
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self._parse_list()  # get video list


    def pre_process(img):
        """change RGB [0,1] valued image to BGR [0,255]"""
        out = np.copy(img) * 255
        out = out[:, :, [2, 1, 0]]
        return out

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]
        print(len(self.video_list))

    def __getitem__(self, index):
        '''
        i Comment all these warning message for simplify, remove this # if need.
        this part code can be simplity, but i think this way is the most straght mehod
        :param index:
        :return:
        '''
        record = self.video_list[index]
        try:
            decoded_images, label = self.get(record, random_select = self.test_mode)
        except IOError:
            # print("Error: there is no video in this place, will random select another video")
            index = randint(1, len(self.video_list))
            record = self.video_list[index]
            decoded_images, label = self.get(record, random_select=self.test_mode)
        except ValueError:
            # print("Error: wrong video stream, ffmpeg can't decode, will random select another video")
            index = randint(1, len(self.video_list))
            record = self.video_list[index]
            decoded_images, label = self.get(record, random_select=self.test_mode)
        except RuntimeError:
            # print("Error: find video, but frame num is 0 or video can't open, will random select another video")
            index = randint(1, len(self.video_list))
            record = self.video_list[index]
            decoded_images, label = self.get(record, random_select=self.test_mode)
        except TypeError as te:
            print(te)
            # print("stream can't be select properly or decoded images is too short, change it again")
            index = randint(1, len(self.video_list))
            record = self.video_list[index]
            decoded_images, label = self.get(record, random_select=self.test_mode)
        except FileNotFoundError:
            index = randint(1, len(self.video_list))
            record = self.video_list[index]
            decoded_images, label = self.get(record, random_select=self.test_mode)
        decoded_images = video_frames_resize(decoded_images, 64)
        decoded_images = self.transform(decoded_images)
        return images_to_tensor(decoded_images), label

    def get(self, record, random_select = False):
        """
        just for one segment
        :param record:
        :param random_select: train:true test:false
        :return:
        """
        # read video in this place, if no video here, random select another video
        # print(self.root + record.path)
        f = open(self.root + record.path, 'rb')
        video = f.read()
        f.close()
        video_frames_num, width, height = utils.video_frame_count(self.root + record.path)
        # print(video_frames_num)
        if video_frames_num == -1:
            raise RuntimeError("No video stream avilable")
        # if video_frames_num < self.new_length:
        #    print("video{} farmes num is:{}".format(self.root + record.path, video_frames_num))
        # opencv ususlly decode more frames, so - 10 here instead of +1
        rand_index = randint(0, max(1, video_frames_num - self.new_length - 1))
        if video_frames_num > self.new_length:
            if random_select:
                decoded_images_indexs = np.arange(rand_index, self.new_length + rand_index)
            else:
                decoded_images_indexs = np.arange(min(rand_index, 10), min(self.new_length + min(rand_index,10), video_frames_num))
        else:
            decoded_images_indexs = np.arange(0, video_frames_num-1)
        # the video may be 224 x 144, need to do resize
        # if decoded_images_index is small than new_length, loop until new_length
        decoded_images = _load_action_frame_nums_to_4darray(video, decoded_images_indexs, width, height)
        if np.shape(decoded_images)[0] < self.new_length:
            for i in range(self.new_length - np.shape(decoded_images)[0]):
                decoded_images = np.concatenate((decoded_images, np.reshape(decoded_images[i%np.shape(decoded_images)[0], :, :, :], newshape=(1, height, width, 3))), axis=0)
        if np.shape(decoded_images)[0] != self.new_length:
            raise TypeError("imgs is short than need.!")
        process_data = np.asarray(decoded_images, dtype=np.float32)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
