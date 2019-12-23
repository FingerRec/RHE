# -*- coding: utf-8 -*-
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import cv2
import torch
#from config import opt
def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def action_class(self):
        return self._data[0]
    @property
    def path(self):
        return self._data[1]

    @property
    def num_frames(self):
        return int(self._data[2])

    @property
    def label(self):
        return int(self._data[3])


class I3dDataSet(data.Dataset):
    def __init__(self, root_path, list_file, dataset = 'ucf101',
                 num_segments=1, new_length=64, modality='rgb',
                 image_tmpl='img_{:06d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.dataset = dataset
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list() #get video list

    #据据idx 返回2张图, v和u
    def _load_image(self, directory, idx):
        if self.modality == 'rgb' or self.modality == 'RGBDiff':
            #print(self.image_tmpl)
            #print(os.path.join(directory, self.image_tmpl.format(idx)))
            img = cv2.imread(os.path.join(directory, self.image_tmpl.format(idx)))[:, :, [2, 1, 0]]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            return img
            #return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'flow':
            if self.dataset == 'ucf101':
                img_path = '/frame'+ str(idx).zfill(6) + '.jpg'
                x_img = Image.open((opt.temporal_train_data_root + 'u/' + directory.split('.')[0].split('/')[-1] + img_path)).convert('L')
                y_img = Image.open((opt.temporal_train_data_root + 'v/' + directory.split('.')[0].split('/')[-1] + img_path)).convert('L')
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
        #video list?
    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        #new_length = 1 ?
        #num_frames : frame num
        #
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            #randint: generate num_segments num in range average_duration
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            if record.num_frames > self.new_length:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1 # ? return array,because rangint is 0 -> num-1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index] # video name?

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        data, label = self.get(record, segment_indices)
        data = self.transform(data)
        #print(label)
        return video_to_tensor(data), label
    #record: {path, num_frames, label}
    #so the txt should be path,num_frames.label
    def get(self, record, indices):
        #print(record.label)
        images = []
        #labels = np.zeros((400,64), np.float32)
        for seg_ind in indices:
            p = int(seg_ind)
            #read img
            #for i in range(1) here
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.append(seg_imgs)
                if p < record.num_frames:
                    p += 1
                else:
                    p = 1
        #labels[record.label, :] = 1
        #process_data = self.transform(images)
        process_data =  np.asarray(images, dtype=np.float32)
       # process_data = process_data.view(9,10,299,299)
       # print("process_data's size is: ")
       # print(process_data.size()) #90L x 299L x 299L 3(segments)x3(channel)x10
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
