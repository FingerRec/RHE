import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

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


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f) # 9848

    i = 0
    all_num = 0
    old_split = split
    if split == 'val_video':
        split = 'testing'
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue
        else:
            all_num += 1
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue
        if not os.path.exists(os.path.join(root, vid)):
            print("no video in: ", os.path.join(root, vid))
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2
            
        if num_frames < 66:
            continue

        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1
        if i%100 == 0:
            print("load {} data {}/{}".format(old_split, i, all_num))
    
    return dataset



class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.split = split

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, dur, nf = self.data[index]
        start_f = random.randint(1,nf-65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)

        imgs = self.transforms(imgs)
        if self.split == 'val_video':
            target = torch.IntTensor(157).zero_()
            for c in range(label.shape[0]):
                for n in range(label.shape[1]):
                    if label[c,n] == 1:
                        target[c] = 1
            return vid, video_to_tensor(imgs), target
            #print(target)
            #target[label.sum(axis=1) > 1] = 1
        else:
            label = label[:, start_f:start_f+64]
            return vid, video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

def make_dataset_val(split_file, split, root, mode, snippets, num_classes=157):
    count_items = 0
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue

        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == "flow":
            num_frames = num_frames // 2

        fps = num_frames / data[vid]['duration']

        for j in range(0, num_frames):
            #if j + snippets > num_frames:
            #    continue
            label = np.zeros((num_classes, num_frames), np.float32)
            for ann in data[vid]['actions']:
                for fr in range(j + 1, j + s + 1, 1):
                    if fr / fps >= ann[1] and fr / fps <= ann[2]:
                        label[ann[0], (fr - 1) % snippets] = 1

            dataset.append((vid, j + 1, label))
            count_items += 1

    print("Make dataset {}: {} examples".format(split, count_items * snippets))

    return dataset

class Charades_eval(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode,  transforms=None):

        self.data = make_dataset_val(split_file, split, root, mode, snippets=64)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.snippets = 64

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, start, label = self.data[index]
        # start_f = random.randint(1,nf-65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start, self.snippets)
        else:
            imgs = load_flow_frames(self.root, vid, start, self.snippets)
        # label = label[:, :] #start_f:start_f+64]

        imgs = self.transforms(imgs)

        return vid, start, video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)
