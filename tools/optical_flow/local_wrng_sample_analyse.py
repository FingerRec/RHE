import cv2
import numpy as np
import os
import shutil
import numpy as np


def main():
    img_dir = '/media/awiny/9224670c-39f0-4964-8a14-7478ec574a6b/dataset/interplate_5_hmdb51_imgs'
    count = 0
    video_list = [(x.strip().split(' ')) for x in open('../../data/hmdb51/hmdb51_rgb_train_split_1.txt')]
    for subdir in os.listdir(img_dir):
        if len(os.listdir(os.path.join(img_dir, subdir))) < 1:
            print(os.path.join(img_dir, subdir)) # , len(os.listdir(os.path.join(back_dir, subdir))))
            # os.removedirs(os.path.join(img_dir, subdir))

            # shutil.copytree(os.path.join(back_dir, subdir), os.path.join(img_dir, subdir))
            # print(len(os.listdir(os.path.join(img_dir, subdir))))

    # print(count)
    # rgb_list = list()
    # for i in range(len(video_list)):
    #     rgb_list.append('{} {} {}\n'.format(video_list[i][0], video_list[i][1], video_list[i][2]))


if __name__ == '__main__':
    main()
