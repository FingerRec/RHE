import cv2
import numpy as np
import os
import shutil
import numpy as np

def main():
    img_dir = '/home/wjp/Desktop/disk2_6T/DataSet/interplate_5_hmdb51_imgs'
    back_dir = '/home/wjp/Data/hmdb51'
    count = 0
    video_list = [(x.strip().split(' ')) for x in open('../../data/hmdb51/hmdb51_rgb_train_split_1.txt')]
    for subdir in os.listdir(img_dir):
        if len(os.listdir(os.path.join(img_dir, subdir))) < 5 * (len(os.listdir(os.path.join(back_dir, subdir)))/3-5):
            # print(len(os.listdir(os.path.join(img_dir, subdir))), len(os.listdir(os.path.join(back_dir, subdir))))
            # os.removedirs(os.path.join(img_dir, subdir))
            print(subdir, len(os.listdir(os.path.join(img_dir, subdir)))//5)
            count += 1
            for i in range(len(video_list)):
                if video_list[i][0] == os.path.join(back_dir, subdir):
                    print(video_list[i])
                    video_list[i][1] = len(os.listdir(os.path.join(img_dir, subdir)))//5
            # shutil.copytree(os.path.join(back_dir, subdir), os.path.join(img_dir, subdir))
            # print(len(os.listdir(os.path.join(img_dir, subdir))))

    print(count)
    rgb_list = list()
    for i in range(len(video_list)):
        rgb_list.append('{} {} {}\n'.format(video_list[i][0], video_list[i][1], video_list[i][2]))
    #
    # with open('../../data/hmdb51/hmdb51_train1.txt', 'w') as f:
    #     f.writelines(rgb_list)


if __name__ == '__main__':
    main()
