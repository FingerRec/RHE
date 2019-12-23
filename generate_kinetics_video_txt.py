import numpy as np
list_file = 'data/kinetics_rgb_train_list.txt'
video_list = [(x.strip().split(' ')) for x in open(list_file)]
video_num = len(video_list)
retrain_rgb_list_tf = list()

retrain_file_tf = 'data/kinetics_video_trainlist.txt'

output_file = 'data/kinetics_video_frame_trainlist.txt'
count = 0
for i in range(video_num):
    if int(video_list[i][2]) in index:
        count += 1
        retrain_rgb_list_tf.append(
            '{} {} {} {}\n'.format(video_list[i][0].split('/')[-1], video_list[i][0], video_list[i][1],
                                   video_list[i][2]))
print("all val num is", count)

with open(output_file, 'w') as f:
    f.writelines(retrain_rgb_list_tf)
