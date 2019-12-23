#!/usr/bin/env bash
python test_video_something_v1.py --dataset something_something_v1 \
--mode rgb --test_list data/something_something_v1/val_videofolder.txt \
--weights checkpoints/something_something_v1/44.323_mpi3d_pt_rgb_model_best.pth.tar \
--dropout 0 --test_clips 10 --arch mpi3d_pt \
--save_scores mpi3d_pt_rgb_sthv1_video_save_scores \
--root /home/wjp/Data/20bn-something-something-v1/ \
--batch_size 1 --gpus 2