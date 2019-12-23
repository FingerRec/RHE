#!/usr/bin/env bash
python test_mp_i3d.py --dataset hmdb51 \
--mode rgb --test_list data/hmdb51_rgb_val_split_1.txt \
--weights checkpoints/hmdb51/75.947_mpi3d_rgb_model_best.pth.tar \
--dropout 0 --test_clips 50 \
--save_scores mpi3d_rgb_hmdb51_save_scores \
--batch_size 1