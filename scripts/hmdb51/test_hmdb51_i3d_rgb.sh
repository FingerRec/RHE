#!/usr/bin/env bash
python test_i3d.py --dataset hmdb51 \
--mode rgb --test_list data/hmdb51_rgb_val_split_1.txt \
--weights checkpoints/73.4_i3d_hmdb51_rgbmodel_best.pth.tar \
--dropout 0 --test_clips 20 \
--save_scores test_output/rgb_hmdb51_save_scores \
--batch_size 1