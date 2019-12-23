#!/usr/bin/env bash
python test_i3d.py --dataset ucf101 \
--mode rgb --test_list data/ucf101_rgb_val_split_1.txt \
--weights i3d_rgb_model_best.pth.tar \
--dropout 0 --test_segments 10 \
--save_scores test_output/rgb_save_scores \
--batch_size 1