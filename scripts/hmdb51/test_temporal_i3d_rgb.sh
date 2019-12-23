#!/usr/bin/env bash
python test_temporal_i3d.py --dataset hmdb51 \
--mode rgb --test_list data/hmdb51/hmdb51_rgb_val_split_1.txt \
--weights checkpoints/hmdb51/43.856_temporal_i3d_rgb_model_best.pth.tar \
--dropout 0 --test_clips 1 \
--save_scores temporal_i3d_rgb_hmdb51_save_scores \
--batch_size 1 --gpus 1