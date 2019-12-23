#!/usr/bin/env bash
python train_i3d.py --train_list data/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51/hmdb51_rgb_val_split_1.txt \
--cluster_list data/hmdb51/hmdb51_rgb_train_cluster_split_1.txt \
--cluster_train 0 \
--dataset hmdb51 \
--mode rgb --arch i3d \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 15 25 35 --epochs 45 --batch-size 5 \
--snapshot_pref checkpoints/hmdb51/hmdb51_rgb_i3d_cluster_ \
--weight_decay 3e-7 --dropout 0.5 --gpus 1 --stride 1