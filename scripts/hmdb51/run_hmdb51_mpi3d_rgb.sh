#!/usr/bin/env bash
python train_i3d.py --train_list data/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51/hmdb51_rgb_val_split_1.txt \
--cluster_list data/hmdb51/hmdb51_rgb_train_cluster_split_1.txt \
--cluster_train 0 \
--dataset hmdb51 \
--mode rgb --arch mpi3d \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 15 25 35 --epochs 45 --batch-size 4 \
--snapshot_pref checkpoints/hmdb51_rgb_mpi3d_cluster_ \
--weight_decay 7e-7 --dropout 0.5 --gpus 3 --stride 1 \
--workers 10
