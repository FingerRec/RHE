#!/usr/bin/env bash
python train_i3d.py --train_list data/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51_rgb_val_split_1.txt \
--cluster_list data/hmdb51_rgb_train_cluster_split_1.txt \
--cluster_train 0 \
--dataset hmdb51 \
--mode rgb --arch fli3d \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 10 20 30 --epochs 40 --batch-size 2 \
--snapshot_pref checkpoints/hmdb51_rgb_i3d_cluster_ \
--weight_decay 7e-7 --dropout 0.7 --gpus 0 --stride 1 \
--weights checkpoints/hmdb51/74.379_i3d_rgb_model_best.pth.tar
