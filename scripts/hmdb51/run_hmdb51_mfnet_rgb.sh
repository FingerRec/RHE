#!/usr/bin/env bash
python train_mfnet.py --train_list data/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51_rgb_val_split_1.txt \
--cluster_list data/hmdb51_rgb_train_cluster_split_1.txt \
--cluster_train 0 \
--dataset hmdb51 \
--mode rgb \
--save_model checkpoints/mfnet_hmdb51_rgb \
--lr 0.005 --lr_steps 10 20 30 --epochs 40 --batch-size 10 \
--snapshot_pref checkpoints/hmdb51_rgb_i3d_cluster_ \
--weight_decay 1e-7 --dropout 0.5 --gpus 1