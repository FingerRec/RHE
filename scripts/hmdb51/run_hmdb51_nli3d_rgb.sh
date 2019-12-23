#!/usr/bin/env bash
python train_nl_i3d.py --train_list data/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51_rgb_val_split_1.txt \
--cluster_list data/hmdb51_rgb_train_cluster_split_1.txt \
--cluster_train 0 \
--dataset hmdb51 \
--mode rgb \
--save_model checkpoints/i3d_hmdb51_rgb \
--lr 0.001 --lr_steps 10 20 25 30 35 40 --epochs 45 --batch-size 8 \
--snapshot_pref checkpoints/hmdb51_rgb_nli3d_ \
--weight_decay 4e-7 --dropout 0.4
