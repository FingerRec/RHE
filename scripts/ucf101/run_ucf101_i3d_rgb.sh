#!/usr/bin/env bash
python train_i3d_full.py --train_list data/ucf101_rgb_train_split_1.txt \
--val_list data/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--mode rgb \
--save_model checkpoints/i3d_ucf101_rgb \
--lr 0.1 --lr_steps 10 20 30 --epochs 40 --batch-size 10 \
--snapshot_pref checkpoints/ucf101_rgb_i3d_ \
--weight-decay 1e-8 --dropout 0.36 \
--mixup False --gpus 0 1 2 3 --moment 0.9