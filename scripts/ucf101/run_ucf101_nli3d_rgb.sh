#!/usr/bin/env bash
python train_nl_i3d.py --train_list data/ucf101_rgb_train_split_1.txt \
--val_list data/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--mode rgb \
--save_model checkpoints/nli3d_ucf101_rgb \
--mixup 0 --dropout 0.64 \
--batch-size 4 --lr 0.001 --weight_decay 7e-7 \
--epochs 45