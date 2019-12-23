#!/usr/bin/env bash
python train_i3d.py --train_list data/ucf101/ucf101_rgb_train_split_1.txt \
--val_list data/ucf101/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--mode rgb \
--save_model checkpoints/i3d_ucf101_rgb \
--mixup 0 --dropout 0.5 \
--batch-size 4 --lr 0.001 --weight_decay 7e-7 \
--epochs 45 --gpus 0