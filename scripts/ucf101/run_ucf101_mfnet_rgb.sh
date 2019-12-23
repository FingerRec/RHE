#!/usr/bin/env bash
python train_mfnet.py --train_list data/ucf101_rgb_train_split_1.txt \
--val_list data/ucf101_rgb_val_split_1.txt \
--dataset ucf101 \
--mode rgb \
--save_model checkpoints/manet_ucf101_rgb \
--mixup 0 --dropout 0.64 \
--batch-size 30 --lr 0.001 --weight_decay 7e-7 \
--epochs 45