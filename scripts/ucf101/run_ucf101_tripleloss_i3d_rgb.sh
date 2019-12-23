#!/usr/bin/env bash
python i3d_triplet_loss.py --train_list data/ucf101_rgb_train_split_1.txt \
--val_list data/ucf101_rgb_val_split_1.txt \
--similar_list data/ucf101_triplet_similar_rgb.txt \
--dataset ucf101 \
--mode rgb \
--save_model checkpoints/triplet_i3d_ucf101_rgb \
--mixup 0 --dropout 0.5 \
--batch_size 1 --lr 0.001 --weight_decay 7e-7 \
--epochs 45