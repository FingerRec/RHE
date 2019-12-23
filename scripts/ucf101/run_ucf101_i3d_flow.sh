#!/usr/bin/env bash
python train_i3d.py --train_list data/ucf101_flow_train_split_1.txt \
--val_list data/ucf101_flow_val_split_1.txt \
--dataset ucf101 \
--mode flow \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 10 20 30 --epochs 40 --batch-size 4 \
--snapshot_pref checkpoints/ucf101_rgb_i3d \
--weight_decay 5e-7 --dropout 0.6 --gpus 3
