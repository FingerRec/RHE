#!/usr/bin/env bash
python train_gcn_i3d.py --train_list data/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--mode rgb \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 15 25 35 40 --epochs 45 --batch-size 5 \
--snapshot_pref checkpoints/ \
--dropout 0.5 --gpus 3