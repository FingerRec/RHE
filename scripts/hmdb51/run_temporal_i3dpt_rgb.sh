#!/usr/bin/env bash
python train_temporal_i3d.py --train_list data/hmdb51/hmdb51_rgb_train_split_1.txt \
--val_list data/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 \
--mode rgb \
--save_model checkpoints/ \
--lr 0.1 --lr_steps 10 20 25 30 35 40 --epochs 45 --batch-size 32 \
--snapshot_pref checkpoints/ \
--dropout 0.5 --gpus 0,1 --weights checkpoints/kinetics28.682_rgb_model_best.pth.tar