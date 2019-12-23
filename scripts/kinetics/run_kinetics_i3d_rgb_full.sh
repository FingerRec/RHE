#!/usr/bin/env bash
python train_i3d_full.py --train_list data/kinetics_rgb_train_list.txt \
--val_list data/kinetics_rgb_val_list.txt \
--dataset kinetics \
--mode rgb \
--save_model checkpoints/i3d_kinetics_rgb \
--lr 0.1 --lr_steps 40 80 --epochs 100 --batch-size 16 \
--snapshot_pref checkpoints/kinetics_rgb_i3d_