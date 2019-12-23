#!/usr/bin/env bash
python train_i3d.py --train_list data/kinetics_rgb_train_list.txt \
--val_list data/kinetics_rgb_val_list.txt \
--dataset kinetics \
--mode rgb \
--save_model checkpoints/kinetics_i3d_rgb \
--dropout 0.63 \
--batch-size 4 --lr 0.01 --weight_decay 7e-7