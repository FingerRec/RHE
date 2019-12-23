#!/usr/bin/env bash
python train_mixup_i3d.py --train_list data/hmdb51/hmdb51_train1.txt \
--val_list data/hmdb51/hmdb51_rgb_val_split_1.txt \
--dataset hmdb51 --arch mpi3d_pt \
--mode rgb \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 10 20 25 30 35 40 --epochs 45 --batch-size 6 \
--snapshot_pref checkpoints/ \
--dropout 0.5 --mixup 1 --gpus 2,3 --print-freq 100
