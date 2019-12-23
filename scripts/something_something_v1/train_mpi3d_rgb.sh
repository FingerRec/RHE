#!/usr/bin/env bash
python train_i3d.py --train_list data/something_something_v1/train_videofolder.txt \
--val_list data/something_something_v1/val_videofolder.txt \
--dataset something_something_v1 \
--arch mpi3d \
--mode rgb \
--save_model checkpoints/ \
--dropout 0.63 \
--batch-size 8 --lr 0.005 --weight_decay 1e-4 \
--root /home/wjp/Data/20bn-something-something-v1/ \
--gpus 0,1 --epochs 50 --lr_steps 20 30 35 40 45