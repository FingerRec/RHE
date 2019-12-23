#!/usr/bin/env bash
python train_i3d.py --train_list data/something_something_v1/train_videofolder.txt \
--val_list data/something_something_v1/val_videofolder.txt \
--dataset something_something_v1 \
--arch mpi3d_pt \
--mode rgb \
--save_model checkpoints/ \
--dropout 0.5 \
--batch-size 10 --lr 0.001 --weight_decay 1e-8 \
--root /home/wjp/Data/20bn-something-something-v1/ \
--gpus 2,3 --epochs 45 --lr_steps 15 25 35 40 \
--print-freq 500