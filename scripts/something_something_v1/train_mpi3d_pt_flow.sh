#!/usr/bin/env bash
python train_i3d.py --train_list data/something_something_v1/flow_train_videofolder.txt \
--val_list data/something_something_v1/flow_val_videofolder.txt \
--dataset something_something_v1 \
--arch mpi3d_pt \
--mode rgb \
--save_model checkpoints/ \
--dropout 0.63 \
--batch-size 10 --lr 0.001 --weight_decay 4e-7 \
--root /home/wjp/Desktop/disk2_6T/DataSet/something-something/flow/ \
--gpus 0,1 --epochs 45 --lr_steps 15 25 35 40