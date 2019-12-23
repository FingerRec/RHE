#!/usr/bin/env bash
python train_r3d.py --train_list data/something_something_v1/train_videofolder.txt \
--val_list data/something_something_v1/val_videofolder.txt \
--dataset something_something_v1 \
--arch resnet18 \
--mode rgb \
--save_model checkpoints/ \
--dropout 0.63 \
--batch-size 64 --lr 0.01 --weight_decay 1e-3 \
--root /home/wjp/Data/20bn-something-something-v1/ \
--gpus 2,3 --stride 2