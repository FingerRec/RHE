#!/usr/bin/env bash
python train_i3d.py --train_list data/something_something_v1/train_videofolder.txt \
--val_list data/something_something_v1/val_videofolder.txt \
--dataset something_something_v1 \
--arch i3dpt \
--mode rgb \
--save_model checkpoints/ \
--dropout 0.63 \
--batch-size 64 --lr 0.1 --weight_decay 7e-7 \
--root /home/wjp/Data/20bn-something-something-v1/ \
--gpus 0,1,2,3 --stride 3 --mixup 1 --workers 8