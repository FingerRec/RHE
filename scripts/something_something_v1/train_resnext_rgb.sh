#!/usr/bin/env bash
python train_resnext.py --train_list data/something_something_v1/train_videofolder.txt \
--val_list data/something_something_v1/val_videofolder.txt \
--dataset something_something_v1 \
--mode rgb \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 10 20 25 30 35 40 --epochs 45 --batch-size 6 \
--snapshot_pref checkpoints/ \
--dropout 0.5 --gpus 0,1 --print-freq 100 --root /home/wjp/Data/20bn-something-something-v1/ \
