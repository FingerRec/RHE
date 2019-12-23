#!/usr/bin/env bash
python train_temporal_i3d_on_the_fly.py \
--root /home/wjp/Data/compress/ \
--train_list data/kinetics/ssd_kinetics_video_trainlist.txt \
--val_list data/kinetics/ssd_kinetics_video_vallist.txt \
--data_set kinetics \
--mode rgb \
--save_model checkpoints/kinetics/temporal_i3d_light \
--mixup False --dropout 0.5 \
--batch-size 80 --lr 0.1 --weight_decay 3e-7 \
--print_freq 100 --gpus 0,1