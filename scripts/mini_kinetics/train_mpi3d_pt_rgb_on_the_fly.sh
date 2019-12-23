#!/usr/bin/env bash
python train_i3d_on_the_fly.py \
--root /home/manager/disk1_6T/Share_Folder/wjp \
--train_list data/mini_kinetics/mini_kinetics_video_trainlist.txt \
--val_list data/mini_kinetics/mini_kinetics_video_vallist.txt \
--data_set mini_kinetics \
--mode rgb --arch mpi3d_pt \
--save_model checkpoints/mini_kinetics/kinetics_mpi3d_pt_rgb \
--mixup False --dropout 0.5 \
--batch-size 10 --lr 0.001 --weight_decay 3e-7 \
--print_freq 500 --prefix checkpoints/mini_kinetics/ --gpus 0,1 # \
# --resume checkpoints/mini_kinetics/mini_kinetics_rgb_checkpoint.pth.tar