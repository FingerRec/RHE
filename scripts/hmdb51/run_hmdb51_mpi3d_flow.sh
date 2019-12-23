#!/usr/bin/env bash
python train_i3d.py --train_list data/hmdb51_flow_train_split_1.txt \
--val_list data/hmdb51_flow_val_split_1.txt \
--dataset hmdb51 \
--mode flow --arch mpi3d  \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 15 25 35 --epochs 45 --batch-size 5 \
--snapshot_pref checkpoints/hmdb51_flow_mpi3d_cluster_ \
--weight_decay 7e-7 --dropout 0.5 --gpus 2 --workers 10
