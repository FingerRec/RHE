#!/usr/bin/env bash
python train_i3d.py --train_list data/hmdb51_flow_train_split_1.txt \
--val_list data/hmdb51_flow_val_split_1.txt \
--dataset hmdb51 \
--mode flow \
--save_model checkpoints/ \
--lr 0.001 --lr_steps 10 20 30 --epochs 40 --batch-size 4 \
--snapshot_pref checkpoints/hmdb51_rgb_i3d_cluster_ \
--weight_decay 3e-7 --dropout 0.5 --gpus 1
