#!/usr/bin/env bash
python test_mp_i3d.py --dataset hmdb51 \
--mode flow --test_list data/hmdb51_flow_val_split_1.txt \
--weights checkpoints/hmdb51/77.647_mpi3d_flow_model_best.pth.tar \
--dropout 0 --test_clips 50 \
--save_scores mpi3d_flow_hmdb51_save_scores \
--batch_size 1