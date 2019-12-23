#!/usr/bin/env bash
python train_charades_i3d.py -mode rgb \
-root /home/wjp/Desktop/disk2_6T/DataSet/charades/Charades_v1_rgb/ \
-save_model checkpoints/charades/ \
-lr 0.1 -b 5 --gpus 0