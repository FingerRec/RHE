# MTP-3DNet for Action Recognition
This code is a re-implementation of the video 
classification experiments in our Revisiting Hard-example. The code is developed based on the PyTorch framework.


## Installation
- Pytorch0.4+
- PilImage
- Lintel(optional, for on-the-fly decode)


## Dataset Preparation
As shown in `data/*/*.txt`, the video should in format [path, frame_num, class]
```bash
/home/wjp/Data/hmdb51/AmericanGangster_drink_u_nm_np1_fr_med_39 77 10
```

## Main Result

DataSet |Modality| accuarcy
---- | ---|---
HMDB51 | RGB + Flow|83.0%
UCF101 |  RGB + Flow|98.4%
Something-something-v1 |RGB| 48.4%
Kinetics-400|RGB|76.7%

Use offered pretrained-model, all these result can be reproduced in 1 1080Ti GPU.

## Train
### HMDB51
#### RGB
```bash
bash scripts/hmdb51/run_hmdb51_mpi3d_rgb.sh
```
#### Flow
```bash
bash scripts/hmdb51/run_hmdb51_mpi3d_flow.sh
```
### UCF101
#### RGB
```bash
bash scripts/ucf101/run_ucf101_i3d_rgb.sh
```
#### Flow
```bash
bash scripts/ucf101/run_ucf101_i3d_flow.sh
```
### Kinetics
#### RGB
```bash
bash scripts/kinetics/run_kinetics_i3d_rgb_on_the_fly.sh
```
### Something-something-v1
#### RGB
```bash
bash scripts/something_something_v1/train_i3d_rgb.sh
```
#### Flow
```bash
bash scripts/something_something_v1/train_mpi3d_pt_flow.sh
```
### Charadess
#### RGB
```bash
bash scripts/charades/run_charades_i3d_rgb.sh
```

After train, the model will be saved in checkpoints/[*]/[*].pth.tar
## Test
### HMDB51
```bash
bash scripts/hmdb51/test_hmdb51_mpi3d_rgb[flow].sh
```
### UCF101
```bash
bash scripts/ucf101/test_ucf101_i3d_rgb.sh
```

### Kinetics
```bash
bash scripts/kinetics/test_kinetics_mpi3d_rgb.sh
```
### Something-something-v1
```bash
bash scripts/something_something_v1/test_mpi3d_rgb_video.sh
```

The score will be save as .npy.
## Two Stream Fusion
```python
python eval_all_network.py RGB_SCORE_FILE FLOW_SCORE_FILE --score_weights 1 1.5

```

## Others
For TSM/Slow-Fast implement. Just combine SL and VTC with these models.

## Acknowledgement
This project is partly based on [TSN](https://github.com/yjxiong/tsn-pytorch) . Also thanks [pytorch-i3d](https://github.com/hassony2/kinetics_i3d_pytorch)
for offer pytorch-i3d version.


