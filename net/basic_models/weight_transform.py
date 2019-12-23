import torch
from torch import nn

def weight_transform(model_dict, pretrain_dict):
    weight_dict = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(weight_dict)
    return model_dict