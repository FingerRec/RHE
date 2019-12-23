import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

class SLSloss(nn.Module):
    def __init__(self):
        super(SLSloss, self).__init__()


    def forward(self, input, target, flg):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        maxRow, _ = torch.max(input.data, 1)
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow
        flg = flg.view(-1, 1)
        flos = F.log_softmax(input)
        flos = torch.sum(flos, 1) / flos.size(1)
        logpt = F.log_softmax(input)
        logpt = torch.mul(logpt, target)
        logpt = torch.sum(logpt, 1, True)
        logpt = logpt.view(-1)
        flg = flg.view(-1)
        flg = flg.type(torch.cuda.FloatTensor)
        loss = -1 * logpt * (1 - flg) - flos * flg
        return loss.mean()