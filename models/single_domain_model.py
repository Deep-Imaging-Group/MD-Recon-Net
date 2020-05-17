import torch
import torch.nn as nn

from .common_model import SingleFoldDomain

class SingleDomainReconstruction(nn.Module):
    def __init__(self):
        super(SingleDomainReconstruction, self).__init__()
        self.fold1 = SingleFoldDomain()
        self.fold2 = SingleFoldDomain()
        self.fold3 = SingleFoldDomain()

    def forward(self, x):
        y1 = self.fold1(x)
        x1 = y1 + x
        y2 = self.fold2(x1)
        x2 = y2 + x
        y3 = self.fold3(x2)

        return y3
    