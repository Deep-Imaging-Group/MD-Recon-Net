import torch
import torch.nn as nn
import numpy as np
from . import complex_conv

ComplexCnn2d = complex_conv.ComplexCnn2d


class KspaceExtractor(nn.Module):
    def __init__(self):
        super(KspaceExtractor, self).__init__()
        self.Ccnn1 = ComplexCnn2d()
