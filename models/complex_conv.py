import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy import signal
import scipy
import time



class ComplexCnn2d(torch.nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexCnn2d, self).__init__()
        self.rel_cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.img_cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.in_channels = in_channels

    def forward(self,imgs):
        mi_ki = self.img_cnn(imgs[:, self.in_channels:, :, :])
        mi_kr = self.rel_cnn(imgs[:, self.in_channels:, :, :])
        mr_ki = self.img_cnn(imgs[:, :self.in_channels, :, :])
        mr_kr = self.rel_cnn(imgs[:, :self.in_channels, :, :])

        img = mr_ki + mi_kr
        rel = mr_kr - mi_ki

        return torch.cat((rel, img), 1)


# x = Variable(torch.randn((5, 2, 32, 32))).cuda()
# cnn = ComplexCnn2d(2, 32, 3, 1, 1).cuda()

# y = Variable(torch.randn((5, 32, 32, 32))).cuda()
# criterion = nn.MSELoss()
# optimer = optim.Adam(cnn.parameters())

# for i in range(100000000):
#     output = cnn(x)
#     loss = criterion(y, output)
#     optimer.zero_grad()
#     loss.backward()
#     optimer.step()
#     print(loss.item())
