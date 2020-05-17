import torch
import torch.nn as nn

from .util import initialize_weights, Sequential
from .fusion_model import Up
from utils import mymath


class ReconstructionForwardUnit(nn.Module):
    def __init__(self, bn):
        super(ReconstructionForwardUnit, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv5 = Sequential(
            nn.Conv2d(256, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv6 = Sequential(
            nn.Conv2d(128, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv7 = Sequential(
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv8 = nn.Conv2d(32, 1, 3, padding=1)
        self.ac8 = nn.LeakyReLU(inplace=True)


    def forward(self, *input):
        x, u_x = input
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        output = self.ac8(x8 + u_x)
        return output
        

#   The stage reconstruction
class ReconstructionUnetUnit(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(ReconstructionUnit, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1), nn.LeakyReLU(inplace=True))

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(inplace=True))

        self.down4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.LeakyReLU(inplace=True))

        self.up1 = Up(64, 32)
        self.ac1 = nn.LeakyReLU(inplace=True)
        self.up2 = Up(64, 32)
        self.ac2 = nn.LeakyReLU(inplace=True)
        self.up3 = Up(64, 32)
        self.ac3 = nn.LeakyReLU(inplace=True)
        self.up4 = Up(42, 2)
        self.ac4 = nn.LeakyReLU(inplace=True)

        initialize_weights(self)

    def forward(self, *input):
        x = input[0]
        raw = input[1]
        #Encoder
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(*(x3, x4))
        x5 = self.ac1(x5)
        x6 = self.up2(*(x2, x5))
        x6 = self.ac2(x6)
        x7 = self.up3(*(x1, x6))
        x7 = self.ac3(x7)
        x8 = self.up4(*(x, x7))
        x8 = self.ac4(x8+raw)
        return x8


class ReconstructionUnitLoss(nn.Module):
    def __init__(self):
        super(ReconstructionUnitLoss, self).__init__()
        self.img_mse = nn.MSELoss()
        self.k_mse = nn.MSELoss()
    
    def forward(self, *input):
        y = input[0]
        target = input[1]
        y_k = mymath.torch_fft2c(y)
        target_k = mymath.torch_fft2c(target)

        loss1 = self.img_mse(y, target)
        loss2 = self.k_mse(y_k, target_k)

        return loss1 + loss2
