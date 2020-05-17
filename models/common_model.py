import torch
import torch.nn as nn

from .util import *

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.01):
        super(Up, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, *input):
        x1 = input[0]
        x2 = input[1]

        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)

#   特征提取模块的处理单元
class FeatureUnit(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(FeatureUnit, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope))
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope))

        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.leakRelu6 = nn.LeakyReLU(negative_slope=negative_slope)

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.leakRelu7 = nn.LeakyReLU(negative_slope=negative_slope)

        self.conv8 = nn.Conv2d(32, 32, 3, padding=1)
        self.leakRelu8 = nn.LeakyReLU(negative_slope=negative_slope)

        self.conv9 = nn.Conv2d(32, 2, 3, padding=1)
        self.leakRelu9 = nn.LeakyReLU(negative_slope=negative_slope)
        initialize_weights(self)

    def forward(self, x):
        #Encoder
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        # Decoder
        out6 = self.conv6(out5)
        out6 = self.leakRelu6(out3 + out6)
        out7 = self.conv7(out6)
        out7 = self.leakRelu6(out2 + out7)
        out8 = self.conv8(out7)
        out8 = self.leakRelu6(out1 + out8)
        out9 = self.conv9(out8)
        out9 = self.leakRelu9(x + out9)

        return out9


# 单域处理模块的一折
class SingleFoldDomain(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(SingleFoldDomain, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.LeakyReLU(inplace=True))

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(inplace=True))

        self.down4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(inplace=True))

        self.up1 = Up(128, 64)
        self.ac1 = nn.LeakyReLU(inplace=True)
        self.up2 = Up(128, 64)
        self.ac2 = nn.LeakyReLU(inplace=True)
        self.up3 = Up(128, 2)
        self.ac3 = nn.LeakyReLU(inplace=True)
        self.up4 = Up(4, 2)
        self.ac4 = nn.LeakyReLU(inplace=True)

        initialize_weights(self)

    def forward(self, x):
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
        x8 = self.ac4(x8+x)
        return x8