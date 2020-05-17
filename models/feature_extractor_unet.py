import torch
import torch.nn as nn

from .complex_conv import ComplexCnn2d
from .util import initialize_weights

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

        self.up1 = Up(64, 32)
        self.ac1 = nn.LeakyReLUnegative_slope=negative_slope()
        self.up2 = Up(64, 32)
        self.ac2 = nn.LeakyReLU(negative_slope=negative_slope)
        self.up3 = Up(64, 32)
        self.ac3 = nn.LeakyReLU(negative_slope=negative_slope)
        self.up4 = Up(64, 32)
        self.ac4 = nn.LeakyReLU(negative_slope=negative_slope)
        self.up5 = Up(34, 2)
        self.ac5 = nn.LeakyReLU(negative_slope=negative_slope)
        initialize_weights(self)

    def forward(self, x):
        #Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Decoder
        x6 = self.up1(*(x4, x5))
        x6 = self.ac1(x6)
        x7 = self.up2(*(x3, x6))
        x7 = self.ac2(x7)
        x8 = self.up2(*(x2, x7))
        x8 = self.ac2(x8)
        x9 = self.up2(*(x1, x8))
        x9 = self.ac2(9)
        x10 = self.up2(*(x, x9))
        x10 = self.ac2(x10)

        return x9


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.kspace_extractor = FeatureUnit()
        self.image_extractor = FeatureUnit()
        initialize_weights(self)

    def forward(self, *input):
        k, img = input
        k_feature = self.kspace_extractor(k)
        img_feature = self.image_extractor(img)

        return k_feature, img_feature


class FeatureExtractorLoss(nn.Module):
    def __init__(self):
        super(FeatureExtractorLoss, self).__init__()
        self.k_loss = nn.MSELoss()
        self.img_loss = nn.MSELoss()

    def forward(self, *input):
        k_loss = self.k_loss(input[0], input[2])
        img_loss = self.img_loss(input[1], input[3])

        return k_loss  + img_loss


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(2, 2, 3, 1, 1)

    def forward(self, input):
        return self.conv1(input)
