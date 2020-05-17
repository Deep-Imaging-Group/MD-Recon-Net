import torch
import torch.nn as nn

from .complex_conv import ComplexCnn2d
from .util import initialize_weights, Sequential

# 简单的前馈，无残差连接无u-net结构
class FeatureForwardUnit(nn.Module):
    def __init__(self, negative_slope=0.01, bn=True):
        super(FeatureForwardUnit, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        # self.conv5 = Sequential(
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv6 = nn.Conv2d(32, 2, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        # out5 = self.conv5(out4)
        out6 = self.conv6(out4)
        output = self.ac6(out6 + x)

        return output

class FeatureExtractor(nn.Module):
    def __init__(self, bn):
        super(FeatureExtractor, self).__init__()
        ############################################################
        # self.kspace_extractor = FeatureResidualUnit()
        # self.image_extractor = FeatureResidualUnit()

        ###########################################################
        self.kspace_extractor = FeatureForwardUnit(bn=bn)
        self.image_extractor = FeatureForwardUnit(bn=bn)

        ############################################################

        initialize_weights(self)

    def forward(self, *input):
        k, img = input
        k_feature = self.kspace_extractor(k)
        img_feature = self.image_extractor(img)

        return k_feature, img_feature


#   特征提取模块的处理单元,使用了residual
class FeatureResidualUnit(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(FeatureResidualUnit, self).__init__()
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
