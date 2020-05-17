import torch
import torch.nn as nn

from .complex_conv import ComplexCnn2d
from .util import initialize_weights, Sequential


#   特征提取模块的处理单元
class FusionUnetUnit(nn.Module):
    def __init__(self, negative_slope=0.01):
        super(FusionUnetUnit, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.down4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

        self.up1 = Up(64, 32)
        self.ac1 = nn.LeakyReLU(negative_slope=negative_slope)
        self.up2 = Up(64, 32)
        self.ac2 = nn.LeakyReLU(negative_slope=negative_slope)
        self.up3 = Up(64, 32)
        self.ac3 = nn.LeakyReLU(negative_slope=negative_slope)
        self.up4 = Up(36, 2)
        self.ac4 = nn.LeakyReLU(negative_slope=negative_slope)

        initialize_weights(self)

    def forward(self, *x):
        #Encoder
        rec, u_data = x
        x1 = self.down1(rec)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(*(x3, x4))
        x5 = self.ac1(x5)
        x6 = self.up2(*(x2, x5))
        x6 = self.ac2(x6)
        x7 = self.up3(*(x1, x6))
        x7 = self.ac3(x7)
        x8 = self.up4(*(rec, x7))
        x8 = self.ac4(x8)
        return x8


class FusionForwardUnit(nn.Module):
    def __init__(self, bn, negative_slope=0.01):
        super(FusionForwardUnit, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv5 = Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv6 = nn.Conv2d(32, 2, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, *input):
        x, u_x = input
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        output = self.ac6(out6 + u_x)

        return output


class FusionModel(nn.Module):
    def __init__(self, bn):
        super(FusionModel, self).__init__()
        ######################################################
        # self.kspace_fusion = FusionUnetUnit()
        # self.image_fusion = FusionUnetUnit()

        ######################################################
        self.kspace_fusion = FusionForwardUnit(bn=bn)
        self.image_fusion = FusionForwardUnit(bn=bn)


    def forward(self, *input):
        rec_k, rec_img, u_k, u_img = input
        kspace = self.kspace_fusion(rec_k, u_k)
        img = self.image_fusion(rec_img, u_img)

        return kspace, img



class Up(nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.01):
        super(Up, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, *input):
        x1 = input[0]
        x2 = input[1]

        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)



class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.k_loss = nn.MSELoss()
        self.img_loss = nn.MSELoss()

    def forward(self, *input):
        k_loss = self.k_loss(input[0], input[2])
        img_loss = self.img_loss(input[1], input[3])

        return k_loss + img_loss
