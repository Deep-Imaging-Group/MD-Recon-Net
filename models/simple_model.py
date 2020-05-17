import torch
import torch.nn as nn

from .util import initialize_weights


class DualCnn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DualCnn, self).__init__()
        self.top = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True))

    def forward(self, *input):
        u_k = input[0]
        u_img = input[1]
        return self.top(u_k), self.down(u_img)

class DualFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DualFusion, self).__init__()
        self.cnn = DualCnn(in_ch*2, out_ch)

    def forward(self, *input):
        _u_k = input[0]
        _u_img = input[1]
        u_k = torch.cat((_u_k, _u_img), dim=1)
        u_img = torch.cat((_u_img, _u_k), dim=1)

        return self.cnn(*(u_k, u_img))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # first 5 layers
        self.conv1 = DualCnn(2, 32)
        self.conv2 = DualCnn(32, 32)
        self.conv3 = DualCnn(32, 32)
        self.conv4 = DualCnn(32, 32)
        self.conv5 = DualCnn(32, 32)

        # second 5 layers
        self.conv6 = DualFusion(32, 64)
        self.conv7 = DualCnn(64, 64)
        self.conv8 = DualCnn(64, 64)
        self.conv9 = DualCnn(64, 64)
        self.conv10 = DualCnn(64, 64)

        # last 5 layers
        self.conv11 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv13 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv14 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv15 = nn.Conv2d(128, 2, 3, padding=1)
        self.leakrelu = nn.LeakyReLU(inplace=True)
        initialize_weights(self)

    def forward(self, *input):
        u_k = input[0]
        u_img = input[1]

        out1 = self.conv1(*(u_k, u_img))
        out2 = self.conv2(*out1)
        out3 = self.conv3(*out2)
        out4 = self.conv4(*out3)
        out5 = self.conv5(*out4)

        out6 = self.conv6(*out5)
        out7 = self.conv7(*out6)
        out8 = self.conv8(*out7)
        out9 = self.conv9(*out8)
        out10 = self.conv10(*out9)

        tmp = torch.cat(out10, dim=1)
        out11 = self.conv11(tmp)
        out12 = self.conv12(out11)
        out13 = self.conv13(out12)
        out14 = self.conv14(out13)
        out15 = self.conv15(out14)
        out15 += u_img
        return self.leakrelu(out15)
