import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd.function as F
from torch.utils.data import Dataset
import numpy as np

class ImageRestoration(nn.Module):
    def __init__(self):
        super(ImageRestoration, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(inplace=True))
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.LeakyReLU(inplace=True))
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

class Data(Dataset):
    def __init__(self, root, acc=4):
        # x.shape: 500, 2, 256, 150
        self.x = np.load(root + "x_img/%d.npy" % 0)
        self.y = np.load(root + "label/%d.npy" % 0)
        self.start = 0
        self.end = 500
        self.root = root

    def __len__(self):
        return 25000

    def __getitem__(self, index):
        if (index < self.start or index >= self.end):
            self.x = np.load(self.root + "x_img/%d.npy" % (index // 500))
            self.y = np.load(self.root + "label/%d.npy" % (index // 500))
        x = self.x[index%500].real
        y = self.y[index % 500]
        z = x.shape
        x = np.reshape(x, (1, z[0], z[1]))
        y = np.reshape(y, (1, z[0], z[1]))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y
