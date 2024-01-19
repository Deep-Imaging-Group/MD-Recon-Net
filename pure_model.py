import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
import matplotlib.pyplot as plt




class MDReconstructionNet(nn.Model):
    def __init__(self, args, mask, w, bn, training):
        super().__init__()
        self.cnn1 = FeatureExtractor(bn=bn)
        self.dc11 = DC()
        self.dc12 = DC()
        self.fusion11 = Fusion()
        self.fusion12 = Fusion()

        self.cnn2 = FeatureExtractor(bn=bn)
        self.dc21 = DC()
        self.dc22 = DC()
        self.fusion21 = Fusion()
        self.fusion22 = Fusion()


        self.cnn3 = FeatureExtractor(bn=bn)
        self.dc31 = DC()
        self.dc32 = DC()
        self.fusion31 = Fusion()
        self.fusion32 = Fusion()


        self.cnn4 = FeatureExtractor(bn=bn)
        self.dc41 = DC()
        self.dc42 = DC()
        self.fusion41 = Fusion()
        self.fusion42 = Fusion()

        self.cnn5 = FeatureExtractor(bn=bn)
        self.dc51 = DC()
        self.dc52 = DC()
        self.fusion51 = Fusion()
        # self.fusion52 = Fusion()

        self.mask = mask
        self.w = w

    def forward(self, *input):
        ############################## First Stage ######################################
        # resstore feature from raw data
        k_x_1 = input[0]
        img_x_1 = input[1]
        u_k = k_x_1

        k_fea_1, img_fea_1 = self.cnn1(*(k_x_1, img_x_1))

        rec_k_1 = self.dc11(k_fea_1, u_k, self.mask)
        rec_img_1 = self.dc12(img_fea_1, u_k, self.mask, True)

        k_to_img_1 = ifft(rec_k_1)  # convert the restored kspace to spatial domain
        img_to_k_1 = fft(rec_img_1) # convert the restored image to frequency domain


        ################################ Second Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_2 = self.fusion11(rec_k_1, img_to_k_1)
        img_x_2 = self.fusion12(rec_img_1, k_to_img_1)

        k_fea_2, img_fea_2 = self.cnn2(*(k_x_2, img_x_2))

        rec_k_2 = self.dc21(k_fea_2, u_k, self.mask)
        rec_img_2 = self.dc22(img_fea_2, u_k, self.mask, True)

        k_to_img_2 = ifft(rec_k_2)  # convert the restored kspace to spatial domain
        img_to_k_2 = fft(rec_img_2) # convert the restored image to frequency domain


        ################################ Third Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_3 = self.fusion21(rec_k_2, img_to_k_2)
        img_x_3 = self.fusion22(rec_img_2, k_to_img_2)

        k_fea_3, img_fea_3 = self.cnn3(*(k_x_3, img_x_3))

        rec_k_3 = self.dc31(k_fea_3, u_k, self.mask)
        rec_img_3 = self.dc32(img_fea_3, u_k, self.mask, True)

        k_to_img_3 = ifft(rec_k_3)  # convert the restored kspace to spatial domain
        img_to_k_3 = fft(rec_img_3) # convert the restored image to frequency domain


        ################################ Forth Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_4 = self.fusion31(rec_k_3, img_to_k_3)
        img_x_4 = self.fusion32(rec_img_3, k_to_img_3)

        k_fea_4, img_fea_4 = self.cnn4(*(k_x_4, img_x_4))

        rec_k_4 = self.dc41(k_fea_4, u_k, self.mask)
        rec_img_4 = self.dc42(img_fea_4, u_k, self.mask,  True)

        k_to_img_4 = ifft(rec_k_4)  # convert the restored kspace to spatial domain
        img_to_k_4 = fft(rec_img_4) # convert the restored image to frequency domain


        ################################ Third Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_5 = self.fusion41(rec_k_4, img_to_k_4)
        img_x_5 = self.fusion42(rec_img_4, k_to_img_4)

        k_fea_5, img_fea_5 = self.cnn5(*(k_x_5, img_x_5))

        rec_k_5 = self.dc51(k_fea_5, u_k, self.mask)
        rec_img_5 = self.dc52(img_fea_5, u_k, self.mask, True)

        k_to_img_5 = ifft(rec_k_5)  # convert the restored kspace to spatial domain



        out = self.fusion51(rec_img_5, k_to_img_5)

        return out
    

    


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


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, x1, x2):
        return x1 * 1 / (1 + self.w) + x2 * self.w / (self.w + 1)

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

def Sequential(cnn, norm, ac, bn=True):
    if bn:
        return nn.Sequential(cnn, norm, ac)
    else:
        return nn.Sequential(cnn, ac)
    
    
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

def fft(input):
    complex_input = torch.complex(input[:, 0, :, :], input[:, 1, :, :])
    kspace = torch.fft.fft2(complex_input, norm="ortho")
    kspace = torch.stack([kspace.real, kspace.imag], dim=1)

    return kspace


    

def ifft(input):
    complex_input = torch.complex(input[:, 0, :, :], input[:, 1, :, :])
    img = torch.fft.ifft2(complex_input, norm="ortho")
    real = img.real
    imag = img.imag

    return torch.stack([real, imag], dim=1)
    


class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, rec, u_k, mask, is_img=False):
        if is_img:
            rec = fft(rec)
        result = mask * (rec * self.w / (1 + self.w) + u_k * 1 / (self.w + 1)) # weighted the undersampling and reconstruction
        result = result + (1 - mask) * rec # non-sampling point

        if is_img:
            result = ifft(result)
        
        return result



class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, x1, x2):
        return x1 * 1 / (1 + self.w) + x2 * self.w / (self.w + 1)

def create_complex_value(x):
    '''
        x: (2, h, w)
        return:
            numpy, (h, w), dtype=np.complex
    '''
    result = np.zeros_like(x[0], dtype=np.complex)
    result.real = x[0]
    result.imag = x[1]
    return result