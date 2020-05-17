import torch
import torch.nn as nn


from .feature_extractor import FeatureExtractor, FeatureExtractorLoss
from .fusion_model import FusionModel, FusionLoss
from .reconstruction_model import ReconstructionUnetUnit, ReconstructionUnitLoss, ReconstructionForwardUnit
from .util import DC
# from utils import mymath

ReconstructionUnit = ReconstructionForwardUnit

def  fft(input):
    # (N, 2, W, H) -> (N, W, H, 2)
    # print(type(input))
    input = input.permute(0, 2, 3, 1)
    input = torch.fft(input, 2, normalized=True)

    # (N, W, H, 2) -> (N, 2, W, H)
    input = input.permute(0, 3, 1, 2)
    return input

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2, normalized=True)

    # (N, D, W, H, 2) -> (N, 2, D, W, H)
    input = input.permute(0, 3, 1, 2)

    return input


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



class MRIReconstruction(nn.Module):
    def __init__(self, mask, w, bn):
        super(MRIReconstruction, self).__init__()
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


        


class MRIReconstructionLoss(nn.Module):
    def __init__(self, l1, l2, l3):
        super(MRIReconstructionLoss, self).__init__()
        self.weight = [l1, l2, l3]
        self.feature_loss = FeatureExtractorLoss()
        self.fusion_loss = FusionLoss()
        self.recon_loss = ReconstructionUnitLoss()

    def forward(self, *input):
        k_feature = input[0]
        img_feature= input[1]

        k_fusion = input[2]
        img_fusion = input[3]

        img_recon = input[4]
        f_k = input[5]
        f_img = input[6]

        loss1 = self.feature_loss(*(k_feature, img_feature, f_k, f_img))
        loss2 = self.fusion_loss(*(k_fusion, img_fusion, f_k, f_img))
        loss3 = self.recon_loss(*(img_recon, f_img))

        return self.weight[0] * loss1 + self.weight[1] * loss2 + self.weight[2] * loss3

import torchvision.models as models

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = models.vgg19(True).features
    
    def forward(self, x):
        x1 = torch.norm(x, dim=1)
        r = torch.unsqueeze(x1 - 103.939, dim=1)
        g = torch.unsqueeze(x1 - 123.68, dim=1)
        b = torch.unsqueeze(x1 - 123.68, dim=1)

        x2 = torch.cat((r, g, b), dim=1)

        out = self.features(x2)

        return out