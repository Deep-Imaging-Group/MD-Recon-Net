import torch
from torch.autograd import Variable
import numpy as np
from libtiff import TIFF

def create_input(*input):
    u_img, u_k, img, k = input
    u_img = Variable(u_img, requires_grad=True).cuda()
    u_k = Variable(u_k, requires_grad=True).cuda()
    img = Variable(img, requires_grad=False).cuda()
    k = Variable(k, requires_grad=False).cuda()
    return u_img, u_k, img, k

def abs(x):
    y = np.abs(x)
    min = np.min(y)
    max = np.max(y)
    y = (y - min)/ (max - min)
    return np.abs(y)

def idc(x, y, mask):
    '''
        x: the undersampled kspace
        y: the restored kspace from x
        mask: the undersampling mask
        return:
    '''
    return x + y * (1 - mask)

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


def create_radial_mask(infile):
    img = TIFF.open(infile, mode="r")
    for im in list(img.iter_images()):
        im = im / 255
    mask = np.zeros_like(im)
    mask[0:128, 0:128] = im[128:, 128:]
    mask[128:, 128:] = im[0:128, 0:128]
    mask[0:128, 128:] = im[128:, 0:128]
    mask[128:, 0:128] = im[0:128, 128:]
    return mask