import os

import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.fft import fftshift
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

from utils.read_tiff import tiff_to_read
from utils.undersample import undersample
from utils.mymath import fft2c
import glob


class KneeDataset(Dataset):
    def __init__(self, mask, training=True, is_3d=False):
        '''
        rate: int, 10, 20, 30, 40, undersampled rate
        training: bool
            True: load training dataset
            False: load valid dataset
        '''
        if training:
            self.root = "./data/knees/db_train/"
        else:
            self.root = "./data/knees/db_valid/"
        self.files = os.listdir(self.root)
        self.mask = mask
        self.indexs = []
        self.is_3d = is_3d

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(index, end="  ")
        tif = tiff_to_read(self.root + self.files[index])
        label_img = np.zeros_like(tif[0], dtype=np.complex)
        label_img.real = tif[0]
        label_img.imag = tif[1]
        max = np.max(np.abs(label_img))
        min = np.min(np.abs(label_img))
        # label_img = (label_img - min) / (max - min)
        u_img, u_k, label_k = undersample(label_img, self.mask, True)

        # print(label_img.shape, label_k.shape, u_img.shape, u_k.shape)

        size = label_img.shape
        size = (2, size[0], size[1])
        under_img = torch.zeros(size)
        under_k = torch.zeros(size)

        full_img = torch.zeros(size)
        full_k = torch.zeros(size)

        under_img[0] = torch.from_numpy(u_img.real)
        under_img[1] = torch.from_numpy(u_img.imag)

        under_k[0] = torch.from_numpy(u_k.real)
        under_k[1] = torch.from_numpy(u_k.imag)

        full_img[0] = torch.from_numpy(label_img.real)
        full_img[1] = torch.from_numpy(label_img.imag)

        full_k[0] = torch.from_numpy(label_k.real)
        full_k[1] = torch.from_numpy(label_k.imag)

        if (self.is_3d):
            under_img = under_img.view(1, 2, 256, 256)
            under_k = under_k.view(1, 2, 256, 256)
            full_img = full_img.view(1, 2, 256, 256)
            full_k = full_k.view(1, 2, 256, 256)
        return under_img, under_k, full_img, full_k


'''
Calgary-Campinas-359 (CC-359)
'''
class LeftBrainDataset(Dataset):
    def __init__(self, mask, training=True, is_3d=False ):
        '''
        rate: int, 10, 20, 30, 40, undersampled rate
        training: bool
            True: load training dataset
            False: load valid dataset
        '''
        data = None
        if training:
            path = "./data/cc_data/left/train/*.npy"
        else:
            path = "./data/cc_data/left/val/*.npy"
        files = np.asarray(glob.glob(path))
        files.sort()
        # print(files)
        for file in files:
            # print(file)
            item = np.load(file)
            if data is None:
                data = item
            else:
                data = np.append(data, item, axis=2)
        self.data = data

        print("data size: ", self.data.shape)
        self.mask = mask
        self.is_3d = is_3d

    def __len__(self):
        return self.data.shape[2]

    def __getitem__(self, index):
        label_img = self.data[:, :, index]

        norm = np.abs(label_img)
        min = np.min(np.abs(label_img))
        max = np.max(np.abs(label_img))

        label_img = (label_img - min) / (max - min) * 255

        u_img, u_k, label_k = undersample(label_img, self.mask, True)

        size = label_img.shape
        size = (2, size[0], size[1])

        under_img = torch.zeros(size)
        under_k = torch.zeros(size)
        full_img = torch.zeros(size)
        full_k = torch.zeros(size)

        under_img[0] = torch.from_numpy(u_img.real)
        under_img[1] = torch.from_numpy(u_img.imag)

        under_k[0] = torch.from_numpy(u_k.real)
        under_k[1] = torch.from_numpy(u_k.imag)

        full_img[0] = torch.from_numpy(label_img.real)
        full_img[1] = torch.from_numpy(label_img.imag)

        full_k[0] = torch.from_numpy(label_k.real)
        full_k[1] = torch.from_numpy(label_k.imag)

        if (self.is_3d):
            under_img = under_img.view(1, 2, 256, 256)
            under_k = under_k.view(1, 2, 256, 256)
            full_img = full_img.view(1, 2, 256, 256)
            full_k = full_k.view(1, 2, 256, 256)
        return under_img, under_k, full_img, full_k


class BrainDataset3D(Dataset):
    def __init__(self, mask, training=True, is_3d=False, files=None ):
        '''
        rate: int, 10, 20, 30, 40, undersampled rate
        training: bool
            True: load training dataset
            False: load valid dataset
        '''
        data = None
        if files is None:
            if training:
                path = "./data/cc_data/left3d/train/*.npy"
            else:
                path = "./data/cc_data/left3d/val/*.npy"
            files = np.asarray(glob.glob(path))
        files.sort()
        # print(files)
        for file in files[:1]:
            # print(file)
            item = np.load(file)
            if data is None:
                data = item
            else:
                data = np.append(data, item, axis=3)
        self.data = data

        print("data size: ", self.data.shape)
        self.mask = mask.cpu()
        # print(type(mask))

    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, index):
        label_img = self.data[:, :, :, index]

        norm = np.abs(label_img)
        min = np.min(np.abs(label_img))
        max = np.max(np.abs(label_img))

        label_img = (label_img - min) / (max - min) * 255
        full_img = torch.zeros(*label_img.shape, 2)
        full_img[:, :, :, 0] = torch.from_numpy(label_img.real)
        full_img[:, :, :, 1] = torch.from_numpy(label_img.imag)

        full_k = torch.fft(full_img, 2, normalized=True)
        tmp = full_k.permute(3, 0, 1, 2)
        under_k = tmp * self.mask

        tmp = under_k.permute(1, 2, 3, 0)
        under_img = torch.ifft(tmp, 2, normalized=True)

        full_k = full_k.permute(3, 0, 1, 2)
        full_img = full_img.permute(3, 0, 1, 2)
        under_img = under_img.permute(3, 0, 1, 2)

        # print(full_k.size(), full_img.size(), under_k.size(), under_img.size())
        
        return under_img, under_k, full_img, full_k


class BrainDataset3D_2D(Dataset):
    def __init__(self, mask, training=True, is_3d=False ):
        '''
        rate: int, 10, 20, 30, 40, undersampled rate
        training: bool
            True: load training dataset
            False: load valid dataset
        '''
        data = None
        if training:
            path = "./data/cc_data/left3d/train/*.npy"
        else:
            path = "./data/cc_data/left3d/val/*.npy"
        files = np.asarray(glob.glob(path))
        files.sort()
        # print(files)
        for file in files:
            # print(file)
            item = np.load(file)
            if data is None:
                data = item
            else:
                data = np.append(data, item, axis=3)
        
        _, h, w, _ = data.shape
        data = np.transpose(data, (1, 2, 0, 3))
        data = np.reshape(data, (h, w, -1))
        self.data = data

        print("data size: ", self.data.shape)
        self.mask = mask.cpu()
        # print(type(mask))

    def __len__(self):
        return self.data.shape[-1]

    def __getitem__(self, index):
        label_img = self.data[:, :, index]

        norm = np.abs(label_img)
        min = np.min(np.abs(label_img))
        max = np.max(np.abs(label_img))

        label_img = (label_img - min) / (max - min) * 255
        full_img = torch.zeros(*label_img.shape, 2)
        full_img[:, :, 0] = torch.from_numpy(label_img.real)
        full_img[:, :, 1] = torch.from_numpy(label_img.imag)

        full_k = torch.fft(full_img, 2, normalized=True)
        tmp = full_k.permute(2, 0, 1)
        under_k = tmp * self.mask

        tmp = under_k.permute(1, 2, 0)
        under_img = torch.ifft(tmp, 2, normalized=True)

        full_k = full_k.permute(2, 0, 1)
        full_img = full_img.permute(2, 0, 1)
        under_img = under_img.permute(2, 0, 1)

        # print(full_k.size(), full_img.size(), under_k.size(), under_img.size())
        
        return under_img, under_k, full_img, full_k

class BrainDataset(Dataset):
    def __init__(self, mask, files=None):
        '''
        rate: int, 10, 20, 30, 40, undersampled rate
        training: bool
            True: load training dataset
            False: load valid dataset
        '''
        data = None
        # print(files)
        for file in files:
            # print(file)
            item = np.load(file)
            if data is None:
                data = item
            else:
                data = np.append(data, item, axis=0)
        
        
        self.data = data

        print("data size: ", self.data.shape)
        self.mask = mask.cpu()
        # print(type(mask))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        label_img = self.data[index]

        norm = np.abs(label_img)
        min = np.min(np.abs(label_img))
        max = np.max(np.abs(label_img))

        label_img = (label_img - min) / (max - min) * 255
        full_img = torch.zeros(*label_img.shape, 2)
        full_img[:, :, 0] = torch.from_numpy(label_img.real)
        full_img[:, :, 1] = torch.from_numpy(label_img.imag)

        full_k = torch.fft(full_img, 2, normalized=True)
        tmp = full_k.permute(2, 0, 1)
        under_k = tmp * self.mask

        tmp = under_k.permute(1, 2, 0)
        under_img = torch.ifft(tmp, 2, normalized=True)

        full_k = full_k.permute(2, 0, 1)
        full_img = full_img.permute(2, 0, 1)
        under_img = under_img.permute(2, 0, 1)

        # print(full_k.size(), full_img.size(), under_k.size(), under_img.size())
        
        return under_img, under_k, full_img, full_k


class KneeDataset(Dataset):
    def __init__(self, mask, files=None):
        '''
        rate: int, 10, 20, 30, 40, undersampled rate
        training: bool
            True: load training dataset
            False: load valid dataset
        '''
        data = None
        # print(files)
        for file in files:
            # print(file)
            item = np.load(file)
            if data is None:
                data = item
            else:
                data = np.append(data, item, axis=0)
        
        
        self.data = data

        print("data size: ", self.data.shape)
        self.mask = mask.cpu()
        # print(type(mask))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        label_img = self.data[index]

        # label_img = label_img[np.newaxis]

        norm = np.abs(label_img)
        min = np.min(np.abs(label_img))
        max = np.max(np.abs(label_img))

        label_img = (label_img - min) / (max - min) * 255
        full_img = torch.zeros(*label_img.shape, 2)
        full_img[:, :, 0] = torch.from_numpy(label_img.real)
        full_img[:, :, 1] = torch.from_numpy(label_img.imag)

        full_k = torch.fft(full_img, 2, normalized=True)
        tmp = full_k.permute(2, 0, 1)
        under_k = tmp * self.mask

        tmp = under_k.permute(1, 2, 0)
        under_img = torch.ifft(tmp, 2, normalized=True)

        full_k = full_k.permute(2, 0, 1)
        full_img = full_img.permute(2, 0, 1)
        under_img = under_img.permute(2, 0, 1)

        # print(full_k.size(), full_img.size(), under_k.size(), under_img.size())
        
        return under_img, under_k, full_img, full_k
