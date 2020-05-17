import time
import random
import argparse
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from numpy.fft import fftshift, ifftshift
import scipy.io as sio
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

from models.dual_domain_fusion_model import MRIReconstruction, MRIReconstructionLoss
# from models.data import KneeDataset as Dataset
from utils.data import BrainDataset as Dataset
from utils.vis_tools import Visualizer
from utils.util import abs, create_input, idc, create_complex_value
from utils import mymath




def main():
    lr = args.lr
    acc = args.rate
    batch_size = args.batch_size
    epochs = args.epochs
    mask_type = args.mask
    bn = args.bn
    w = args.w
    model_type = args.model
    data_type = args.data

    test_val = True

    if bn:
        path = "./params/%s_bn_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    else:
        # path = "./params/%s_%s_%s_%d_6.pkl" % (model_type, data_type,mask_type, acc)
        path = "./params/%s_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    print(path)
    # mask = np.load("./mask/%s/%s/%d.npy" % (data_type, mask_type, acc))
    # mask = np.load("./mask/%s/%s/256_256_%d.npy" % (data_type, mask_type, acc))

    mask = sio.loadmat("./mask/%s/%s/%s_256_256_%d.mat" % (data_type, mask_type, mask_type, acc))['Umask']
    print("./mask/%s/%s/%s_256_256_%d.mat" % (data_type, mask_type, mask_type, acc))
    mask = fftshift(mask, axes=(-2, -1))
    mask_torch = torch.from_numpy(mask).float().cuda()
    # print(mask_torch.size())
    


    model = MRIReconstruction(mask_torch, w, bn).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 2, 3])
    dataset = Dataset(mask_torch)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = MRIReconstructionLoss(0.01, 0.1, 1)
    criterion = nn.MSELoss()

    valid_dataset = Dataset(mask_torch, training=False)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    
    if os.path.exists(
            path):
        # model.load_state_dict(
        #     torch.load(path))
        model.load_state_dict(
            torch.load(path, map_location='cuda:0'))
        print("Finished load model parameters!")
    
    print(lr, acc, batch_size, epochs)

    vis = Visualizer()
    for epoch in range(epochs):
        timestr = time.strftime("%H:%M:%S", time.localtime())
        print(timestr, end=": ")
        loss = 0
        valid_dataset.indexs = []
        _i = 1

        if test_val:
            with torch.no_grad():
                for data in valid_dataloader:
                    u_img, u_k, f_img, f_k = create_input(*data)
                    result = model(*(u_k, u_img))
                    loss += criterion(f_img, result).cpu().item()
                    _i += 1


        ############################## 可视化###########################
        # the common section
        

        print("Epoch: ", epoch, "Rate: ", acc, "Test Loss: ", loss / _i, end=" ")
        index = 40

        data = valid_dataset[index]
        a = torch.unsqueeze(data[0], dim=0)
        b = torch.unsqueeze(data[1], dim=0)
        c = torch.unsqueeze(data[2], dim=0)
        d = torch.unsqueeze(data[3], dim=0)
        u_img, u_k, f_img, f_k = create_input(*(a, b, c, d))

        result = model(*(u_k, u_img))
        index=0

        u_img, u_k, f_img, f_k = create_input(*(a, b, c, d))
        u_img = u_img[index].detach().cpu().numpy()
        u_img = create_complex_value(u_img)
        img = f_img[index].detach().cpu().numpy()
        img = create_complex_value(img)
        vis.plot("%s %s Test Loss - %d" % (model_type, mask_type, acc), loss)
        vis.img("%s %s Undersampled image %d" % (model_type, mask_type, acc), abs(u_img))
        vis.img("%s %s full image %d" % (model_type, mask_type, acc), abs(img))
        vis.img("%s %s Mask %d" % (model_type, mask_type, acc), mask)

        out = result[index].detach().cpu().numpy()
        out = create_complex_value(out)
        vis.img("%s %s Reconstructed image %d" % (model_type, mask_type, acc), np.abs(out))

        max_value = np.max(np.abs(img))
        max_value = 255
        print("SSIM: ", compare_ssim(np.abs(img), np.abs(out), data_range=max_value), "  ", compare_ssim(np.abs(img), np.abs(u_img), data_range=max_value), end="  ")

        print("SSIM: ", compare_psnr(np.abs(img), np.abs(out), data_range=max_value), "  ", compare_psnr(np.abs(img), np.abs(u_img), data_range=max_value))

        np.save("./data/rec_model3.npy", out)
        np.save("./data/under.npy", u_img)
        np.save("./data/full_img", img)



        for data in dataloader:
            u_img, u_k, f_img, f_k = create_input(*data)
            result = model(*(u_k, u_img))
            loss = criterion(f_img, result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), path)


def test():
    lr = args.lr
    acc = args.rate
    batch_size = args.batch_size
    epochs = args.epochs
    mask_type = args.mask
    bn = args.bn
    w = args.w
    model_type = args.model
    data_type = args.data

    test_val = True

    if bn:
        path = "./params/%s_bn_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    else:
        # path = "./params/%s_%s_%s_%d_6.pkl" % (model_type, data_type,mask_type, acc)
        path = "./params/%s_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    print(path)
    # mask = np.load("./mask/%s/%s/%d.npy" % (data_type, mask_type, acc))
    # mask = np.load("./mask/%s/%s/256_256_%d.npy" % (data_type, mask_type, acc))

    mask = sio.loadmat("./mask/%s/%s/%s_256_256_%d.mat" % (data_type, mask_type, mask_type, acc))['Umask']
    mask = fftshift(mask, axes=(-2, -1))
    mask_torch = torch.from_numpy(mask).float().cuda()


    model = MRIReconstruction(mask_torch, w, bn).cuda()

    from utils.test_data import BrainDataset as Dataset
    dataset = Dataset(mask_torch, ["./output/selected/selected.npy"])
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

    
    if os.path.exists(
            path):
        model.load_state_dict(
            torch.load(path))
        print("Finished load model parameters!")
    
    print(lr, acc, batch_size, epochs)

    for data in dataloader:
        u_img, u_k, f_img, f_k = create_input(*data)
        result = model(*(u_k, u_img)).cpu().detach().numpy()
        c, _, h, w = result.shape

        res = np.zeros((c, h, w), dtype=np.complex)

        res.real = result[:, 0]
        res.imag = result[:, 1]

        np.save("./output/result_%s_%d_ours.npy" % (mask_type, acc), res)
        print("INFO: Saved result . size: ", res.shape)


def test1():
    lr = args.lr
    acc = args.rate
    batch_size = args.batch_size
    epochs = args.epochs
    mask_type = args.mask
    bn = args.bn
    w = args.w
    model_type = args.model
    data_type = args.data

    test_val = True

    if bn:
        path = "./params/%s_bn_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    else:
        # path = "./params/%s_%s_%s_%d_6.pkl" % (model_type, data_type,mask_type, acc)
        path = "./params/%s_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    print(path)
    # mask = np.load("./mask/%s/%s/%d.npy" % (data_type, mask_type, acc))
    # mask = np.load("./mask/%s/%s/256_256_%d.npy" % (data_type, mask_type, acc))

    mask = sio.loadmat("./mask/%s/%s/%s_256_256_%d.mat" % (data_type, mask_type, mask_type, acc))['Umask']
    mask = fftshift(mask, axes=(-2, -1))
    mask_torch = torch.from_numpy(mask).float().cuda()


    model = MRIReconstruction(mask_torch, w, bn).cuda()

    from utils.test_data import BrainDataset as Dataset
    

    
    if os.path.exists(
            path):
        model.load_state_dict(
            torch.load(path))
        print("Finished load model parameters!")
    
    print(lr, acc, batch_size, epochs)
    if not os.path.exists('./pano/%s/%d/res' % (mask_type, acc)):
        os.makedirs('./pano/%s/%d/res' % (mask_type, acc))
    
    for i in range(190):
        data = sio.loadmat("./pano/%s/%d/f_img/%d.mat" % (mask_type, acc, i))['im_ori']

        np.save('./output/selected/selected.npy', data)

        dataset = Dataset(mask_torch, ["./output/selected/selected.npy"])
        dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

        for data in dataloader:
            u_img, u_k, f_img, f_k = create_input(*data)
            result = model(*(u_k, u_img)).cpu().detach().numpy()
            c, _, h, w = result.shape

            res = np.zeros((c, h, w), dtype=np.complex)

            res.real = result[:, 0]
            res.imag = result[:, 1]
            
            np.save("./pano/%s/%d/res/%d_%s_%s_%d" % (mask_type, acc, i, mask_type, model_type, acc), res)
            print("INFO: Saved result . size: ", res.shape)

def testKnee():
    lr = args.lr
    acc = args.rate
    batch_size = args.batch_size
    epochs = args.epochs
    mask_type = args.mask
    bn = args.bn
    w = args.w
    model_type = args.model
    data_type = args.data

    test_val = True

    if bn:
        path = "./params/%s_bn_brain_%s_%d.pkl" % (model_type,mask_type, acc)
    else:
        # path = "./params/%s_%s_%s_%d_6.pkl" % (model_type, data_type,mask_type, acc)
        path = "./params/%s_knee_%s_%d.pkl" % (model_type, mask_type, acc)
    print(path)
    # mask = np.load("./mask/%s/%s/%d.npy" % (data_type, mask_type, acc))
    # mask = np.load("./mask/%s/%s/256_256_%d.npy" % (data_type, mask_type, acc))

    # mask = sio.loadmat("./mask/knee/%s/mask_512_512_%d.mat" % (mask_type, acc))['Umask']
    mask = sio.loadmat("./mask/brain/%s/%s_256_256_%d.mat" % (mask_type, mask_type, acc))['Umask']
    mask = np.transpose(mask, (1, 0))
    # mask = np.pad(mask, 128, 'constant')
    mask = fftshift(mask, axes=(-2, -1))
    mask_torch = torch.from_numpy(mask).float().cuda()


    model = MRIReconstruction(mask_torch, w, bn).cuda()

    from utils.test_data import KneeDataset as Dataset
    dataset = Dataset(mask_torch, ["./output/selected/selected.npy"])
    dataloader = DataLoader(dataset, batch_size=6, shuffle=False)

    
    if os.path.exists(
            path):
        model.load_state_dict(
            torch.load(path))
        print("Finished load model parameters!")
    
    print(lr, acc, batch_size, epochs)

    for data in dataloader:
        u_img, u_k, f_img, f_k = create_input(*data)
        import time
        start = time.time()
        
        result = model(*(u_k, u_img))
        end = time.time()
        print((end -start)/6)
        result = result.cpu().detach().numpy()
        c, _, h, w = result.shape

        res = np.zeros((c, h, w), dtype=np.complex)

        res.real = result[:, 0]
        res.imag = result[:, 1]
        print(res.shape)
        np.save("./output/result_%s_%d_ours.npy" % (mask_type, acc), res)
        print("INFO: Saved result . size: ", res.shape)
    
        # import matplotlib.pyplot as plt

        # plt.subplot(2, 2, 1)
        # res = np.load("./output/result_%s_%d_ours.npy" % (mask_type, acc))
        # res = np.squeeze(res)
        # plt.imshow(np.abs(res), cmap="gray")

        # plt.subplot(2, 2, 2)
        # res = np.zeros((c, h, w), dtype=np.complex)
        # f_img = f_img.cpu().detach().numpy()
        # res.real = f_img[:, 0]
        # res.imag = f_img[:, 1]
        # res = np.squeeze(res)
        # plt.imshow(np.abs(res), cmap="gray") 


        # plt.subplot(2, 2, 3)
        # res = np.zeros((c, h, w), dtype=np.complex)
        # u_img = u_img.cpu().detach().numpy()
        # res.real = u_img[:, 0]
        # res.imag = u_img[:, 1]

        # res = np.squeeze(res)
        # plt.imshow(np.abs(res), cmap="gray")


        # plt.subplot(2, 2, 4)
        # # mask = ifftshift(mask, axes=(-2, -1))
        # plt.imshow(mask, cmap="gray")


        # plt.show()

def train_knee():
    from utils.data import KneeDataset as Dataset
    lr = args.lr
    acc = args.rate
    batch_size = args.batch_size
    epochs = args.epochs
    mask_type = args.mask
    bn = args.bn
    w = args.w
    model_type = args.model
    data_type = args.data
    cuda = args.cuda

    test_val = True

    if bn:
        path = "./params/%s_bn_%s_%s_%d.pkl" % (model_type, data_type, mask_type, acc)
    else:
        # path = "./params/%s_%s_%s_%d_6.pkl" % (model_type, data_type,mask_type, acc)
        path = "./params/%s_%s_%s_%d.pkl" % (model_type, data_type,mask_type, acc)
    print(path)
    # mask = np.load("./mask/%s/%s/%d.npy" % (data_type, mask_type, acc))
    # mask = np.load("./mask/%s/%s/256_256_%d.npy" % (data_type, mask_type, acc))

    # mask = sio.loadmat("./mask/%s/%s/%s_320_320_%d.mat" % (data_type, mask_type, mask_type, acc))['Umask']
    # mask = np.transpose(mask)
    # mask = fftshift(mask, axes=(-2, -1))
    # mask_torch = torch.from_numpy(mask).float().cuda()

    mask = sio.loadmat("./mask/brain/%s/%s_256_256_%d.mat" % (mask_type, mask_type, acc))['Umask']
    mask = np.transpose(mask, (1, 0))
    # print("./mask/brain/%s/%s_256_256_%d.mat" % (mask_type, mask_type, acc))

    # import matplotlib.pyplot as plt

    # plt.imshow(mask, cmap="gray")
    # plt.show()
    
    mask = fftshift(mask, axes=(-2, -1))
    mask_torch = torch.from_numpy(mask).float().cuda()
    # print(mask_torch.size())
    vis = Visualizer()


    model = MRIReconstruction(mask_torch, w, bn).cuda()
    # model = nn.DataParallel(model, device_ids=[0, 2, 3])
    dataset = Dataset(mask_torch)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = MRIReconstructionLoss(0.01, 0.1, 1)
    criterion = nn.MSELoss()

    valid_dataset = Dataset(mask_torch, training=False)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False)

    
    if os.path.exists(
            path):
        model.load_state_dict(
            torch.load(path, map_location='cuda:%d' % cuda))
        print("Finished load model parameters!")
        print(path)
    
    print(lr, acc, batch_size, epochs)


    for epoch in range(epochs):
        timestr = time.strftime("%H:%M:%S", time.localtime())
        print(timestr, end=": ")
        loss = 0
        valid_dataset.indexs = []
        _i = 1

        if test_val:
            with torch.no_grad():
                for data in valid_dataloader:
                    u_img, u_k, f_img, f_k = create_input(*data)        
                    result = model(*(u_k, u_img))
                    loss += criterion(f_img, result).cpu().item()
                    _i += 1


        ############################## 可视化###########################
        # the common section
        

        print("Epoch: ", epoch, "Rate: ", acc, "Test Loss: ", loss / _i, end=" ")
        index = 40

        data = valid_dataset[index]
        a = torch.unsqueeze(data[0], dim=0)
        b = torch.unsqueeze(data[1], dim=0)
        c = torch.unsqueeze(data[2], dim=0)
        d = torch.unsqueeze(data[3], dim=0)
        u_img, u_k, f_img, f_k = create_input(*(a, b, c, d))

        result = model(*(u_k, u_img))
        index=0

        u_img, u_k, f_img, f_k = create_input(*(a, b, c, d))
        u_img = u_img[index].detach().cpu().numpy()
        u_img = create_complex_value(u_img)
        img = f_img[index].detach().cpu().numpy()
        img = create_complex_value(img)
        vis.plot("%s %s Test Loss - %d" % (model_type, mask_type, acc), loss)
        vis.img("%s %s Undersampled image %d" % (model_type, mask_type, acc), abs(u_img))

        h, w = img.shape
        # img_ = img[h//2-128:h//2+128, w//2-128:w//2+128]
        vis.img("%s %s full image %d" % (model_type, mask_type, acc), abs(img))
        vis.img("%s %s Mask %d" % (model_type, mask_type, acc), mask)

        out = result[index].detach().cpu().numpy()
        out = create_complex_value(out)
        vis.img("%s %s Reconstructed image %d" % (model_type, mask_type, acc), np.abs(out))

        max_value = np.max(np.abs(img))
        max_value = 255
        print("SSIM: ", compare_ssim(np.abs(img), np.abs(out), data_range=max_value), "  ", compare_ssim(np.abs(img), np.abs(u_img), data_range=max_value), end="  ")

        print("SSIM: ", compare_psnr(np.abs(img), np.abs(out), data_range=max_value), "  ", compare_psnr(np.abs(img), np.abs(u_img), data_range=max_value))

        np.save("./data/rec_model3.npy", out)
        np.save("./data/under.npy", u_img)
        np.save("./data/full_img", img)



        for data in dataloader:
            u_img, u_k, f_img, f_k = create_input(*data)
            result = model(*(u_k, u_img))
            loss = criterion(f_img, result)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=5e-6, help="The learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=13, help="The batch size")
    parser.add_argument(
        "--epochs", type=int, default=10000, help="The number of epoch")
    parser.add_argument(
        "--w", type=float, default=0.2, help="The weighted parameter")
    parser.add_argument(
        "--rate",
        type=int,
        default=20,
        choices=[5, 10, 20, 25],
        help="The undersampling rate")
    parser.add_argument(
        "--mask",
        type=str,
        default="radial",
        choices=["cartesian", "radial", "random"],
        help="The type of mask")
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="The GPU device")
    parser.add_argument(
        "--bn",
        type=bool,
        default=False,
        choices=[True, False],
        help="Is there the batchnormalize")
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="which model")
    parser.add_argument(
        "--data",
        type=str,
        default="brain",
        choices=["brain", "knee"],
        help="which dataset")

    args = parser.parse_args()

    lr = args.lr
    acc = args.rate
    batch_size = args.batch_size
    epochs = args.epochs
    mask_type = args.mask
    bn = args.bn
    w = args.w
    model_type = args.model
    data_type = args.data
    cuda = args.cuda
    torch.cuda.set_device(cuda)
    main()
    # test()
    # if not os.path.exists('./pano/%s/%d/res' % (mask_type, acc)):
    #     os.makedirs('./pano/%s/%d/res' % (mask_type, acc))
    
    # for i in range(190):
    #     data = np.load("./pano/%s/%d/f_img/%d.npy" % (mask_type, acc, i))
    #     np.save('./output/selected/selected.npy', data)
    #     test()

    #     data = np.load("./output/result_%s_%d_ours.npy" % (mask_type, acc))

    #     np.save("./pano/%s/%d/res/%d_%s_%s_%d" % (mask_type, acc, i, mask_type, model_type, acc), data)


    # train_knee()
    # testKnee()



    #########################################################################################################################################

    # if not os.path.exists('./pano/%s/%d/diff/res' % (mask_type, acc)):
    #     os.makedirs('./pano/%s/%d/diff/res' % (mask_type, acc))
    
    # for i in range(26):
    #     data = np.load("./pano/%s/%d/diff/f_img/%d.npy" % (mask_type, acc, i))
    #     np.save('./output/selected/selected.npy', data)
    #     test()

    #     data = np.load("./output/result_%s_%d_ours.npy" % (mask_type, acc))

    #     np.save("./pano/%s/%d/diff/res/%d_%s_%s_%d" % (mask_type, acc, i, mask_type, model_type, acc), data)

    testKnee()
    
    # if not os.path.exists('./knee_pano/%s/%d/res' % (mask_type, acc)):
    #     os.makedirs('./knee_pano/%s/%d/res' % (mask_type, acc))
    
    # for i in range(190):
    #     data = np.load("./knee_pano/%s/%d/f_img/%d.npy" % (mask_type, acc, i))
    #     np.save('./output/selected/selected.npy', data)
    #     testKnee()

    #     data = np.load("./output/result_%s_%d_ours.npy" % (mask_type, acc))

    #     np.save("./knee_pano/%s/%d/res/%d_%s_%s_%d" % (mask_type, acc, i, mask_type, model_type, acc), data)


