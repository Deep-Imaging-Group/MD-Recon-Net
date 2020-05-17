# import psutil
import torch.nn as nn
from utils import mymath

# def is_enough_memory(size, threshold):
#     '''
#         size: int 表示所需内存(以G为单位)
#         threshold: float 表示剩余内存比size要多threshold G 的内存
#     '''
#     info = psutil.virtual_memory()
#     total = info.total / 1e9
#     percent = 1 - info.percent / 100
#     remains = total * percent
#     print(remains)
#     return (remains - size) >= threshold


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

def DC(rec, u_k, mask, w, is_img=False):
    if is_img:
        rec = mymath.torch_fft2c(rec)
    result = mask * (rec * w / (1 + w) + u_k * 1 / (w + 1)) # weighted the undersampling and reconstruction
    result = result + (1 - mask) * rec # non-sampling point

    if is_img:
        result = mymath.torch_ifft2c(result)
    
    return result

def Sequential(cnn, norm, ac, bn=True):
    if bn:
        return nn.Sequential(cnn, norm, ac)
    else:
        return nn.Sequential(cnn, ac)




