from libtiff import TIFF
import numpy as np


def tiff_to_read(tiff_image_name):
    tif = TIFF.open(tiff_image_name, mode="r")
    im_stack = list()
    for im in list(tif.iter_images()):
        im_stack.append(im)
    return np.stack(im_stack)