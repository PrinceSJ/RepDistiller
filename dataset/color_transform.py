from __future__ import print_function

import numpy as np
from skimage import color

class RGB2L(object):
    """Convert RGB PIL image to ndarray Lab and keep only L-dimension"""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)        #(32,32,3)
        img[:,:,1:]=0
        return img

class RGB2ab(object):
    """Convert RGB PIL image to ndarray Lab and keep ab-dimension"""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        img[:,:,0]=0
        return img

