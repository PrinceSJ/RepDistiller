from __future__ import print_function

import os
import socket
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# from utils import RGB2L, RGB2ab

"""
mean = {
    
}

std = {
    
}
"""


def test():
    """
    plot med dicom images
    """

    fname='./MedData/Lung-PET-CT-Dx/Lung_Dx-A0164/04-12-2010-PET01PTheadlung Adult-08984/8.000000-Thorax  1.0  B31f-52757/1-001.dcm'    
    
    ds=pydicom.dcmread(fname)
    # print(ds.pixel_array.shape)
    print(ds.pixel_array[1])
    plt.figure(figsize=(10,10))
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
    plt.show()




def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './MedData/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def get_lung_dataloader(batch_size=32, num_workers=8):
    """
    Lung-PET-CT-Dx
    """
    
    




if __name__ =='__main__':
    # test()
