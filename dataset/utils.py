from __future__ import print_function

import numpy as np
from skimage import color
from torchvision import datasets
from cifar100 import CIFAR100Instance


# class RGB2L(object):
#     """Convert RGB PIL image to ndarray Lab and keep only L-dimension"""
#     def __call__(self, img):
#         img = np.asarray(img, np.uint8)
#         img = color.rgb2lab(img)        #(32,32,3)
#         img[:,:,1:]=0
#         return img

# class RGB2ab(object):
#     """Convert RGB PIL image to ndarray Lab and keep ab-dimension"""
#     def __call__(self, img):
#         img = np.asarray(img, np.uint8)
#         img = color.rgb2lab(img)
#         img[:,:,0]=0
#         return img



def m_std():    
    data_folder = './MedData/data'

    train_data = CIFAR100Instance(root=data_folder,
                                # download=True,
                                train=True)
                                # transform=train_transform)
    # use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    # print(x)
    print(x.shape)
    # calculate the mean and std along the (0, 1) axes
    train_mean = np.mean(x, axis=(0, 1))
    train_std = np.std(x, axis=(0, 1))
    # the the mean and std
    print(train_mean, train_std)



def read_filepaths(file, mode):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if '/ c o' in line:
                break
            try:
                subjid, path1, label = line.split(' ')
                path = './MedData/data/COVID-CT/' + mode + '/' + path1
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels

def read_filepaths2(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if '/ c o' in line:
                break
            try:
                subjid, path1, path2, label = line.split(' ')
                path = path1 + ' ' + path2
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels




# if __name__ == '__main__':
    # main()