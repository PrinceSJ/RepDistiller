import numpy as np
from torchvision import datasets
from .cifar100 import CIFAR100Instance

def main():    
    data_folder = './data'

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

if __name__ == '__main__':
    main()