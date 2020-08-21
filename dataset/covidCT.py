##todo 参考github上处理covid数据库的办法让teacher模型读入
# mdoel archi


import collections
import os
import pprint
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage.segmentation import slic, mark_boundaries
import cv2
from .utils import read_filepaths, read_filepaths2


class COVID_CT_Dataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """
    def __init__(self, args, mode, n_classes=2, dataset_path='./MedData/data', dim=(224, 224)):
        self.mode = mode
        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'normal': 0, 'COVID-19': 1}
        trainfile = os.path.join(dataset_path, 'COVID-CT', 'train_split.txt')
        testfile = os.path.join(dataset_path, 'COVID-CT', 'test_split.txt')

        # newtrainpath, newtrainlabel = read_filepaths2('./MedData/data/SARS-Cov-2/train_split.txt')
        # newtestpath, newtestlabel = read_filepaths2('./MedData/data/SARS-Cov-2/test_split.txt')

        if mode == 'train':
            self.paths, self.labels = read_filepaths(trainfile, self.mode)
            self.paths.extend(self.paths)
            self.labels.extend(self.labels)
            self.paths.extend(self.paths)
            self.labels.extend(self.labels)

            # self.paths.extend(newtrainpath)
            # self.labels.extend(newtrainlabel)

        elif mode == 'test':
            self.paths, self.labels = read_filepaths(testfile, self.mode)
            # self.paths.extend(newtestpath)
            # self.labels.extend(newtestlabel)

        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_tensor, site = self.load_image(self.paths[index])
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor#, site

    def load_image(self, img_path):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))

        # print(img_path.split('/'))
        if img_path.split('/')[3] == 'COVID-CT':
            site = 'ucsd'
        else:
            site = 'new'

        image = Image.open(img_path).convert('RGB')

        inputsize = 224
        transform = {
            'train': transforms.Compose(
                [transforms.Resize(256),
                 transforms.RandomCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 ]),
            'test': transforms.Compose(
                [transforms.Resize([inputsize, inputsize]),
                 ])
        }

        transformtotensor = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        if self.mode == 'train':
            image = transform['train'](image)
        else:
            image = transform['test'](image)

        image_tensor = transformtotensor(image)

        return image_tensor, site


def get_covidCT_dataloaders(args):
    # train_params = {'batch_size': args.batch_size,
    #                 'shuffle': True,
    #                 'num_workers': 3}
    # test_params = {'batch_size': args.batch_size,
    #                'shuffle': False,
    #                'num_workers': 2}

    train_loader = COVID_CT_Dataset(args, mode='train', n_classes=args.classes, dataset_path=args.dataset_path,
                                    dim=(224, 224))
    test_loader = COVID_CT_Dataset(args, mode='test', n_classes=args.classes, dataset_path=args.dataset_path,
                                   dim=(224, 224))

    training_generator = DataLoader(train_loader, 
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=3)

    test_generator = DataLoader(test_loader, 
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2)

    return training_generator, test_generator





# if __name__ == '__main__':
#     covid=COVID_CT_Dataset(args=None, mode='train')
#     covid.load_image(img_path='MedData/data/COVID-CT/train/2%0.jpg')
