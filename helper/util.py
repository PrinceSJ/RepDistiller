from __future__ import print_function

import torch
import numpy as np
from skimage import color
from torchvision import datasets
from dataset.cifar100 import CIFAR100Instance


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
                # print(label)
                path = path1 + ' ' + path2
            except:
                print(line)

            paths.append(path)
            labels.append(label)
    return paths, labels




def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    # print(steps)
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print(correct)

        # binary classi, top1
        if len(topk)==1:
            # print('rright way')
            correct_k = correct.view(-1).float().sum(0, keepdim=True)
            res=correct_k.mul_(100.0 / batch_size)

        else:
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        
        return res


if __name__ == '__main__':

    pass
