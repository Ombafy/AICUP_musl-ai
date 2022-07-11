import glob
import numpy as np
# import cv2
import os
from torch.utils import data
import scipy.io as sci
import math
import torch

# code adapted from:
'''
@article{Iwana_2021,
  doi = {10.1371/journal.pone.0254841},
  url = {https://doi.org/10.1371%2Fjournal.pone.0254841},
  year = 2021,
  month = {jul},
  publisher = {Public Library of Science ({PLoS})},
  volume = {16},
  number = {7},
  pages = {e0254841},
  author = {Brian Kenji Iwana and Seiichi Uchida},
  title = {An empirical survey of data augmentation for time series classification with neural networks},
  journal = {{PLOS} {ONE}}
} 
'''
def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

def scaling(x, factor):
    y = np.copy(x)
    y = y * factor
    return y

# def scaling(x, sigma=0.1):
#     # https://arxiv.org/pdf/1706.00527.pdf
#     factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
#     return np.multiply(x, factor[:,np.newaxis,:])

def rotation(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)    
    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def flip(x):
    return np.flip(x).copy()

def magnitude_warp(x, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    orig_steps = np.arange(x.shape[1])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot+2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2],1))*(np.linspace(0, x.shape[1]-1., num=knot+2))).T
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper

    return ret

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device = "cpu"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size())).to(self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Training_Dataset(data.Dataset):
    def __sort__func(self, inp):
        st = os.path.split(inp)
        name = st[1]
        ind = name.find('-')
        sub = name[ind + 1: -4]
        sub = int(sub)
        return sub

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_file = np.load(os.path.join(config.train_file))
        self.ground_truth = np.load(os.path.join(config.gt_train_file))
        self.transforms = ['scale', 'same']
        # print(self.train_file.shape)
        # print(self.ground_truth.shape)
    
    def __getitem__(self, index):
        out = self.train_file[index, :]
        tran = np.random.choice(self.transforms)
        if tran == 'scale':
          out = scaling(out, 2)
        # elif tran == 'scale2':
        #   out = scaling(out, 4)
        else:
          out = out
        # print(out.shape)
        return out, self.ground_truth[index]

    def __len__(self):
        return len(self.train_file)

class Val_Dataset(data.Dataset):
    def __sort__func(self, inp):
        st = os.path.split(inp)
        name = st[1]
        ind = name.find('-')
        sub = name[ind + 1: -4]
        sub = int(sub)
        return sub

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.test_file = np.load(os.path.join(config.val_file))
        self.ground_truth = np.load(os.path.join(config.gt_val_file))
        
    def __getitem__(self, index):
        return self.test_file[index, :], self.ground_truth[index]

    def __len__(self):
        return len(self.test_file)

class Test_Dataset(data.Dataset):
    def __sort__func(self, inp):
        st = os.path.split(inp)
        name = st[1]
        ind = name.find('-')
        sub = name[ind + 1: -4]
        sub = int(sub)
        return sub

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.test_file = np.load(os.path.join(config.test_file))
        
    def __getitem__(self, index):
        return self.test_file[index, :]

    def __len__(self):
        return len(self.test_file)