# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:38:54 2024

@author: yl5922
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

class DVSGFeature(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split

        self.h = h5py.File(root + '/{}_features.hdf5'.format(split), 'r')
        self.img = self.h['data']
        self.label = self.h['label']

    def close(self):
        self.h.close()

    def __getitem__(self, index):
        img = torch.from_numpy(self.img[index])
        label = torch.tensor(self.label[index])
        
        return img, label

    def __len__(self):
        return len(self.label)