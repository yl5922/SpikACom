import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import loadmat
import math

class ChannelDataSet(Dataset):

    def __init__(self, data_root, split, env, normalize = True):

        data_list = []
        label_list = []
        
        data_file_name = f'snr_{split}_data_winner.mat'
        label_file_name = f'snr_{split}_label_winner.mat'
        
        data_path = os.path.join(data_root, data_file_name)
        label_path = os.path.join(data_root, label_file_name)

        data = loadmat(data_path)
        label = loadmat(label_path)
        
        self.data = data[list(data.keys())[-1]]
        self.label = label[list(label.keys())[-1]]
        
        self.data = torch.tensor(self.data)
        self.label = torch.tensor(self.label)
                
        self.env = env
        self.normalize = True
            
    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, i):
        data = self.data[self.env, i]
        label = self.label[self.env, i]
        
        if self.normalize:
            factor = torch.norm(data)/math.sqrt(data.numel())
            data = data/factor
            label = label/factor
            
        return data, label 

