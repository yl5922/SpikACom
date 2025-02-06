# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:48:17 2024

@author: yl5922
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional
import math
import numpy as np
import time
import os
import argparse
from models import *
from scipy.special import comb
from tqdm import tqdm
from dataset import ChannelDataSet
import matplotlib.pyplot as plt
import utils

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def train_net(net, train_data_loader, test_data_loader, device, dtype, lr, T_max, epochs, out_dir):
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    min_mse = 1e9
    
    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_batches = 0
        
        for frame, label in train_data_loader:
            if isinstance(net, ChannelNet):
                frame = utils.interpolation(frame, 48, 'rbf')
            optimizer.zero_grad()
            frame = frame.to(device = device, dtype = dtype)
            label = label.to(device = device, dtype = dtype)
            y_predict = net(frame)
          
            loss = F.mse_loss(y_predict, label)
            loss.backward()
            optimizer.step()
                
            train_batches += 1
            train_loss += loss.item() 

            functional.reset_net(net)
        
        train_loss /= train_batches
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_batches = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                if isinstance(net, ChannelNet):
                    frame = utils.interpolation(frame, 48, 'rbf')
                frame = frame.to(device = device, dtype = dtype)
                label = label.to(device = device, dtype = dtype)
                y_predict = net(frame)
                
                loss = F.mse_loss(y_predict, label)

                test_batches += 1
                test_loss += loss.item()
                functional.reset_net(net)

        test_loss /= test_batches
        
        save_max = False
        if min_mse > test_loss:
            min_mse = test_loss
            save_max = True

        if save_max:
            torch.save(net.state_dict(), os.path.join(out_dir, 'checkpoint_max.pth'))

        print(f'epoch={epoch}, train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, total_time={time.time() - start_time:.4f}')

def test_SNR(net, test_set, device, dtype, batch_size, num_workers):
    T = test_set.data.size(0)
    
    continuous_loss_rec = []
    for t in range(T):
        test_set.env = t
        test_data_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True)
        
        net.eval()
        test_loss = 0
        test_batches = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                if isinstance(net, ChannelNet):
                    frame = utils.interpolation(frame.numpy(), 48, 'rbf')
                frame = frame.to(device = device, dtype = dtype)
                label = label.to(device = device, dtype = dtype)
                y_predict = net(frame)
                
                loss = F.mse_loss(y_predict, label)

                test_batches += 1
                test_loss += loss.item()
                functional.reset_net(net)

        test_loss /= test_batches
        continuous_loss_rec.append(test_loss)
    
    print(f'The test acc is {continuous_loss_rec}')
    continuous_loss_rec = torch.tensor(continuous_loss_rec)
    return continuous_loss_rec

def train_SNR(net, train_data_set, test_data_set, device, dtype, lr, T_max, epochs, batch_size, num_workers, out_dir):
    T = train_data_set.data.size(0)
    result_rec = []
    for t in range(T):             
        train_data_set.env = t
        test_data_set.env = t
        train_data_loader = DataLoader(
            dataset=train_data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True)
        
        test_data_loader = DataLoader(
            dataset=test_data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True)       
        
        _ = train_net(net, train_data_loader, test_data_loader, device, dtype, lr, T_max, epochs, out_dir)
        print(f'The {t}th environment finishes training')
        result_t = test_SNR(net, test_data_set, device, dtype, batch_size, num_workers)
        
        if isinstance(net, ChannelNet):
            print('Start training the DnCNN')
            net.intermediate = False
            _ =train_net(net, train_data_loader, test_data_loader, device, dtype, lr, T_max, epochs, out_dir)
            result_t = test_SNR(net, test_data_loader_list, device, dtype, batch_size, num_workers)
            net.intermediate = True
        result_rec.append(result_t)
    
    result_rec = torch.stack(result_rec)
    print(result_rec)
    return result_rec

def getZFperformance(test_set, device, dtype, batch_size, num_workers):
    T = test_set.data.size(0)
    loss_rec = []
    for t in range(T):
        test_set.env = t
        test_data_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True)

        total_loss = 0
        total_batch = 0
        for frame, label in test_data_loader:
            frame = utils.interpolation(frame.numpy(), 48, 'rbf')
            frame = frame.to(device = device, dtype = dtype)
            label = label.to(device = device, dtype = dtype)        
            loss = F.mse_loss(frame, label)
            
            total_batch += 1
            total_loss += loss.detach()
        current_ZF_mse = total_loss/total_batch
        loss_rec.append(current_ZF_mse)
    
    print(f'The mse of the ZF+interpolation is {loss_rec}')
          
def main():
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-dtype', default=torch.float32, help='dtype')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', default= './data', type=str, help='root dir of DVS128 Gesture dataset')
    parser.add_argument('-out_dir', default= './checkpoints', type=str, help='root dir for saving logs and checkpoint')
    
    parser.add_argument('-net', default='snnres', help='snn or ann or resnet or snnres or channelnet')
    
    parser.add_argument('-opt', default = 'Adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-reg', default = '0e-10', type=float, help='The regular loss on the sparsity of spikes')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-T_max', default=256, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-hidden_num', default=3200, type=int, help='Number of hidden neurons')
    parser.add_argument('-conv_channel', default=16, type=int, help='Number of decoder conv channel')
    
    args = parser.parse_args()
    print(args)
    
    if args.net == 'snn':
        base_net = SNNNet(args.T)
    elif args.net == 'resnet':
        base_net = ReEsNet()
    elif args.net == 'ann':
        base_net = ANNNet()
    elif args.net == 'snnres':
        base_net = SNNResNet(args.T)
    elif args.net == 'channelnet':
        base_net = ChannelNet()
        
    base_net.to(args.device)
     
    train_set = ChannelDataSet(data_root = args.data_dir, split = 'train', env = 0)
    test_set = ChannelDataSet(data_root = args.data_dir, split = 'test', env = 0)
    
    out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_{args.opt}_lr_{args.lr}_hidden_{args.hidden_num}')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Mkdir {out_dir}.')
    
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
    
    getZFperformance(test_set, args.device, args.dtype, args.b, args.j)
    train_SNR(base_net, train_set, test_set, args.device, args.dtype, args.lr, args.T_max, args.epochs, args.b, args.j, out_dir)

if __name__ == '__main__':
        main()