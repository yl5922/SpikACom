# -*- coding: utf-8 -*-
"""
Created on Wed May 15 10:48:17 2024

@author: yl5922
"""
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import functional, surrogate, layer, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import math
import numpy as np
import time
import os
import argparse
import utils
from models import *
from tqdm import tqdm
import h5py
from dataset import DVSGFeature
import matplotlib.pyplot as plt
import seaborn as sns

_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def compute_EMD_matrix(PDP_set):
    inner_product = PDP_set.mm(PDP_set.t())
    P_square = torch.sum(PDP_set**2, dim=1)
    EMD_matrix = P_square.unsqueeze(0) + P_square.unsqueeze(1) - 2*inner_product
    EMD_matrix = torch.exp(-EMD_matrix)
    
    return EMD_matrix

def compute_EMD_sum_energy(PDP_set):
    sum_variance = PDP_set.pow(2).sum(1)
    eqv_variance = torch.sqrt(sum_variance/2)
    
    EMD_matrix = torch.pow(eqv_variance.unsqueeze(0) - eqv_variance.unsqueeze(1), 2)
    
    EMD_matrix = torch.pow(EMD_matrix, 0.5)
    EMD_matrix = torch.exp(-EMD_matrix*3)
    
    return EMD_matrix
    
def train_hypernet_parallel(hyper_net, train_PDPs, test_PDPs, device, sum_energy_EMD, spasity_penalty = 5e-4, target_support = 512):
    num_env, K, L = train_PDPs.size()

    optimizer = torch.optim.Adam(hyper_net.parameters(),lr = 2e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5)
    
    num_batch = min(256, num_env)
    if sum_energy_EMD:
        EMD_matrix = compute_EMD_sum_energy(train_PDPs.reshape(num_env,-1))
    else:
        EMD_matrix = compute_EMD_matrix(train_PDPs.reshape(num_env,-1))
    mseloss = nn.MSELoss()
    min_mse = 1e8
    for e in range(10):
        total_loss = 0
        total_num = 0
        t1 = time.time()
        
        for i in range(300):
            index = torch.randperm(num_env)[:num_batch]
            PDPs = train_PDPs[index]
            PDPs = PDPs.reshape(PDPs.size(0),-1)
            
            gates = hyper_net(PDPs)
            
            cosine_distance_matrix = utils.compute_cosine(gates)
            selected_EMD = EMD_matrix[index,:][:,index]
            loss_mse = mseloss(cosine_distance_matrix, selected_EMD)  
            
            loss_spasity = utils.compute_sparsity_loss(gates, target_support) 

            loss = 1000*loss_mse + spasity_penalty*loss_spasity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach()
            total_num += 1
            functional.reset_net(hyper_net)
        
        if e%10 == 0:
            test_gate, test_mse = return_ground_truth_gate(hyper_net, test_PDPs, device, sum_energy_EMD)
            
            if test_mse<min_mse:
                torch.save(test_gate, 'MU_gate.pt')
                min_mse = test_mse
            
        t2 = time.time()
        print(f'The {e}th epoch training loss is {total_loss/total_num} time is {t2-t1}')
        lr_scheduler.step()

def return_ground_truth_gate(hyper_net, PDP_set, device, sum_energy_EMD, plot_heat_map = False):
    num_env, K, L = PDP_set.size()
    
    EMD_matrix = torch.zeros([num_env,num_env], device = device)
    support_matrix = torch.zeros([num_env], device = device)
    gate_list = []
    mse_loss = nn.MSELoss()
    for i in range(num_env):
        PDP1 = PDP_set[i]
        PDP1 = PDP1.reshape(1, -1)
        gate1 = hyper_net(PDP1).mean(0).detach()
               
        gate_list.append(gate1)
        
        for j in range(num_env):
            PDP2 = PDP_set[j]         
            PDP2 = PDP2.reshape(1, -1)
    
            # Compute the Earth-mover distance
            combined_PDP = torch.concatenate([PDP1,PDP2],dim=0)
            if sum_energy_EMD:
                EMD = compute_EMD_sum_energy(combined_PDP)
            else:
                EMD = compute_EMD_matrix(combined_PDP)
            
            EMD_matrix[i,j] = EMD[0,1] 
            
    gate_set = torch.stack(gate_list)
    
    gate_set[gate_set<=0.5] = 0
    gate_set[gate_set>0.5] = 1
    gate_cosine_distance = utils.compute_cosine(gate_set)
    
    support_list = torch.sum(gate_set, dim=1)
    test_loss = mse_loss(EMD_matrix, gate_cosine_distance)
    
    print(f'The EMD matrix is')
    print(EMD_matrix)
    print(f'The trained gate cosine distance is')
    print(gate_cosine_distance)
    print(f'The test mse is {test_loss}')
    
    if plot_heat_map:
        fig = plt.figure()
        colormap = sns.color_palette("Blues", 100) 
        labels = [i+1 for i in range(8)]
        sns.heatmap(EMD_matrix.cpu().numpy(), cmap = colormap, annot = True, fmt = '.2f', xticklabels=labels, yticklabels=labels)
        plt.savefig('EMD.eps', format = 'eps', dpi = 300)
        fig = plt.figure()
        sns.heatmap(gate_cosine_distance.cpu().numpy(), cmap = colormap, annot = True, fmt = '.2f', xticklabels=labels, yticklabels=labels)
        plt.savefig('trained_cosine_distance.eps', format = 'eps', dpi = 300)
        plt.show()
        
    print(f'The number of support is {support_list}')
    
    return gate_set,test_loss


def train_continuous(net, gate_set, conv_gate_set, PDP_set, train_data_loader, test_data_loader, SRC, snr, lr, T_max, reg, epochs, device, out_dir):
    num_env = len(gate_set)
    K = PDP_set.size(1)
        
    acc_recorder = []
    train_acc_recorder = []
    test_acc_recorder = []
    max_acc_recorder = []
    spks_recorder = []
    
    for t in range(num_env):
        gate = gate_set[t]
        conv_gate = conv_gate_set[t]        
        PDP = PDP_set[t]
        
        t_start = time.time()
        net.train()
        train_curve, test_curve, max_test_acc, srv_set, rate_set = train_one_env(t, net, gate, conv_gate, PDP, train_data_loader, test_data_loader, SRC, snr, lr, T_max, reg, epochs, device, out_dir)
        t_end = time.time()
        print(f'The used time for {t}th environment is {t_end-t_start}')
            
        if SRC == True:
            for sr in srv_set:
                sr.data = sr.data**2 
            net.register_rate(srv_set)
        
        current_acc = test_continuous(net, gate_set, PDP_set, test_data_loader, device, snr, lr, T_max, reg, epochs, out_dir, conv_gate_set)
        acc_recorder.append(current_acc)
        train_acc_recorder.append(train_curve)
        test_acc_recorder.append(test_curve)
        max_acc_recorder.append(max_test_acc)
        spks_recorder.append(rate_set)
        
        result = {'acc': acc_recorder,
                  'train_curve': train_acc_recorder,
                  'test_curve': test_acc_recorder,
                  'max_acc': max_acc_recorder,
                  'spks': spks_recorder,
                  'gate': gate_set,
                  'conv_gate': conv_gate_set
                  }
        
        if t>0:
            plot_continuous_curve(acc_recorder, t+1)
            print('The acc_matrix is')
            tmp_recorder = torch.tril(torch.stack(acc_recorder))[:,:t+1]
            print(tmp_recorder)
            tmp_recorder = torch.diag(tmp_recorder).unsqueeze(0) - tmp_recorder
            print(f'The average dropped accracy is {torch.sum(torch.tril(tmp_recorder))/(t*(t+1)/2)}')
    
    return result

def train_one_env(t, net, gate, conv_gate, PDP, train_data_loader, test_data_loader, SRC, snr, lr, T_max, reg, epochs, device, out_dir):
    K = PDP.size(0)
    channel = [MultiPathChannel(PDP[k], snr, device) for k in range(K)]
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max)
    
    max_test_acc = 0       
    train_curve = []
    test_curve = []
    
    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        in_spk_counter = 0
        sc_spk_counter = 0
        ch_spk_counter = 0
        cv_spk_counter = 0
        de_spk_counter = 0
        ot_spk_counter = 0
             
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.float().to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 11).float()

            out_fr, spk_rec = net(frame, channel, gate, conv_gate)
                
            [sc_spk, ch_spk, cv_spk, de_spk] = spk_rec
                               
            reg_loss = torch.sum(sc_spk) + torch.sum(ch_spk) + torch.sum(cv_spk) + torch.sum(de_spk) # L1 loss on total number of spikes
            reg_loss = reg_loss*reg
            
            loss = F.mse_loss(out_fr, label_onehot) + reg_loss        
            if SRC == True:
                src_loss = net.spiking_rate_penalty() * net.src_lambda
                loss += src_loss
                   
            loss.backward()
            optimizer.step()
            
            in_spk_counter += frame.detach().sum(0).mean(1)
            sc_spk_counter += sc_spk.detach().sum(1).mean(1)
            ch_spk_counter += ch_spk.detach().sum(1).mean(1)
            cv_spk_counter += cv_spk.detach().sum(0).mean(1)
            de_spk_counter += de_spk.detach().sum(0).mean(0)
            ot_spk_counter += out_fr.detach().sum(0)

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
                          
            functional.reset_net(net)
            
        in_srv = in_spk_counter/train_samples
        sc_srv = sc_spk_counter/train_samples
        ch_srv = ch_spk_counter/train_samples
        cv_srv = cv_spk_counter/train_samples
        de_srv = de_spk_counter/train_samples
        ot_srv = ot_spk_counter/train_samples
        
        in_rate = in_srv.mean()
        sc_rate = sc_srv.mean()
        ch_rate = ch_srv.mean()
        cv_rate = cv_srv.mean()
        de_rate = de_srv.mean()
        
        'Add some none-zeros elements to mitigate the effect of dead neurons'
        srv_set = [sc_srv + 0.1*gate.unsqueeze(0), ch_srv + 0.1, cv_srv, de_srv + 0.1*gate, ot_srv]
        rate_set = [in_rate, ]    
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            input_spc_rec = []
            hidden_spc_rec = []
            for frame, label in test_data_loader:
                frame = frame.float().to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr, _ = net(frame, channel, gate, conv_gate)
                
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples

        if test_acc > max_test_acc:
            max_test_acc = test_acc

        print(f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}, in_rate = {in_rate:.4f}, sc_rate = {sc_rate:.4f}, ch_rate = {ch_rate:.4f}, cv_rate = {cv_rate:.4f}, de_rate = {de_rate:.4f}, total_time={time.time() - start_time:.4f}')
        train_curve.append(train_acc)
        test_curve.append(test_acc)
        
    return train_curve, test_curve, max_test_acc, srv_set, rate_set

def test_continuous(net, gate_set, PDP_set, test_data_loader, device, snr, lr, T_max, reg, epochs, out_dir, conv_gate_set):
    num_env = len(gate_set)
    acc_recorder = []
    t0 = time.time()
    K = PDP_set.size(1)
    for t in range(num_env):
        gate = gate_set[t]
        conv_gate = conv_gate_set[t]
        PDP = PDP_set[t]
        channel = [MultiPathChannel(PDP[k], snr, device) for k in range(K)]
          
        net.eval()
        test_acc = 0
        test_samples = 0
        test_loss = 0
        with torch.no_grad():
            for rpt in range(10): 
                for frame, label in test_data_loader:
                    frame = frame.float().to(device)
                    label = label.to(device)
                    label_onehot = F.one_hot(label, 11).float()
                    out_fr, spk_rec = net(frame, channel, gate, conv_gate)
                    
                    loss = F.mse_loss(out_fr, label_onehot)
    
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)

            test_loss /= test_samples
            test_acc /= test_samples
            
            acc_recorder.append(test_acc)
    
    t1 = time.time()
    acc_recorder = torch.tensor(acc_recorder)
    print(f'The test performance is {acc_recorder}, the time is {t1 - t0}')
    
    return acc_recorder
        
def gen_orthogonal_gate(num_neurons, T, device):
    gate = torch.zeros(T, num_neurons).to(device)
    num_block = num_neurons // T
    for t in range(T):
        gate[t, t*num_block:(t+1)*num_block] = 1
    
    return gate

def plot_continuous_curve(acc_recorder, T):
    # Plot the sum rate of the continuously learned environment
    rate_matrix = torch.stack(acc_recorder).detach().cpu().numpy().T
    
    for t in range(T):
        x_axis = np.arange(t, T)
        y_axis = rate_matrix[t, t:T]
        plt.plot(x_axis, y_axis, label = f'env {t}')
        
    plt.xlabel('Task index')
    plt.ylabel('Sum rate (bps/Hz)')
    plt.xticks([i for i in range(T)])
    plt.xlim(0,T-1)
    plt.ylim(0.10,0.95)
    
    plt.legend()
    plt.grid(True)
    plt.show()

def train_initial_net(net, train_data_loader, test_data_loader, device, lr, T_max, reg, epochs, out_dir):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    max_test_acc = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        sc_spk_counter = 0
        ch_spk_counter = 0
        de_spk_counter = 0
        
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.float().to(device)
            label = label.to(device)
            label_onehot = F.one_hot(label, 11).float()

            out_fr, spk_rec = net(frame)
            [sc_spk, ch_spk, de_spk] = spk_rec
            
            reg_loss = torch.sum(sc_spk) + torch.sum(ch_spk) + torch.sum(de_spk) # L1 loss on total number of spikes
            reg_loss += torch.mean(torch.sum(torch.sum(sc_spk,dim=0),dim=0)**2) + torch.mean(torch.sum(torch.sum(ch_spk,dim=0),dim=0)**2) + torch.mean(torch.sum(torch.sum(de_spk,dim=0),dim=0)**2)
            reg_loss = reg_loss*reg
            
            loss = F.mse_loss(out_fr, label_onehot) + reg_loss
            loss.backward()
            optimizer.step()
                
            sc_spk_counter += torch.sum(sc_spk.detach())
            ch_spk_counter += torch.sum(ch_spk.detach())
            de_spk_counter += torch.sum(de_spk.detach())

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
        
        sc_rate = sc_spk_counter/train_samples/sc_spk.numel()*sc_spk.size(0)
        ch_rate = ch_spk_counter/train_samples/ch_spk.numel()*ch_spk.size(0)
        de_rate = de_spk_counter/train_samples/de_spk.numel()*de_spk.size(0)
        
        train_loss /= train_samples
        train_acc /= train_samples

        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            input_spc_rec = []
            hidden_spc_rec = []
            for frame, label in test_data_loader:
                frame = frame.float().to(device)
                label = label.to(device)
                label_onehot = F.one_hot(label, 11).float()
                out_fr, _ = net(frame)
                
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        if save_max:
            torch.save(net.state_dict(), os.path.join(out_dir, 'Initial_max.pth'))

        print(f'epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}, sc_rate = {sc_rate:.4f}, ch_rate = {ch_rate:.4f}, de_rate = {de_rate:.4f}, total_time={time.time() - start_time:.2f}')
    
def save_spk_feature(Initial_net, train_data_loader, test_data_loader, device, batch_size, T, channels, K, save_dir):
    #save the features of the Initial net to an h5py file to accelerate training
    Initial_net.eval()
    
    size = len(train_data_loader)
    f = h5py.File(save_dir + '/train_features.hdf5', 'w', libver='latest')
    dset = f.create_dataset('data', (size * batch_size, K, T, channels * 4 * 4 // K),
                    dtype='i8')
    lbset = f.create_dataset('label', (size * batch_size),
                    dtype='i8')
    with torch.no_grad():
        start = 0
        for frame, label in train_data_loader:
            frame = frame.float().to(device)
            num = label.numel()
    
            out_fr, spk_rec = Initial_net(frame)
            [sc_spk, ch_spk, de_spk] = spk_rec
            sc_spk = sc_spk.permute(1,0,2,3)
            sc_spk = sc_spk.detach().cpu().numpy()
            
            dset[start:start + num] = sc_spk
            lbset[start:start + num] = label
            start = start + num
            
            functional.reset_net(Initial_net)
    f.close()
    
    f = h5py.File(save_dir + '/test_features.hdf5', 'w', libver='latest')
    dset = f.create_dataset('data', (288, K, T, channels * 4 * 4 // K),
                    dtype='i8')
    lbset = f.create_dataset('label', (288),
                    dtype='i8')
    with torch.no_grad():
        start = 0
        for frame, label in test_data_loader:
            frame = frame.float().to(device)
            num = label.numel()
            
            out_fr, spk_rec = Initial_net(frame)
            [sc_spk, ch_spk, de_spk] = spk_rec
            sc_spk = sc_spk.permute(1,0,2,3)
            sc_spk = sc_spk.detach().cpu().numpy()
            
            dset[start:start + num] = sc_spk
            lbset[start:start + num] = label
            start = start + num
            
            functional.reset_net(Initial_net)
    f.close()
            
def main():
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-dtype', default=torch.float32, help='dtype')
    parser.add_argument('-b', default=20, type=int, help='batch size')
    parser.add_argument('-epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-channels', default=64, type=int, help='channels of Conv2d in SNN')
    parser.add_argument('-data_dir', default= './data', type=str, help='root dir of DVS128 Gesture dataset')
    parser.add_argument('-out_dir', default= './sc_checkpoints', type=str, help='root dir for saving logs and checkpoint')
    
    parser.add_argument('-opt', default = 'Adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-reg', default = '0e-10', type=float, help='The regular loss on the sparsity of spikes')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-T_max', default=32, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-num_train_env', default=64, type=int, help='Number of training environments')
    parser.add_argument('-num_test_env', default=8, type=int, help='Number of testing environments')
    parser.add_argument('-hidden_num', default=3200, type=int, help='Number of hidden neurons')
    parser.add_argument('-support_num', default=600, type=int, help='Number of support for the gate')
    parser.add_argument('-decoder_channel', default=16, type=int, help='Number of decoder conv channel')
    parser.add_argument('-block_length', default=256, type=int, help='Block length')
    parser.add_argument('-L', default=8, type=int, help='Number of Channel Taps')
    parser.add_argument('-snr', default=0, help='SNR')
    parser.add_argument('-K', default=2, help='Number of users')
    parser.add_argument('-FDMA', default=True, help='using different carriers or not')
    parser.add_argument('-sum_energy_EMD', default=True, help='using sum energy for computing EMD')    
    parser.add_argument('-use_feature', default=True, help='using the conv feature for training or the original DVS-Gesture data')
    parser.add_argument('-feature_dir', default='./saved_features', help='saving directory of features')
    parser.add_argument('-SRC', default=True, help='Spiking rate consolidation')
    parser.add_argument('-square_sr', default=True, help='square the spiking rate')
    parser.add_argument('-gate_type', default='all_one', help='gate type options: trained, all_one, othogonal, leaky')
    parser.add_argument('-whether_decoder_conv', default = False, help='add a convolutional layer before the decoder to reduce complexity')
    
    args = parser.parse_args()
    print(args)
    
    PDP = torch.ones(args.L).to(device = args.device, dtype = args.dtype)
    channel = MultiPathChannel(PDP, args.snr, args.device)
    hyper_net = EmbeddingNet(args.L*args.K, args.hidden_num)
    hyper_net.to(args.device)
    base_net = MU_SCBaseNet(args.channels, args.hidden_num, args.block_length, args.K, args.L, args.device, args.FDMA, args.use_feature, args.whether_decoder_conv, args.decoder_channel)
    base_net.to(args.device)
    
    train_set = DVS128Gesture(args.data_dir, train=True, data_type='frame', split_by='number', frames_number=args.T)
    test_set = DVS128Gesture(args.data_dir, train=False, data_type='frame', split_by='number', frames_number=args.T)
    
    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.j,
        drop_last=True,
        pin_memory=True)
    
    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.j,
        drop_last=False,
        pin_memory=True)
    
    if not os.path.exists(args.feature_dir):
        os.mkdir(args.feature_dir)
        print(f'Mkdir {args.feature_dir}.')
        initial_net = MU_InitialNet(args.channels, args.support_num, args.block_length, args.K, args.L, args.device, args.FDMA)
        initial_net.to(device = args.device)
        if os.path.exists('Initial_max.pth'):
            print('Exsits trained initial model')
            initial_net.load_state_dict(torch.load('Initial_max.pth'))
        else:
            initial_model_dir = './'
            print('Start training the initial model for extracting the semantic features.')
            train_initial_net(initial_net, train_data_loader, test_data_loader, args.device, 1e-3, 60, args.reg, 120, initial_model_dir)
        print('start processing the raw DVSGesture data and save the features to a feature directory')
        save_spk_feature(initial_net, train_data_loader, test_data_loader, args.device, args.b, args.T, args.channels, args.K, args.feature_dir)

    if args.use_feature:
        train_set = DVSGFeature(root = args.feature_dir, split='train')
        train_data_loader = DataLoader(
            dataset=train_set,
            batch_size=args.b,
            shuffle=True,
            num_workers=args.j,
            drop_last=True,
            pin_memory=True)
        
        test_set = DVSGFeature(root = args.feature_dir, split='test')
        test_data_loader = DataLoader(
            dataset=test_set,
            batch_size=args.b,
            shuffle=False,
            num_workers=args.j,
            drop_last=False,
            pin_memory=True)
    
    out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_c_{args.channels}_{args.opt}_lr_{args.lr}_hidden_{args.hidden_num}_reg_{args.reg}')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Mkdir {out_dir}.')
    
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        
    train_PDPs = torch.rand(args.num_train_env, args.K, args.L, device = args.device)
    train_mask = torch.rand(args.num_train_env, 1, 1, device = args.device) + 0.01
    train_PDPs = train_PDPs*train_mask

    # 4 dB SNR interval
    test_PDPs = torch.rand(args.num_test_env, args.K, args.L, device = args.device)
    step_mask = torch.arange(args.num_test_env, device = args.device)*2
    step_mask = torch.pow(10, -step_mask/10).flip(dims=[0])
    PDPs_energy = test_PDPs.pow(2).sum(-1).sqrt()
    step_mask = step_mask.unsqueeze(-1)/PDPs_energy
        
    test_PDPs = test_PDPs*step_mask.unsqueeze(-1)
    test_PDPs = test_PDPs.flip(dims=[0])
        
    train_hypernet_parallel(hyper_net, train_PDPs, test_PDPs, args.device, args.sum_energy_EMD, target_support=args.support_num)
    gate,_ = return_ground_truth_gate(hyper_net, test_PDPs, args.device, args.sum_energy_EMD, plot_heat_map=True)

    if args.whether_decoder_conv:
        conv_hyper_net = EmbeddingNet(args.L*args.K, args.decoder_channel)
        conv_hyper_net.to(args.device)
        train_hypernet_parallel(conv_hyper_net, train_PDPs, test_PDPs, args.device, args.sum_energy_EMD, spasity_penalty=1e1, target_support=4)
        conv_gate,_ = return_ground_truth_gate(conv_hyper_net, test_PDPs, args.device, args.sum_energy_EMD, plot_heat_map=True)
    else:
        conv_gate = torch.zeros(test_PDPs.size(0),args.decoder_channel).to(args.device)
    
    if args.gate_type == 'trained':
        gate_set = gate
        conv_gate_set = conv_gate
    elif args.gate_type == 'all_one':
        gate_set = torch.ones(test_PDPs.size(0),args.hidden_num).to(args.device)
        conv_gate_set = torch.ones(test_PDPs.size(0),args.decoder_channel).to(args.device)
    elif args.gate_type == 'othogonal':
        gate_set = gen_orthogonal_gate(args.hidden_num, 8, args.device)
        conv_gate_set = gen_orthogonal_gate(args.decoder_channel, 8, args.device)
    elif args.gate_type == 'leaky':
        gate_set = (gate + 0.2*torch.ones_like(gate))/1.2
        conv_gate_set = conv_gate+ 0.2*torch.ones_like(conv_gate)
    else:
        raise NotImplementedError('This gate type is not implemented')

    acc_recorder = train_continuous(base_net, gate_set, conv_gate_set, test_PDPs, train_data_loader, test_data_loader, args.SRC, args.snr, args.lr, args.T_max, args.reg, args.epochs, args.device, out_dir)
    torch.save(acc_recorder, f'accuracy_list_SRC_{base_net.src_lambda}_{args.SRC}_gate_{args.gate_type}_hidden_num_{args.hidden_num}_conv_{args.whether_decoder_conv}')

    if args.use_feature:
        train_set.close()
        test_set.close()
        
if __name__ == '__main__':
        main()