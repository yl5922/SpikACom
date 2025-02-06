import numpy as np
import torch
import utils
import math
import torch.nn as nn
from spikingjelly.activation_based import functional
import os
from wmmse_torch import Wmmse
from tqdm import tqdm
import random
import time
from models import SNNConvNet, model_layer, without_model_layer
import matplotlib.pyplot as plt
import argparse

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

def train_model(mainnet, mask, train_data_loader, test_data_loader, reg_weight, lr, K, Nt, Nr, d, Pmax, sigma, device, model_driven):
    optimizer = torch.optim.Adam(mainnet.parameters(),lr = 1e-4)
    [W_mask, U_mask, V_mask] = mask
    
    for e in range(10):
        mainnet.train()
        total_ewc_loss = 0
        total_src_loss = 0
        total_rate_loss = 0
        num = 0
        
        conv_spk_counter = 0
        input_spk_counter = 0
        hidden_spk_counter = 0
        
        for H in tqdm(train_data_loader):
            H = H.to(device=device, dtype=torch.complex64)
            H_real = utils.complex_to_real(H).permute(1, 0, 2, 3, 4)
            
            embedding_out, spk_rec = mainnet(H_real)
            conv_spk = spk_rec[0]
            input_spk = spk_rec[1]
            hidden_spk = spk_rec[2]
            
            if model_driven:
                V = model_layer(embedding_out, W_mask, U_mask, H, K, d, Nt, Nr, Pmax, sigma)
            else:
                V = without_model_layer(embedding_out, V_mask, K, d, Nt, Pmax)
                
            sum_rate, individual_rate = utils.compute_rate(V, H, sigma, Pmax, device=device,
                                                           equivalent=True)
            average_sum_rate = -torch.mean(sum_rate)
            
            reg_loss = reg_weight * (torch.sum(conv_spk) + torch.sum(input_spk) + torch.sum(hidden_spk))  # L1 loss on total number of spikes
            constrain_loss = torch.norm(embedding_out)*0.05 #original value 0.01
            
            loss = average_sum_rate + reg_loss + constrain_loss
                      
            optimizer.zero_grad()
            loss.real.backward()
            optimizer.step()
            functional.reset_net(mainnet)

            total_rate_loss += average_sum_rate.detach()
            num += 1
            
            conv_spk_counter += torch.mean(conv_spk)
            input_spk_counter += torch.mean(input_spk)
            hidden_spk_counter += torch.mean(hidden_spk)

        conv_intensity = conv_spk_counter.detach()/num
        input_intensity = input_spk_counter.detach()/num
        hidden_intensity = hidden_spk_counter.detach()/num           
        current_test_rate = test_model(mainnet, mask, test_data_loader, K, Nt, Nr, d, Pmax, sigma, device, model_driven)

        if e % 1 ==0:
            print(f"The norm of output is {embedding_out.norm():.2f}")
            print(f'epoch {e}: rate = {total_rate_loss/num:.2f}, test_rate = {current_test_rate:.2f}, conv_firing = {conv_intensity:.2f}, input_firing = {input_intensity:.2f}, hidden_firing = {hidden_intensity:.2f}')
                                  
def test_model(mainnet, mask, test_data_loader, K, Nt, Nr, d, Pmax, sigma, device, model_driven):
    total_rate_loss = 0
    num = 0
    [W_mask, U_mask, V_mask] = mask
    
    for H in test_data_loader:
        H = H.to(device=device, dtype=torch.complex64)
        H_real = utils.complex_to_real(H).permute(1, 0, 2, 3, 4)
        
        embedding_out, spk_rec = mainnet(H_real)
        conv_spk = spk_rec[0]
        input_spk = spk_rec[1]
        hidden_spk = spk_rec[2]
        
        if model_driven:
            V = model_layer(embedding_out, W_mask, U_mask, H, K, d, Nt, Nr, Pmax, sigma)
        else:
            V = without_model_layer(embedding_out, V_mask, K, d, Nt, Pmax)
                
        sum_rate, individual_rate = utils.compute_rate(V, H, sigma, Pmax, device=device,
                                                       equivalent=True)
        average_sum_rate = torch.mean(sum_rate)
                
        functional.reset_net(mainnet)
        total_rate_loss += average_sum_rate.detach()
        num += 1
      
    return total_rate_loss/num
            
def main():
    parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
    parser.add_argument('-T', default=4, type=int, help='Simulation time steps')
    parser.add_argument('-K', default=6, type=int, help='number of users')
    parser.add_argument('-Nt', default=64, type=int, help='number of transmitting antennas')
    parser.add_argument('-Nr', default=4, type=int, help='number of transmitting antennas')
    parser.add_argument('-d', default=4, type=int, help='number of transmitting antennas')
    parser.add_argument('-Pmax', default=1, help='maximum transmitting power')
    parser.add_argument('-sigma', default=0.1, help='noise power')
    parser.add_argument('-num_neurons', default=800, help='number of neurons')
    parser.add_argument('-conv_channel', default=4, help='number of convolutional channels')
    parser.add_argument('-lr', default=1e-3, help='learning rate')
    parser.add_argument('-reg_weight', default=0, help='regulation on spiking rate')
    parser.add_argument('-model_driven', default=True, help='add the model-driven layer')
    parser.add_argument('-batch_size', default=128, help='number of convolutional channels')
    parser.add_argument('-device', default=torch.device('cuda'), help='used device')
    parser.add_argument('-dtype', default=torch.complex64, help='data type')
    
    args = parser.parse_args()
    print(args)
    
    batch_size = args.batch_size; K = args.K; Nt = args.Nt; Nr = args.Nr; d = args.d
    
    input_dim = K * Nt * Nr * 2
    
    if args.model_driven:
        output_dim = K * Nr * d * 2 + K * d * d * 2
    else:
        output_dim = K * Nt* Nr * 2
    
    net = SNNConvNet(input_dim, output_dim, args.num_neurons, args.conv_channel, args.device, args.T)
    net.to(args.device)
    
    # Generate training samples and testing sample
    train_set = torch.randn(20000, K, Nr, Nt).to(device = args.device, dtype = args.dtype)
    test_set = torch.randn(2000, K, Nr, Nt).to(device = args.device, dtype = args.dtype)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,        
        shuffle=True,
        drop_last=True)
    
    test_data_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,        
        shuffle=False)
    
    #  add a mask to avoid numerical problem
    W_mask = torch.randn([1, K, d, Nr], device = args.device, dtype = args.dtype)*1e-2
    U_mask = torch.randn([1, K, d, Nr], device = args.device, dtype = args.dtype)*1e-2
    V_mask = torch.randn([1, K, Nt, d], device = args.device, dtype = args.dtype)*1e-2
    masks = [W_mask, U_mask, V_mask]
    
    train_model(net, masks, train_data_loader, test_data_loader, args.reg_weight, args.lr, K, Nt, Nr, d, args.Pmax, args.sigma, args.device, args.model_driven)
    
    wmmse_agent = Wmmse(K, Nt, Nr, d, args.Pmax, args.sigma)
    _,_,_,rate_list = wmmse_agent.run_batch_wmmse(test_set, max_iteration = 400, device = args.device, dtype = args.dtype)
    rate_list = torch.stack(rate_list)
    print(f'The performance of wmmse is {rate_list.mean():.2f}')
    
if __name__ == '__main__':
        main()
