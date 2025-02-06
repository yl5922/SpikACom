# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:47:16 2024

@author: yl5922
"""

import torch
import utils
import math
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, layer, functional
import torch.nn.functional as F
from wmmse_torch import compute_B

class SNNConvNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons, conv_channels, device = torch.device('cuda'), time_step = 4):
        super(SNNConvNet, self).__init__()
        self.time_step = time_step
        self.lif1 = neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
        self.lif2 = neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(),
                                          detach_reset=True)

        self.fc1 = nn.Linear(input_dim*(conv_channels//2), num_neurons, bias = False)
        self.fc2 = nn.Linear(num_neurons, num_neurons, bias = False)
        self.conv = nn.Conv2d(2, conv_channels, kernel_size = 3, padding = 1, bias = False)
        self.lifconv = neuron.LIFNode(tau=2.0, decay_input=False, v_threshold=1.0, surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
        self.fc_out = nn.Linear(num_neurons, output_dim, bias = False)
        
    def forward(self, input):
        input_spk_rec = []
        hidden_spk_rec = []

        input = input.reshape(input.size(0), input.size(1), -1, input.size(4))
        conv_input = self.conv(input)
            
        input_list = []
        for t in range(self.time_step):
            convsnn_input = self.lifconv(conv_input)
            input_list.append(convsnn_input)
        
        out_list = []
        for t in range(self.time_step):
            input_spk = input_list[t].reshape(convsnn_input.size(0), -1)
            
            input_spk = self.fc1(input_spk)
            input_spk = self.lif1(input_spk)
            
            x = self.fc2(input_spk)
            x = self.lif2(x)
            hidden_spk = x
            
            out_list.append(hidden_spk)
            
            input_spk_rec.append(input_spk)
            hidden_spk_rec.append(hidden_spk)
        
        conv_spk_rec = torch.stack(input_list, dim = 1)
        input_spk_rec = torch.stack(input_spk_rec,dim=1)
        hidden_spk_rec = torch.stack(hidden_spk_rec,dim=1)
            
        stacked_out = torch.stack(out_list)
        out = self.fc_out(stacked_out.mean(0))
        
        spk_rec = [conv_spk_rec, input_spk_rec, hidden_spk_rec]
        
        return out,spk_rec
    
def model_layer(embedding_out, W_mask, U_mask, H, K, d, Nt, Nr, Pmax, sigma):        
    batch_size = embedding_out.size(0)
    W_out_size = 2*K*d*d
    W_out = embedding_out[:, :W_out_size]
    U_out = embedding_out[:, W_out_size:]
    W = W_out.reshape(batch_size, 2, K, d, d)
    U = U_out.reshape(batch_size, 2, K, d, Nr)
    
    W = utils.real_to_complex(W.permute(1, 0, 2, 3, 4))
    U = utils.real_to_complex(U.permute(1, 0, 2, 3, 4))

    # add with W_mask and U_mask to avoid all zero matrix
    W = W + W_mask
    U = U + U_mask

    B = compute_B(U, H, W, Pmax, sigma, device = W.device, dtype = W.dtype)
    B_inv = torch.linalg.inv(B)
    
    HH = torch.conj(torch.transpose(H, 2, 3))
    V = torch.einsum("acd,abde,abef,abfg->abcg", B_inv, HH, U, W)
    V = utils.normalizeV(V, Pmax)
    
    return V

def without_model_layer(embedding_out, V_mask, K, d, Nt, Pmax):        
    V = embedding_out.reshape(embedding_out.size(0), 2, K, Nt, d)
    V = utils.real_to_complex(V.permute(1, 0, 2, 3, 4))
    V += V_mask
    utils.normalizeV(V, Pmax)
    
    return V
