# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:11:14 2024

@author: yl5922
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, layer, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import math

firing_thres = 0.5

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)
    def forward(self, x: torch.Tensor):
        return self.voting(x.unsqueeze(1)).squeeze(1)

class EmbeddingNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(EmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2400)
        self.fc2 = nn.Linear(2400, 2400)
        self.fc_out = nn.Linear(2400, output_dim, bias = False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        out = self.fc_out(x)
        out = self.sigmoid(out*10)
        # out = self.sigmoid(out*100) this provides a more accurate gate but difficult for conv gate to converge
        return out
    
class SemanticEncoder(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(SemanticEncoder.conv3x3(2, channels))
        conv.append(nn.MaxPool2d(2, 2))
        for i in range(4):
            conv.extend(SemanticEncoder.conv3x3(channels, channels))
            conv.append(nn.MaxPool2d(2, 2))
        conv.append(nn.Flatten())
        self.conv = nn.Sequential(*conv)
    
    def forward(self, x:torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        spk_rec = []
        
        for t in range(x.shape[0]):
            source_spk = self.conv(x[t])
            spk_rec.append(source_spk)
        
        spk_rec = torch.stack(spk_rec, dim = 1)
    
        return spk_rec
        
    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        ]        
        
    
class Encoder(nn.Module):
    def __init__(self, channels: int, hidden_num, block_length, use_feature = True, train_sc = True, K = 1):
        super(Encoder, self).__init__()
        self.train_sc = train_sc
        self.use_feature = use_feature
        
        self.sc_encoder = SemanticEncoder(channels)
        self.fc1 = nn.Linear(channels * 4 * 4 // K, hidden_num, bias=False)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=firing_thres, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.fc_out = nn.Linear(hidden_num, block_length*2, bias=False)
        self.lif_out = neuron.LIFNode(tau=2.0, v_threshold=firing_thres, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        
    def forward(self, x: torch.Tensor, gate = None):
        if self.use_feature:
            sc_spk = x
        else:
            if self.train_sc:
                sc_spk = self.sc_encoder(x) #[N, T, L]
            else:
                with torch.no_grad():
                    sc_spk = self.sc_encoder(x) #[N, T, L]
                
        ch_spk_rec = []
        com_spk_rec = []
        
        for t in range(sc_spk.shape[1]):
            spk = self.fc1(sc_spk[:,t])
            spk = self.lif1(spk)
            if gate is not None:
                spk = spk*gate
            ch_spk_rec.append(spk)
            
            spk = self.fc_out(spk)
            spk = self.lif_out(spk)
            com_spk_rec.append(spk)
        
        ch_spk = torch.stack(ch_spk_rec, dim = 1) # [N, T, L]
        com_spk = torch.stack(com_spk_rec, dim = 1) # [N, T, L]
        
        return com_spk, ch_spk, sc_spk

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_num, apply_drop_out = False):
        super(Decoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_num, bias=False)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=firing_thres, surrogate_function=surrogate.ATan(), detach_reset=True)     
        self.fc_out = nn.Linear(hidden_num, 110, bias=False)
        self.lif_out = neuron.LIFNode(tau=2.0, v_threshold=firing_thres, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.vote = VotingLayer(10)
        
        self.apply_drop_out = apply_drop_out
        self.drop1 = layer.Dropout(0.5)
        self.drop2 = layer.Dropout(0.5)
        
    def forward(self, x: torch.Tensor, gate = None):
        # x dimension is [N, T, L]
        out_spikes = 0
        de_spk_rec = []
        
        for t in range(x.size(1)):
            if self.apply_drop_out:
                input = self.drop1(x[:, t])
            else:
                input = x[:, t]
                
            spk = self.fc1(input)
            spk = self.lif1(spk)

            if gate is not None:
                spk = spk*gate
            de_spk_rec.append(spk)
            
            if self.apply_drop_out:
                spk = self.drop2(spk)
            out_spk = self.fc_out(spk)
            out_spk = self.lif_out(out_spk)                      
            out_spikes += self.vote(out_spk)
            
        de_spk = torch.stack(de_spk_rec, dim=1)
        
        return out_spikes / x.shape[1], de_spk

class Decoder_convolution_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder_convolution_layer, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size = 13, padding = 6, bias = False)
        
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=firing_thres, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    def forward(self, x):
        # input dim is [N, T, K*2, M/2+L-1 ]
        [N, T, C, L] = x.size()
        spk_list = []
        
        for t in range(T):
            spk = self.conv1(x[:, t])
            spk = self.lif1(spk)
            
            spk_list.append(spk)
        
        return torch.stack(spk_list, dim=1)
    
    def freeze_decoder_conv(self):
        self.conv1.weight.requires_grad = False
        if self.conv1.bias is not None:
            self.conv1.bias.requires_grad = False
    
        
class MultiPathChannel(nn.Module):
    def __init__(self, PDP, snr, device):
        super(MultiPathChannel, self).__init__()
        self.PDP = PDP
        self.L = len(PDP)
        self.snr = 10**(snr/10)
        self.device = device
        
    def forward(self, x):
        # sample a random channel and conv with the input data
        batch_size = x.size(0)
        h = self.sample_h(batch_size)
        
        x_power = torch.norm(x.detach(), dim = 1)/math.sqrt(x.size(1))
        # conv with the data
        x = x.unsqueeze(0)
        h = h.unsqueeze(1)
        
        # pad the input with zeros of dim L-1
        pad = (self.L-1, self.L-1)
        x = F.pad(x, pad)
        
        output = F.conv1d(x, h, groups = batch_size).squeeze(0)
        
        # add gaussian noise
        noise = 1/math.sqrt(2)*(torch.randn(output.size()) +1j*torch.randn(output.size()))
        noise = noise.to(device = self.device)
        noise = noise/math.sqrt(self.snr)*x_power.unsqueeze(1)
       
        output += noise
        
        return output
    
    def sample_h(self, batch_size):
        h = 1/math.sqrt(2)*(torch.randn(batch_size, self.L) +1j*torch.randn(batch_size, self.L))      
        h = h.to(device = self.device)
        h = h*self.PDP.unsqueeze(0)
        
        return h
    
class MU_InitialNet(nn.Module):
    
    def __init__(self, channels, hidden_num, block_length, K, L, device, FDMA):
        super(MU_InitialNet, self).__init__()
       
        self.block_length = block_length
        self.K = K
        self.L = L
        self.device = device
        self.FDMA = FDMA
        
        self.encoder = nn.ModuleList([Encoder(channels, hidden_num, block_length, use_feature = False, train_sc = True, K = self.K) for k in range(self.K)])
        if self.FDMA:
            self.decoder = Decoder(2*(self.block_length)*self.K, hidden_num, apply_drop_out = True)
        else:
            self.decoder = Decoder(2*(self.block_length), hidden_num, apply_drop_out = True)
    
    def silent_sc_encoder(self):
        for encoder in self.encoder:
            encoder.train_sc = False
        
    def forward(self, x):
        # split x into K strides
        [N, T, P, H, W] = x.size()
        x = x.reshape(N, T, P, self.K, int(H/self.K), W)
        x = x.permute(3,0,1,2,4,5)
        
        spk_rec = []
        # feed x into the encoder
        
        zipped = [self.encoder[k](x[k]) for k in range(self.K)]   
        x = torch.stack([zipped[k][0] for k in range(self.K)])
        ch_spk = torch.stack([zipped[k][1] for k in range(self.K)])
        sc_spk = torch.stack([zipped[k][2] for k in range(self.K)])
        
        spk_rec.append(sc_spk)
        spk_rec.append(ch_spk)
        
        x = sc_spk
        x = x.reshape(self.K, N, T, -1) # [K, N, T, M] 
        y = x.permute(1,2,0,3) #-> [N, T, K, M]
               
        reshaped_y = y.reshape(N, T, -1)
        out, de_spc = self.decoder(reshaped_y)
        spk_rec.append(de_spc)
        
        return out, spk_rec
    
class MU_SCBaseNet(nn.Module):
    
    def __init__(self, channels, hidden_num, block_length, K, L, device, FDMA, use_feature, whether_decoder_conv, conv_out_channels):
        super(MU_SCBaseNet, self).__init__()
        
        self.src_lambda = 0.5
        self.SRC_count = 0
        self.normalize_weight = True # Normalize the average power of the weight matrix
        
        self.block_length = block_length
        self.K = K
        self.L = L
        self.device = device
        self.FDMA = FDMA
        self.use_feature = use_feature
        self.hidden_num = hidden_num
        self.whether_decoder_conv = whether_decoder_conv
        self.conv_out_channels = conv_out_channels
        
        self.encoder = nn.ModuleList([Encoder(channels, hidden_num, block_length, use_feature = use_feature, train_sc = True, K = self.K) for k in range(self.K)])
        if self.FDMA:
            if self.whether_decoder_conv:
                self.decoder_conv = Decoder_convolution_layer(self.K*2, conv_out_channels)
                self.decoder = Decoder(conv_out_channels*(self.block_length + self.L -1), hidden_num)
            else:
                self.decoder = Decoder(2*(self.block_length + self.L -1)*self.K, hidden_num)
        else:
            print('Now only support FDMA')
            self.decoder = Decoder(2*(self.block_length + self.L -1), hidden_num)
        
    def forward(self, x, h, gate, conv_gate, neuron_noise = None):
        # split x into K strides
        if self.use_feature:
            [N, K, T, D] = x.size()
            x = x.permute(1,0,2,3)
        else:
            [N, T, P, H, W] = x.size()
            K = self.K
            x = x.reshape(N, T, P, self.K, int(H/self.K), W)
            x = x.permute(3,0,1,2,4,5)
        
        if neuron_noise is not None:
            self.add_noise_to_neuron(neuron_noise, N)
        
        spk_rec = []
        # feed x into the encoder
        
        zipped = [self.encoder[k](x[k], gate) for k in range(self.K)]   
        x = torch.stack([zipped[k][0] for k in range(self.K)])
        ch_spk = x
        sc_spk = torch.stack([zipped[k][1] for k in range(self.K)])
        
        spk_rec.append(sc_spk)
        spk_rec.append(ch_spk)
        
        x = x.reshape(self.K, N*T, 2, -1) # [K, N, T, M] -> [K, N*T, 2, M/2]
        
        # convert x into complex number and feed into the channel
        x = x[:,:,0] + 1j*x[:,:,1] 
        y = torch.stack([ h[k](x[k])  for k in range(self.K)], dim = 1) # [N*T, K, M/2+L-1]
        if not self.FDMA:
            y = y.sum(1)
               
        # convert y into real numbers
        y = torch.stack([y.real, y.imag], dim=2) # [N*T, K, 2, M/2+L-1]
        if self.whether_decoder_conv:
            y = y.reshape(N, T, -1, y.size(-1))
            y = self.decoder_conv(y) #[N, T, C, M/2+L-1]
            y = y*conv_gate.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            spk_rec.append(y)
        else:
            spk_rec.append(torch.zeros([N,T,self.conv_out_channels,y.size(-1)]).to(device = y.device))
            
        reshaped_y = y.reshape(N, T, -1)
        out, de_spc = self.decoder(reshaped_y, gate)
        spk_rec.append(de_spc)
        
        return out, spk_rec
    
    def spiking_rate_penalty(self):
        if self.SRC_count > 0: 
            losses = []
            for env_idx in range(self.SRC_count):
                """Spiking rate penalty"""
                [en1, en2, de0, de1, de2] = self.all_rate_list[env_idx]
                
                for k in range(self.K):
                    M,N = self.encoder[k].fc1.weight.size()
                    pre_vector = torch.ones(N).to(self.device)
                    SRC_penalty = pre_vector.unsqueeze(0)*en1[k].unsqueeze(1)
                    if self.normalize_weight:
                        SRC_penalty = SRC_penalty/SRC_penalty.norm()
                    mean = getattr(self, 'encoder__{}__fc1__weight_SRC_prev_task'.format(k))
                    loss = (SRC_penalty*(self.encoder[k].fc1.weight- mean)**2).sum()
                    losses.append(loss)
                    
                    SRC_penalty = en1[k].unsqueeze(0)*en2[k].unsqueeze(1)
                    if self.normalize_weight:
                        SRC_penalty = SRC_penalty/SRC_penalty.norm()
                    mean = getattr(self, 'encoder__{}__fc_out__weight_SRC_prev_task'.format(k))
                    loss = (SRC_penalty*(self.encoder[k].fc_out.weight- mean)**2).sum()
                    losses.append(loss)
                
                if not self.whether_decoder_conv:
                    M,N = self.decoder.fc1.weight.size()
                    pre_vector = torch.ones(N).to(self.device)
                    SRC_penalty = pre_vector.unsqueeze(0)*de1.unsqueeze(1)
                    if self.normalize_weight:
                        SRC_penalty = SRC_penalty/SRC_penalty.norm()
                    mean = getattr(self, 'decoder__fc1__weight_SRC_prev_task')
                    loss = (SRC_penalty*(self.decoder.fc1.weight- mean)**2).sum()
                    losses.append(loss)
                else:
                    pre_vector = torch.ones(2*self.K).to(self.device)
                    conv_vector = de0.mean(1)
                    SRC_penalty = pre_vector.unsqueeze(0)*conv_vector.unsqueeze(1)
                    SRC_penalty = SRC_penalty.unsqueeze(-1)
                    if self.normalize_weight:
                        SRC_penalty = SRC_penalty/SRC_penalty.norm()
                    mean = getattr(self, 'decoder_conv__conv1__weight_SRC_prev_task')
                    loss = (SRC_penalty*(self.decoder_conv.conv1.weight- mean)**2).sum()
                    losses.append(loss)
                    
                    pre_vector = de0.reshape(-1)
                    SRC_penalty = pre_vector.unsqueeze(0)*de1.unsqueeze(1)
                    if self.normalize_weight:
                        SRC_penalty = SRC_penalty/SRC_penalty.norm()
                    mean = getattr(self, 'decoder__fc1__weight_SRC_prev_task')
                    loss = (SRC_penalty*(self.decoder.fc1.weight- mean)**2).sum()
                    losses.append(loss)
                
                SRC_penalty = de1.unsqueeze(0)*torch.ones(110).to(self.device).unsqueeze(1)
                if self.normalize_weight:
                    SRC_penalty = SRC_penalty/SRC_penalty.norm()
                mean = getattr(self, 'decoder__fc_out__weight_SRC_prev_task')
                loss = (SRC_penalty*(self.decoder.fc_out.weight- mean)**2).sum()
                losses.append(loss)
                
            return (1. / 2) * sum(losses)
        else:
            return torch.tensor(0., device=self.device)
                
    def register_rate(self, current_rate_list):
        if self.SRC_count == 0:
            self.all_rate_list = []
        self.all_rate_list.append(current_rate_list) 
        # Store new values in the network
        for n, p in self.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.register_buffer('{}_SRC_prev_task'.format(n), p.detach().clone())
        
        self.SRC_count += 1
        
        
                