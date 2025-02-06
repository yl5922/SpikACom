# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:56:05 2024

@author: yl5922
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional
        
class ANNNet(nn.Module):
    def __init__(self):
        super(ANNNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16*2, 3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768*2, 2016)
        self.fc2 = nn.Linear(2016, 2016)
    
    def forward(self, input):
        conved_x = self.conv1(input)
        x = self.relu(conved_x)
        x = x.reshape(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        out = self.fc2(x)
        
        out = out.reshape(-1, 2, 72, 14)
        
        return out

class resBlock(nn.Module):
    def __init__(self, channels):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, input):
        
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        
        out = x + input

        return out
    
class ReEsNet(nn.Module):
    def __init__(self, channels = 16):
        super(ReEsNet, self).__init__()
        self.conv1 = nn.Conv2d(2, channels, 3, padding=1)
        self.relu = nn.ReLU()
        self.resBlockList = nn.ModuleList([resBlock(channels) for i in range(4)])
        self.conv2 = nn.Conv2d(channels, channels, 3, padding = 1)
        self.conv3 = nn.Conv2d(channels, 2, 3, padding = 1)
        
        self.tpconv = nn.ConvTranspose2d(channels, channels, kernel_size=(6, 7), stride=(6, 3), padding=(0, 1))
              
    def forward(self, input):
        conved_x = self.conv1(input)
        x = conved_x
        for resblock in self.resBlockList:
            x = resblock(x)
        x = self.conv2(x)
        x = x + conved_x
        
        x = self.tpconv(x)
        out = self.conv3(x)

        return out
    
class SNNNet(nn.Module):
    def __init__(self, time_step):
        super(SNNNet, self).__init__()
        self.time_step = time_step
        
        self.conv1 = nn.Conv2d(2, 16*2, 3, padding=1)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)

        self.fc1 = nn.Linear(768*2, 2016)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.fc2 = nn.Linear(2016, 2016)
        self.lif3 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.fc3 = nn.Linear(2016, 2016)
    
    def forward(self, x):
        conved_x = self.conv1(x)
        spk_rec = []
        
        for t in range(self.time_step):
            x = self.lif1(conved_x)
            x = x.reshape(x.size(0), -1)
            x = self.fc1(x)
            x = self.lif2(x)
            x = self.fc2(x)
            x = self.lif3(x)
            
            spk_rec.append(x)
        
        spks = torch.stack(spk_rec).mean(0)
        out = self.fc3(spks)
        
        out = out.reshape(-1, 2, 72, 14)
        
        return out

class SNNresBlock(nn.Module):
    def __init__(self, channels):
        super(SNNresBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias = False)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias = False)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
    
    def forward(self, input):
        
        x = self.lif1(input)
        x = self.conv1(x)
        x = self.lif2(x)
        x = self.conv2(x)
        out = x + input
        
        return out
    
class SNNResNet(nn.Module):
    def __init__(self, time_step, channels = 16):
        super(SNNResNet, self).__init__()
        self.time_step = time_step

        self.conv1 = nn.Conv2d(2, channels, 3, padding=1, bias = False)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.resBlockList = nn.ModuleList([SNNresBlock(channels) for i in range(4)]) # orignal 4 block
        
        self.fc1 = nn.Linear(channels*12*4, 2016, bias = False)
        
        self.res_out = nn.Linear(2*12*4, 2*72*14, bias = False)
        
        self.snn_res1 = nn.Linear(2*12*4, 2*16*14, bias = False)
        self.snn_res2 = nn.Linear(2*16*14, 2*72*14, bias = False)
        self.lifres = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), detach_reset=True)
  
        self.drop = nn.Dropout(0.70)
        
    def forward(self, input):
        conved_x = self.conv1(input)
        conv_spk_rec = []
        fc_rec = []
        
        spk_rec = []
        for t in range(self.time_step):
            x = conved_x
            conv_spk = []
            for resblock in self.resBlockList:
                x = resblock(x)      
                conv_spk.append(x)
            conv_spk = torch.stack(conv_spk)
            conv_spk_rec.append(conv_spk)
            
            x = x + conved_x
            x = self.lif1(x)
            
            x = x.reshape(x.size(0),-1)
            fc_rec.append(x)
            
            x = self.drop(x)             
            x = self.fc1(x)
                               
            spk_rec.append(x)
        out = torch.stack(spk_rec).mean(0)
        
        res_out = self.res_out(input.reshape(input.size(0),-1))
        out += res_out
        
        out = out.reshape(out.size(0),2,72,14)
        conv_spk_rec = torch.stack(conv_spk_rec)
        fc_rec = torch.stack(fc_rec)
        
        return out

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 2, kernel_size = 5, padding = 2)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        out = self.conv3(x)
        
        return out

class DnCNN_conv_block(nn.Module):
    def __init__(self):
        super(DnCNN_conv_block, self).__init__()  
        self.conv = nn.Conv2d(64, 64, kernel_size=3, padding = 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        out = self.relu(x)
        
        return out

class DnCNN(nn.Module):
    def __init__(self):
        super(DnCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding = 1)
        self.conv_list = nn.ModuleList([DnCNN_conv_block() for i in range(18)])
        self.conv_out = nn.Conv2d(64, 2, kernel_size=3, padding = 1)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        
        for conv_layer in self.conv_list:
            x = conv_layer(x)
        
        out = self.conv_out(x)
        
        return input - out

class ChannelNet(nn.Module):
    def __init__(self, intermediate = True):
        super(ChannelNet, self).__init__()
        self.srcnn = SRCNN()
        self.dncnn = DnCNN()
        self.intermediate = intermediate
    
    def forward(self, input):
        if self.intermediate:
            out = self.srcnn(input)
        
        else:
            with torch.no_grad():
                x = self.srcnn(input)
            out = self.dncnn(x)
        
        return out
    
    
        
        

