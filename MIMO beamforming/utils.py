# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:48:49 2022

@author: Yanzhen
"""
import torch
import math
import numpy as np

def complex_to_real(x:torch.Tensor):
    return torch.stack([x.real,x.imag])
    
def real_to_complex(x:torch.Tensor):
    return x[0] + 1j*x[1]

def normalizeV(V,Pmax):
    batch_size,K,N_t,d = V.size()
    power = torch.linalg.norm(V,dim = (2,3))**2
    sum_power = torch.sum(power,dim=1)+1e-20
    scale_factor = torch.sqrt(Pmax/sum_power).unsqueeze(1).unsqueeze(1).unsqueeze(1)
    scale_factor = scale_factor.repeat(1,K,N_t,d) #This can be deleted
    
    normedV = V*scale_factor
    
    return normedV

def compute_rate(V,H,sigma,Pmax,device=torch.device("cpu"),dtype = torch.complex64, equivalent = False):
    VH = torch.conj(torch.transpose(V,2,3))
    HH = torch.conj(torch.transpose(H, 2, 3))
    
    VkVkH = torch.einsum("abcd,abde->abce",V,VH)
    sum_VkVkH = torch.einsum("abcd->acd",VkVkH)
    
    sum_VkVkH_minus_VkVk = sum_VkVkH.unsqueeze(1) - VkVkH
    
    if equivalent:
        tr_sum_VkVkH = torch.einsum("abb->a",sum_VkVkH).unsqueeze(1).unsqueeze(1)
        div_term = torch.einsum("abcd,abde,abef->abcf",H,sum_VkVkH_minus_VkVk,HH) + (sigma**2/Pmax*tr_sum_VkVkH*torch.eye(H.size(2),dtype = dtype,device=device)).unsqueeze(1)
        
    else:
        div_term = torch.einsum("abcd,abde,abef->abcf",H,sum_VkVkH_minus_VkVk,HH) + sigma**2*torch.eye(H.size(2),dtype = dtype,device=device).unsqueeze(0).unsqueeze(0)
    try:
        div_term_inv = torch.linalg.inv(div_term)
    except Exception as e:
        print(e)
        div_term_inv = torch.linalg.pinv(div_term)
    
    Hk_VkVkH_HkH = torch.einsum("abcd,abde,abef->abcf",H,VkVkH,HH)
    SINR = torch.einsum("abcd,abde->abce",Hk_VkVkH_HkH,div_term_inv)    
    
    I = torch.eye(H.size(2),dtype = dtype,device = device).unsqueeze(0).unsqueeze(0)
    
    sign,rate = torch.linalg.slogdet(I+SINR)
    sum_rate = torch.sum(rate,dim=1)
    
    return sum_rate,rate
    