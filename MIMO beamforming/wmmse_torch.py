# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:35:29 2022

@author: Yanzhen
"""

import numpy as np
import torch
import utils
import math

class Wmmse(object):
    def __init__(self, K, Nt, Nr, d, Pmax, sigma):
        super(Wmmse,self).__init__()
        self.K = K
        self.Nt = Nt
        self.Nr = Nr
        self.d = d
        self.Pmax = Pmax
        self.sigma = sigma
        
    def run_batch_wmmse(self, H, max_iteration = 60, init = None, device = torch.device('cuda'), dtype = torch.complex128):
        H = H.to(device = device, dtype = dtype)
        batch_size = H.size(0)
        rec = []
        if init is None:
            V = torch.randn((batch_size, self.K, self.Nt, self.d), device = device, dtype = dtype)
            V = utils.normalizeV(V, self.Pmax)
            W = torch.randn((batch_size, self.K, self.d, self.d), device = device, dtype = dtype)
            U = torch.randn((batch_size, self.K, self.Nr, self.d), device = device, dtype = dtype)
        else:
            V = init[0]
            W = init[1]
            U = init[2]
               
        for i in range(max_iteration):
            sum_rate,individual_rate = utils.compute_rate(V, H, self.sigma, self.Pmax, device = device, dtype = dtype, equivalent=True)
            rec.append(sum_rate)

            A = compute_A(V, H, self.Pmax, self.sigma, device, dtype)
            A_inv = torch.linalg.inv(A)            
            U = torch.einsum("abcd,abde,abef->abcf", A_inv, H, V)       
            
            UH = torch.conj(torch.transpose(U, 2, 3))
            prod_UHV = torch.einsum("abcd,abde,abef->abcf", UH, H, V)
            I = torch.eye(self.d, device = device, dtype = dtype).unsqueeze(0).unsqueeze(0)
            W = torch.linalg.inv(I-prod_UHV)
            
            HH = torch.conj(torch.transpose(H, 2, 3))
            B = compute_B(U, H, W, self.Pmax, self.sigma, device, dtype)
            B_inv = torch.linalg.inv(B)
            V = torch.einsum("acd,abde,abef,abfg->abcg", B_inv, HH, U, W)
        
        return W,U,V,rec

def compute_A(V, H, Pmax, sigma, device, dtype):
    VH = torch.conj(torch.transpose(V, 2, 3))
    HH = torch.conj(torch.transpose(H, 2, 3))
    sumk_VkVkH = torch.einsum("abcd,abde->ace", V, VH)
    
    tr_sumk_VkVkH = torch.einsum("acc->a", sumk_VkVkH)
    
    tr_sumk_VkVkH = tr_sumk_VkVkH.unsqueeze(1).unsqueeze(1)
    I = torch.eye(H.size(2), device=device, dtype = dtype).repeat(H.size(0), 1, 1)
    first_term = (sigma**2/Pmax*tr_sumk_VkVkH*I).unsqueeze(1)
    second_term = torch.einsum("abcd,ade,abef->abcf", H, sumk_VkVkH, HH)
    
    return first_term+second_term
    
def compute_B(U, H, W, Pmax, sigma, device, dtype):
    UH = torch.conj(torch.transpose(U, 2, 3))
    HH = torch.conj(torch.transpose(H, 2, 3))
    
    sum_UWUH = torch.einsum("abcd,abde,abef->acf", U, W, UH)
    
    tr_sum_UWUH = torch.einsum("acc->a", sum_UWUH)
    tr_sum_UWUH = tr_sum_UWUH.unsqueeze(1).unsqueeze(1)
    
    I = torch.eye(H.size(3), device=device, dtype = dtype).repeat(H.size(0), 1, 1)
    first_term =  (sigma**2/Pmax*tr_sum_UWUH*I)
    
    second_term = torch.einsum("abcd,abde,abef,abfg,abgh->ach", HH, U, W, UH, H)
    
    return first_term+second_term
    