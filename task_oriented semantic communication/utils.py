# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:48:49 2022

@author: Yanzhen
"""
import torch
import math
import numpy as np
    
def compute_distribution_loss(gate1, gate2, FID, device):
    N,L = gate1.shape
    # generate FID matrix
    base = torch.ones([N, N],device = device)
    Y1 = torch.cat([base, base*FID], dim=1)
    Y2 = torch.cat([base*FID, base], dim=1)
    Y = torch.cat([Y1, Y2], dim=0)

    # generate Inner product matrix
    gate = torch.cat((gate1, gate2), dim=0)
    cosine = compute_cosine(gate)
    loss = (cosine-Y)**2
    return loss.mean()

def compute_sparsity_loss(gate, level):
    sparsity_level = torch.sum(gate, dim=1) - level
    loss = torch.mean(sparsity_level**2)
    
    return loss

def compute_cosine(gate):
    eps = 1e-10
    inner_product = gate.mm(gate.t())
    f_norm = torch.norm(gate, p='fro', dim=1, keepdim=True)
    outter_product = f_norm.mm(f_norm.t())
    cosine = inner_product / (outter_product + eps)

    return cosine

