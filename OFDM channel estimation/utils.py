# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:41:10 2024

@author: yl5922
"""

import numpy as np
import math
from scipy import interpolate
from scipy.io import loadmat, savemat
import torch

def interpolation(noisy, Number_of_pilot, interp):

    if (Number_of_pilot == 48):
        #idx = [14*i for i in range(1, 72,6)]+[4+14*(i) for i in range(4, 72,6)]+[7+14*(i) for i in range(1, 72,6)]+[11+14*(i) for i in range(4, 72,6)]
        idx = [14*i for i in range(0, 72,6)]+[4+14*(i) for i in range(2, 72,6)]+[8+14*(i) for i in range(3, 72,6)]+[12+14*(i) for i in range(5, 72,6)]
    elif (Number_of_pilot == 16):
        idx= [4+14*(i) for i in range(1, 72,9)]+[9+14*(i) for i in range(4, 72,9)]
    elif (Number_of_pilot == 24):
        idx = [14*i for i in range(1,72,9)]+ [6+14*i for i in range(4,72,9)]+ [11+14*i for i in range(1,72,9)]
    elif (Number_of_pilot == 8):
      idx = [4+14*(i) for  i in range(5,72,18)]+[9+14*(i) for i in range(8,72,18)]
    elif (Number_of_pilot == 36):
      idx = [14*(i) for  i in range(1,72,6)]+[6+14*(i) for i in range(4,72,6)] + [11+14*i for i in range(1,72,6)]

    r = [x//14 for x in idx]
    c = [x%14 for x in idx]

    interp_noisy = np.zeros((noisy.shape[0],72,14,2))

    for i in range(len(noisy)):
        #z = [noisy_image[i,j,k,0] for j,k in zip(r,c)]
        z = noisy[i,0].T.reshape(-1)
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,0] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,0] = z_intp
        #z = [noisy_image[i,j,k,1] for j,k in zip(r,c)]
        z = noisy[i,1].T.reshape(-1)
        if(interp == 'rbf'):
            f = interpolate.Rbf(np.array(r).astype(float), np.array(c).astype(float), z,function='gaussian')
            X , Y = np.meshgrid(range(72),range(14))
            z_intp = f(X, Y)
            interp_noisy[i,:,:,1] = z_intp.T
        elif(interp == 'spline'):
            tck = interpolate.bisplrep(np.array(r).astype(float), np.array(c).astype(float), z)
            z_intp = interpolate.bisplev(range(72),range(14),tck)
            interp_noisy[i,:,:,1] = z_intp


    interp_noisy = np.stack([interp_noisy[:,:,:,0], interp_noisy[:,:,:,1]], axis=1)
    interp_noisy = torch.tensor(interp_noisy)
    
    return interp_noisy


